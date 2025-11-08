import argparse
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    PreTrainedModel
)

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.llama.modeling_llama import LlamaModel
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import math

# --- QST 核心架构实现 ---

class AdapterModule(nn.Module):
    """
    论文中推荐的 Downsampler 实现 (Section 4.6, Table 6) [cite: 425]
    这是一个瓶颈(bottleneck)结构的适配器：Linear -> Activation -> Linear
    """
    def __init__(self, in_features, out_features, bottleneck_dim, activation=nn.GELU()):
        super().__init__()
        self.down_proj = nn.Linear(in_features, bottleneck_dim)
        self.activation = activation
        self.up_proj = nn.Linear(bottleneck_dim, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        
    def forward(self, x):
        x_down = self.down_proj(x)
        x_act = self.activation(x_down)
        x_up = self.up_proj(x_act)
        return self.layer_norm(x_up)

class QSTLlamaForSequenceClassification(PreTrainedModel):
    """
    QST (Quantized Side Tuning) 论文架构的完整实现
    
    该模型包含:
    1. base_model (f): 冻结的 4-bit 量化 Llama 模型 [cite: 10]
    2. side_network (g): 一个小型的、可训练的 BF16 Llama 模型 [cite: 10]
    3. downsamplers: N个适配器模块，将 f 的输出维度降低到 g 的输入维度 [cite: 217]
    4. upsampler: 1个线性层，将 g 的最终输出恢复到 f 的维度 [cite: 224]
    5. gates (alpha, betas): 可训练的门控参数 [cite: 213, 216]
    """
    config_class = AutoConfig # 告诉 Hugging Face 这是一个 PreTrainedModel

    def __init__(self, config, base_model_4bit, reduction_factor_r=16, adapter_rank_r=16):
        super().__init__(config)
        
        # 1. 主网络 (f) - 冻结的 4-bit Llama 
        # self.base_model = base_model_4bit
        
        # 2. QST 参数
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.d_side = self.d_model // reduction_factor_r 
        
        print(f"[QST] 主网络 (f) d_model: {self.d_model}")
        print(f"[QST] 侧网络 (g) d_side: {self.d_side} (r={reduction_factor_r})")
        print(f"[QST] Downsampler 秩: {adapter_rank_r}")

        # 3. 侧网络 (g) - 可训练的 BF16 Llama [cite: 10]
        side_config = AutoConfig.from_pretrained(config._name_or_path, trust_remote_code=True)
        side_config.hidden_size = self.d_side
        side_config.num_hidden_layers = self.num_layers
        side_config.intermediate_size = side_config.intermediate_size // reduction_factor_r
        # 我们创建一个新的LlamaModel作为侧网络，但不加载其预训练权重
        self.side_network = LlamaModel(side_config)
        
        # 4. Downsamplers (N个层 + 1个嵌入层)
        # 论文提到也下采样嵌入层 [cite: 216]
        self.downsampler_embed = AdapterModule(self.d_model, self.d_side, adapter_rank_r)
        self.downsamplers_layers = nn.ModuleList(
            [AdapterModule(self.d_model, self.d_side, adapter_rank_r) for _ in range(self.num_layers)]
        )
        
        # 5. Upsampler [cite: 224]
        self.upsampler = nn.Linear(self.d_side, self.d_model)
        
        # 6. Gating 参数 [cite: 213, 216]
        # betas: 每一层的混合权重，初始化为0 [cite: 216]
        self.betas = nn.Parameter(torch.zeros(self.num_layers))
        # alpha: 最终输出的混合权重，初始化为1 [cite: 214]
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        # 7. 分类头 (我们复用 base_model 的分类头)
        self.classifier = base_model_4bit.score

        # === 关键修复 ===
        # 同样, 提取 base_model 的核心 LlamaModel, 
        # 以避免 __getattr__ 冲突
        self.base_llama_model = base_model_4bit.model
        # === 修复结束 ===

        # 1. 主网络 (f) - 现在再赋值
        self.base_model = base_model_4bit
        
        # 8. 冻结主网络并解冻QST组件
        self.freeze_base_model_and_enable_qst()
        
        # 9. 关键修复: 伪装成PEFT模型绕过Trainer检查
        self._hf_peft_config_loaded = True
    
    def save_pretrained(self, save_directory, **kwargs):
        """自定义保存方法，只保存QST侧网络参数"""
        import os, torch, json
        os.makedirs(save_directory, exist_ok=True)
        qst_state_dict = {name: param.cpu() for name, param in self.named_parameters() if param.requires_grad}
        torch.save(qst_state_dict, os.path.join(save_directory, "qst_adapter.bin"))
        json.dump({"model_type": "qst_llama", "num_labels": self.config.num_labels, "d_model": self.d_model, "d_side": self.d_side}, open(os.path.join(save_directory, "qst_config.json"), "w"), indent=2)
        print(f"✅ QST侧网络已保存到: {save_directory} ({len(qst_state_dict)} 参数)")
        print("[QST] 模型初始化完成，主网络已冻结。")

    # ... 在 __init__ 方法结束后 ...

    @staticmethod
    def _prepare_4d_causal_attention_mask(attention_mask, input_shape, dtype, device, past_key_values_length=0):
        """
        在本地复现 transformers 内部的掩码创建逻辑
        """
        bsz, tgt_len = input_shape

        # [bsz, 1, tgt_len, tgt_len]
        # 创建一个填充了极小值（表示-inf）的掩码
        causal_mask = torch.full((bsz, 1, tgt_len, tgt_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        
        # 创建因果（causal）部分
        # 我们需要 causal_mask[b, 0, i, j] = 0.0 当 j <= i 时
        
        # 1. 创建一个 [tgt_len] 的张量: [0, 1, 2, ..., tgt_len-1]
        mask_cond = torch.arange(tgt_len, device=device)
        
        # 2. 创建一个 [tgt_len, tgt_len] 的布尔掩码，其中 bool_mask[i, j] = (j <= i)
        #    这是通过广播 (mask_cond < (mask_cond + 1).view(tgt_len, 1)) 实现的
        causal_bool_mask = mask_cond < (mask_cond + 1).view(tgt_len, 1)
        
        # 3. 将 [tgt_len, tgt_len] 的布尔掩码应用到 [bsz, 1, tgt_len, tgt_len] 的 causal_mask
        #    布尔掩码会自动广播到正确的维度
        causal_mask.masked_fill_(causal_bool_mask.bool(), 0.0)
        

        if past_key_values_length > 0:
            causal_mask[..., :, :past_key_values_length] = 0.0

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [bsz, seq_len] -> [bsz, 1, 1, seq_len]
                attention_mask = attention_mask[:, None, None, :]

            # [bsz, 1, 1, seq_len] -> [bsz, 1, tgt_len, tgt_len]
            attention_mask = attention_mask.expand((bsz, 1, tgt_len, tgt_len))
            # 将 padding 掩码 (attention_mask == 0) 应用到因果掩码
            causal_mask = causal_mask.masked_fill(attention_mask == 0, torch.finfo(dtype).min)

        return causal_mask

    # ... get_input_embeddings 方法开始的地方 ...

    def freeze_base_model_and_enable_qst(self):
        # 冻结所有 base_model 参数 [cite: 229]
        self.base_model.requires_grad_(False)
        # 确保 QST 组件是可训练的 (它们默认是)
        self.side_network.requires_grad_(True)
        self.downsampler_embed.requires_grad_(True)
        self.downsamplers_layers.requires_grad_(True)
        self.upsampler.requires_grad_(True)
        self.betas.requires_grad_(True)
        self.alpha.requires_grad_(True)
        # 解冻我们复用的分类头
        self.classifier.requires_grad_(True)
        
    def get_input_embeddings(self):
        return self.base_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.base_model.model.embed_tokens = value

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,  # <-- 1. 恢复 labels=None
        **kwargs  
    ):
        # 0. 准备侧网络的注意力掩码
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        side_attention_mask = self._prepare_4d_causal_attention_mask(
            attention_mask, 
            (batch_size, seq_length), 
            past_key_values_length=0, 
            dtype=self.side_network.dtype,
            device=input_ids.device
        )

        # 1. 运行主网络 (f) - 无梯度
        with torch.no_grad():
            base_outputs = self.base_llama_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
            base_hidden_states = base_outputs.hidden_states

        # 2. 运行侧网络 (g) - 有梯度
        h_f_0 = base_hidden_states[0]
        h_g_prev = self.downsampler_embed(h_f_0)
        
        side_position_embeddings = self.side_network.rotary_emb(h_g_prev, position_ids)
        
        for i in range(self.num_layers):
            h_f_i = base_hidden_states[i + 1]
            downsampled_h_f_i = self.downsamplers_layers[i](h_f_i)
            beta_i = torch.sigmoid(self.betas[i])
            side_input = (1 - beta_i) * downsampled_h_f_i + beta_i * h_g_prev
            
            layer_outputs = self.side_network.layers[i](
                side_input,
                attention_mask=side_attention_mask,
                position_embeddings=side_position_embeddings,
            )
            h_g_prev = layer_outputs[0]

        h_g_N = h_g_prev
        h_f_N = base_hidden_states[-1]
        
        final_hidden_state = self.alpha * h_f_N + (1 - self.alpha) * self.upsampler(h_g_N)

        # 4. 分类
        batch_size = input_ids.shape[0]
        if self.config.pad_token_id is None:
             sequence_lengths = -1
        else:
            sequence_lengths = (input_ids != self.config.pad_token_id).sum(-1) - 1
            
        last_token_hidden_states = final_hidden_state[torch.arange(batch_size, device=final_hidden_state.device), sequence_lengths]
        
        logits = self.classifier(last_token_hidden_states)

        # 5. 计算损失
        loss = None
        final_labels = labels if labels is not None else kwargs.get("labels")
        
        if final_labels is not None:
            if self.config.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), final_labels.squeeze())
            elif self.config.num_labels > 1:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), final_labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None, # 我们不返回隐藏状态以节省内存
            attentions=None,
        )


# --- 训练脚本 (与您的原代码类似) ---

DEFAULT_PAD_TOKEN = "[PAD]"

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def compute_metrics_sklearn(task, eval_pred):
    predictions, labels = eval_pred
    if task == "stsb":
        predictions = predictions[:, 0]
        pearson_corr = pearsonr(predictions, labels)[0]
        spearman_corr = spearmanr(predictions, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "combined": (pearson_corr + spearman_corr) / 2
        }
    else:
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        if task == "cola":
            mcc = matthews_corrcoef(labels, predictions)
            return {"matthews_correlation": mcc, "accuracy": acc, "f1": f1}
        elif task in ["mrpc", "qqp"]:
            return {"accuracy": acc, "f1": f1}
        else:
            return {"accuracy": acc, "f1": f1}

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"\n📊 参数统计:"
        f"\n  可训练参数: {trainable_params:,}"
        f"\n  总参数: {all_param:,}"
        f"\n  可训练比例: {100 * trainable_params / all_param:.4f}%"
    )

def train_qst_model(task, parameters):
    model_checkpoint = parameters["model_checkpoint"]
    batch_size = parameters["batch_size"]
    max_len = parameters["max_len"]
    epochs = parameters["epochs"]
    r = parameters.get("r", 16) # 论文默认r=16 [cite: 253]
    alpha_r = parameters.get("alpha_r", 16) # 论文中Downsampler的秩 [cite: 254]

    print("\n" + "="*60)
    print(f"QST (论文实现) 4-bit量化训练: {task}")
    print(f"模型: {model_checkpoint}, 侧网络r: {r}, Downsampler秩: {alpha_r}")
    print("="*60 + "\n")
    
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("nyu-mll/glue", actual_task)
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, trust_remote_code=True)
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    
    # 1. 4-bit 量化配置 [cite: 9, 76]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", # 论文推荐 NF4 [cite: 167, 254, 415]
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 将模型完全加载到单个GPU上
    compute_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"加载4-bit量化主网络 (f) 到 {compute_device}: {model_checkpoint}")
    base_model_4bit = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        device_map=compute_device, # 将整个模型放在一个设备上
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
        base_model_4bit.resize_token_embeddings(len(tokenizer))
    
    base_model_4bit.config.pad_token_id = tokenizer.pad_token_id
    
    # 2. 创建 QST 包装模型
    print("创建 QST 包装模型 (f + g)...")
    model = QSTLlamaForSequenceClassification(
        config=base_model_4bit.config,
        base_model_4bit=base_model_4bit,
        reduction_factor_r=r,
        adapter_rank_r=alpha_r
    )
    # 将新创建的 QST 组件 (侧网络等) 移动到 GPU
    model.to(compute_device, dtype=torch.bfloat16)

    print_trainable_parameters(model)
    
    # 3. 数据预处理
    sentence1_key, sentence2_key = task_to_keys[task]
    
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=max_len)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=max_len)
    
    print("数据预处理...")
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    
    # 4. 训练
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    args = TrainingArguments(
        f"llama3-qst-4bit-{task}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4, # 论文在 MMLU 上使用 2E-04 [cite: 671, 681]
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=False,
        fp16=False,
        bf16=True, # 必须使用 BF16 [cite: 254]
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics_sklearn(task, x)
    )
    
    print("🚀 开始 QST 训练...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_memory_gb = 0
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    
    print("\n📈 评估最终模型...")
    final_metrics = trainer.evaluate()
    final_metrics["peak_memory_gb"] = peak_memory_gb
    final_metrics["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_metrics["total_params"] = sum(p.numel() for p in model.parameters())
    final_metrics["trainable_ratio"] = (final_metrics["trainable_params"] / final_metrics["total_params"]) * 100
    return final_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QST (论文实现) + 4-bit量化训练")
    parser.add_argument("--model_checkpoint", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--batch_size", type=int, default=8) # 1B 模型可以尝试稍大的批量
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--task", type=str, default="sst2", help=f"GLUE 任务: {list(task_to_keys.keys())}")
    parser.add_argument("--r", type=int, default=16, help="侧网络缩减因子 (论文默认16)") # [cite: 253]
    parser.add_argument("--alpha_r", type=int, default=16, help="Downsampler 适配器秩 (论文默认16)") # [cite: 254]
    
    args = parser.parse_args()
    
    parameters = {
        "model_checkpoint": args.model_checkpoint,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "r": args.r,
        "alpha_r": args.alpha_r,
    }
    
    tasks = [args.task]
    
    results = {}
    for task in tasks:
        try:
            results[task] = train_qst_model(task, parameters)
        except Exception as e:
            print(f"\n❌ 任务 {task} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print("\n" + "="*60)
    print("训练完成! 结果:")
    print("="*60)
    for task, result in results.items():
        print(f"\n{task}:")
        for metric, value in result.items():
            print(f"  {metric}: {value:.4f}")