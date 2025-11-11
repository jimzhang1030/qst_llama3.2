from datetime import datetime
import os, torch, random, argparse
import numpy as np
from torch import nn
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
import json
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import pandas as pd

# 任务特定超参数（优化版 - 修复准确率下降问题）

TASK_HYPERPARAMS = {
    # 小数据集任务：更小学习率，更多epoch
    "rte": {"epochs": 20, "batch_size": 8, "lr": 3e-4, "warmup_ratio": 0.06, "max_len": 256},
    "mrpc": {"epochs": 20, "batch_size": 8, "lr": 3e-4, "warmup_ratio": 0.06, "max_len": 256},
    "stsb": {"epochs": 20, "batch_size": 8, "lr": 3e-4, "warmup_ratio": 0.06, "max_len": 256},
    "cola": {"epochs": 20, "batch_size": 8, "lr": 3e-4, "warmup_ratio": 0.1, "max_len": 256},
    # 大数据集任务：稍大学习率
    "sst2": {"epochs": 5, "batch_size": 16, "lr": 3e-4, "warmup_ratio": 0.06, "max_len": 128},
    "qnli": {"epochs": 5, "batch_size": 8, "lr": 3e-4, "warmup_ratio": 0.06, "max_len": 256},
    "qqp": {"epochs": 5, "batch_size": 8, "lr": 3e-4, "warmup_ratio": 0.06, "max_len": 256},
    "mnli": {"epochs": 5, "batch_size": 16, "lr": 3e-4, "warmup_ratio": 0.06, "max_len": 256},
}

# --- QST 核心架构实现 (完全适配最新transformers + Llama3.2) ---
class QSTTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        """覆盖保存方法以避免 PEFT 集成问题"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取实际模型（处理 DataParallel 包装）
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 只保存可训练的 QST 组件
        qst_state_dict = {
            'qst_layers': model.qst_layers.state_dict(),
            'downsamplers_layers': model.downsamplers_layers.state_dict(),
            'downsampler_embed': model.downsampler_embed.state_dict(),
            'z': model.z,
            'score_z': model.score_z,
            'norm_qst': model.norm_qst.state_dict(),
            'upsampler': model.upsampler.state_dict(),
            'classifier': model.classifier.state_dict(),
        }
        
        # 保存 QST 组件（自定义格式）
        torch.save(qst_state_dict, os.path.join(output_dir, 'qst_components.bin'))
        
        # 🔥 关键修复：同时保存为 pytorch_model.bin，让 load_best_model_at_end 能找到
        torch.save(qst_state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
        
        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        
        # 🔥 新增：保存 trainer_state.json（Trainer 用来追踪最佳模型）
        # TrainerState 本身就是可序列化的，不需要 to_dict()
        if hasattr(self, 'state'):
            import json
            # TrainerState 可以直接使用 asdict 转换
            from dataclasses import asdict
            with open(os.path.join(output_dir, 'trainer_state.json'), 'w') as f:
                json.dump(asdict(self.state), f, indent=2)
        
        print(f"✅ QST 组件已保存到: {output_dir}")

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        覆盖 _save_checkpoint 以确保 load_best_model_at_end 正常工作
        """
        # 🔥 关键修复：调用父类的 _save_checkpoint，它会正确处理所有逻辑
        # 包括：保存模型、更新 best_model_checkpoint、清理旧 checkpoint
        checkpoint_folder = super()._save_checkpoint(model, trial)
        return checkpoint_folder

    def _load_best_model(self):
        """
        重写加载最佳模型的方法，使用我们自己的 QST checkpoint 格式
        """
        import os
        
        # 获取最佳 checkpoint 路径
        best_model_checkpoint = self.state.best_model_checkpoint
        
        if best_model_checkpoint is None:
            print("⚠️ 未找到最佳模型 checkpoint，使用当前模型")
            return
        
        print(f"\n🔄 加载最佳模型: {best_model_checkpoint}")
        
        # 使用我们自己的加载方法
        self._load_from_checkpoint(best_model_checkpoint)
        
        print(f"✅ 已成功加载最佳模型 (准确率: {self.state.best_metric:.4f})")

    def _load_from_checkpoint(self, resume_from_checkpoint):
        """覆盖加载方法以支持 QST checkpoint 格式"""
        import os
        import torch
        
        # 尝试加载 pytorch_model.bin（我们新保存的格式）
        checkpoint_path = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        
        if not os.path.exists(checkpoint_path):
            # 降级到 qst_components.bin
            checkpoint_path = os.path.join(resume_from_checkpoint, 'qst_components.bin')
        
        if os.path.exists(checkpoint_path):
            print(f"🔄 从 {checkpoint_path} 加载 QST checkpoint...")
            qst_state_dict = torch.load(checkpoint_path, map_location=self.model.device)
            
            # 获取实际模型
            model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # 加载各个组件
            model.qst_layers.load_state_dict(qst_state_dict['qst_layers'])
            model.downsamplers_layers.load_state_dict(qst_state_dict['downsamplers_layers'])
            model.downsampler_embed.load_state_dict(qst_state_dict['downsampler_embed'])
            model.z.data = qst_state_dict['z']
            model.score_z.data = qst_state_dict['score_z']
            model.norm_qst.load_state_dict(qst_state_dict['norm_qst'])
            model.upsampler.load_state_dict(qst_state_dict['upsampler'])
            model.classifier.load_state_dict(qst_state_dict['classifier'])
            
            print(f"✅ 成功加载 QST checkpoint")
        else:
            print(f"⚠️ 未找到 checkpoint 文件: {checkpoint_path}")

class AdapterModule(nn.Module):
    """论文推荐的Downsampler实现"""
    def __init__(self, in_features, out_features, bottleneck_dim, activation=nn.SiLU()):
        super().__init__()
        self.down_proj = nn.Linear(in_features, bottleneck_dim, bias=False)
        self.activation = activation
        self.up_proj = nn.Linear(bottleneck_dim, out_features, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=0, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_uniform_(self.up_proj.weight, a=0, mode="fan_in", nonlinearity="linear")
        self.layer_norm = nn.LayerNorm(out_features)
        
    
    def forward(self, x):
        x = self.dropout(x)
        x_down = self.down_proj(x)
        x_act = self.activation(x_down)
        x_up = self.up_proj(x_act)
        return self.layer_norm(x_up)

class QSTLlamaForSequenceClassification(PreTrainedModel):
    """
    QST (Quantized Side Tuning) - 完全适配最新transformers + Llama3.2
    
    关键修复:
    1. 使用backbone.forward()而不是直接调用layer (避免position_embeddings/cache_position问题)
    2. output_hidden_states=True获取所有层输出
    3. 侧网络layers是新创建的，可以直接调用
    """
    config_class = AutoConfig

    def __init__(self, config, base_model_4bit, reduction_factor_r=16, adapter_rank_r=16):
        super().__init__(config)
        
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.d_side = self.d_model // reduction_factor_r 
        self.num_labels = config.num_labels
        
        print(f"[QST] 主网络 (f) d_model: {self.d_model}")
        print(f"[QST] 侧网络 (g) d_side: {self.d_side} (r={reduction_factor_r})")
        print(f"[QST] Downsampler 秩: {adapter_rank_r}")

        # 侧网络配置
        side_config = AutoConfig.from_pretrained(config._name_or_path, trust_remote_code=True)
        side_config.hidden_size = self.d_side
        side_config.num_hidden_layers = self.num_layers
        side_config.intermediate_size = side_config.intermediate_size // reduction_factor_r
        side_num_heads = max(1, self.d_side // 64)
        side_config.num_attention_heads = side_num_heads
        side_config.num_key_value_heads = side_num_heads
        
        # 只保存侧网络的layers和norm
        side_network = LlamaModel(side_config)
        self.qst_layers = side_network.layers
        self.norm_qst = side_network.norm
        del side_network
        
        # Downsamplers
        self.downsampler_embed = AdapterModule(self.d_model, self.d_side, adapter_rank_r)
        self.downsamplers_layers = nn.ModuleList(
            [AdapterModule(self.d_model, self.d_side, adapter_rank_r) for _ in range(self.num_layers)]
        )
        
        # Upsampler
        self.upsampler = nn.Linear(self.d_side, self.d_model)
        
        # 门控参数
        self.z = nn.Parameter(torch.zeros(self.num_layers))
        self.score_z = nn.Parameter(torch.ones(self.d_model))
        
        # 分类头 & backbone
        self.classifier = base_model_4bit.score
        self.backbone = base_model_4bit.model
        self.base_model = base_model_4bit
        
        # 冻结主网络
        self.freeze_base_model_and_enable_qst()
        
        # 伪装成PEFT模型
        self._hf_peft_config_loaded = True
    
    def freeze_base_model_and_enable_qst(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.qst_layers.parameters():
            param.requires_grad = True
        for param in self.norm_qst.parameters():
            param.requires_grad = True
        for param in self.downsampler_embed.parameters():
            param.requires_grad = True
        for param in self.downsamplers_layers.parameters():
            param.requires_grad = True
        for param in self.upsampler.parameters():
            param.requires_grad = True
        self.z.requires_grad = True
        self.score_z.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        
    def get_input_embeddings(self):
        return self.backbone.embed_tokens

    def set_input_embeddings(self, value):
        self.backbone.embed_tokens = value

    def get_adapter_state_dict(self, *args, **kwargs):
        """Override to avoid PEFT integration bug - we don't use PEFT"""
        return {}
    
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        前向传播 - 完全适配最新transformers + Llama3.2
        
        核心策略: 使用backbone.forward()而不是手动调用每一层
        这样transformers会自动处理所有兼容性问题
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # ===== 关键修复: 使用LlamaModel.forward() =====
        # 这会自动处理cache_position, position_embeddings等所有兼容性问题
        with torch.no_grad():
            backbone_outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,  # 获取所有层的hidden_states
                return_dict=True,
            )
            
            hidden_states = backbone_outputs.last_hidden_state
            all_backbone_hidden_states = backbone_outputs.hidden_states  # tuple: (embed, layer1, ..., layerN)
        
        # 侧网络运行
        qst_hidden_states = self.downsampler_embed(all_backbone_hidden_states[0])  # embedding layer
        
        # 为侧网络创建position_embeddings (侧网络也需要RoPE)
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        if not hasattr(self, '_qst_rotary_emb'):
            # 懒初始化：最新版 transformers 只接受 config 参数
            config_copy = self.qst_layers[0].self_attn.config
            self._qst_rotary_emb = LlamaRotaryEmbedding(config=config_copy, device=qst_hidden_states.device)
        
        # 计算侧网络的position_embeddings
        qst_position_embeddings = self._qst_rotary_emb(qst_hidden_states, position_ids)
        
        # 为 QST 层准备 4D causal attention_mask 和 cache_position
        # 使用 transformers 内置方法确保格式正确
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
        
        if attention_mask is not None:
            # 创建 4D causal attention mask (与 LlamaModel 内部逻辑一致)
            attention_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=None)
            # to_4d 返回的 mask 已经在正确的设备上
            qst_attention_mask = attention_mask_converter.to_4d(
                attention_mask,
                query_length=qst_hidden_states.shape[1],
                key_value_length=qst_hidden_states.shape[1],  # causal mask 需要
                dtype=qst_hidden_states.dtype,
            )
        else:
            qst_attention_mask = None
        
        # 创建 cache_position (QST 层需要)
        cache_position = torch.arange(qst_hidden_states.shape[1], device=qst_hidden_states.device)
        
        for idx in range(self.num_layers):
            # Z门控
            z_gate = torch.sigmoid(self.z[idx])
            downsampled = self.downsamplers_layers[idx](all_backbone_hidden_states[idx + 1])
            qst_hidden_states = (1 - z_gate) * downsampled + z_gate * qst_hidden_states
            
            # 侧网络layer (需要传入position_embeddings和cache_position)
            layer_outputs = self.qst_layers[idx](
                qst_hidden_states,
                attention_mask=qst_attention_mask,  # 4D causal mask
                position_ids=position_ids,
                cache_position=cache_position,  # 最新 transformers 必需
                position_embeddings=qst_position_embeddings,  # RoPE embeddings
            )
            qst_hidden_states = layer_outputs[0]
        
        qst_hidden_states = self.norm_qst(qst_hidden_states)
        
        # 细粒度混合
        score_z_gate = torch.sigmoid(self.score_z)
        upsampled_side = self.upsampler(qst_hidden_states)
        final_hidden = (1 - score_z_gate) * upsampled_side + score_z_gate * hidden_states
        
        # 分类：使用最后一个有效token（Llama是因果模型，不是BERT）
        # 这是关键修复：原作者用最后一个token，你用的是第一个token！
        batch_size = input_ids.shape[0]
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            # 找到每个样本的最后一个非padding token
            sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(final_hidden.device)
            # 如果整行都不是padding，argmax返回0，此时应该用最后一个token
            sequence_lengths = torch.where(
                (attention_mask.sum(dim=1) == attention_mask.shape[1]),  # 没有padding
                torch.tensor(input_ids.shape[1] - 1, device=final_hidden.device),
                sequence_lengths
            )
        
        pooled_hidden = final_hidden[torch.arange(batch_size, device=final_hidden.device), sequence_lengths]
        logits = self.classifier(pooled_hidden)
        
        # 损失
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits}

# --- 训练脚本 ---

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
        predictions = predictions.squeeze()
        labels = labels.squeeze()
        pearson_corr = pearsonr(predictions, labels)[0]
        spearman_corr = spearmanr(predictions, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
    else:
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        if task == "cola":
            mcc = matthews_corrcoef(labels, predictions)
            return {"accuracy": accuracy, "matthews_correlation": mcc}
        elif task in ["mrpc", "qqp"]:
            f1 = f1_score(labels, predictions)
            return {"accuracy": accuracy, "f1": f1}
        else:
            return {"accuracy": accuracy}

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"\n📊 参数统计:"
        f"\n  可训练参数: {trainable_params:,}"
        f"\n  总参数: {all_param:,}"
        f"\n  可训练比例: {100 * trainable_params / all_param:.4f}%"
    )

def train_qst_model(task, parameters):
    model_checkpoint = parameters["model_checkpoint"]
    batch_size = parameters["batch_size"]
    max_len = parameters["max_len"]
    epochs = parameters["epochs"]
    r = parameters.get("r", 16)
    alpha_r = parameters.get("alpha_r", 16)
    learning_rate = parameters.get("learning_rate", 2e-4)
    seed = parameters.get("seed", 42)
    warmup_ratio = parameters.get("warmup_ratio", 0.06)
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TRANSFORMERS_SEED'] = str(seed)
    print(f"✅ 已设置随机种子: {seed} (确保结果可重复)")

    print("\n" + "="*60)
    print(f"QST (论文实现) 4-bit量化训练: {task}")
    print(f"模型: {model_checkpoint}, 侧网络r: {r}, Downsampler秩: {alpha_r}, 种子: {seed}")
    print("="*60 + "\n")
    
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("nyu-mll/glue", actual_task)
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, trust_remote_code=True)
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    
    # 4-bit量化配置
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    compute_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"加载4-bit量化主网络 (f) 到 {compute_device}: {model_checkpoint}")
    base_model_4bit = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        device_map=compute_device,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})
        base_model_4bit.resize_token_embeddings(len(tokenizer))
    
    base_model_4bit.config.pad_token_id = tokenizer.pad_token_id
    
    # 创建QST包装模型
    print("创建 QST 包装模型 (f + g)...")
    model = QSTLlamaForSequenceClassification(
        config=base_model_4bit.config,
        base_model_4bit=base_model_4bit,
        reduction_factor_r=r,
        adapter_rank_r=alpha_r
    )
    # 关键: 确保所有QST组件都是bfloat16
    model = model.to(compute_device, dtype=torch.bfloat16)

    print_trainable_parameters(model)
    
    # 数据预处理
    sentence1_key, sentence2_key = task_to_keys[task]
    
    def preprocess_function(examples):
        args = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        return tokenizer(*args, truncation=True, max_length=max_len, padding="max_length")
    
    print("数据预处理...")
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    
    # 训练
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    args = TrainingArguments(
        f"llama3-qst-4bit-{task}",
        eval_strategy="epoch",
        save_strategy="epoch",  # 临时禁用保存
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,  # 降低正则化
        load_best_model_at_end=True,
        dataloader_num_workers=2,
        metric_for_best_model=metric_name,
        push_to_hub=False,
        fp16=False,
        bf16=True,
        logging_steps=50,
        save_total_limit=2,
        max_grad_norm=1,  
        seed=seed,
        data_seed=seed,
        report_to="none",
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = QSTTrainer(
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
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    
    # ✅ 训练结束后，load_best_model_at_end=True 已自动加载最佳 checkpoint
    print("\n📈 评估最佳模型 (已自动加载 best checkpoint)...")
    final_metrics = trainer.evaluate()
    final_metrics["peak_memory_gb"] = peak_memory_gb
    final_metrics["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_metrics["total_params"] = sum(p.numel() for p in model.parameters())
    final_metrics["trainable_ratio"] = (final_metrics["trainable_params"] / final_metrics["total_params"]) * 100
    return final_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QST (论文实现) + 4-bit量化训练")
    parser.add_argument("--model_checkpoint", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--task", type=str, default="sst2", help=f"GLUE 任务: {list(task_to_keys.keys())}")
    parser.add_argument("--r", type=int, default=16, help="侧网络缩减因子 (论文默认16)")
    parser.add_argument("--alpha_r", type=int, default=16, help="Downsampler 适配器秩 (论文默认16)")
    parser.add_argument("--seed", type=int, default=68, help="随机种子 (确保结果可重复)")
    
    args = parser.parse_args()
    
    parameters = {
        "model_checkpoint": args.model_checkpoint,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "r": args.r,
        "alpha_r": args.alpha_r,
        "seed": args.seed,
    }
    
    tasks = [args.task]
    
    results = {}
    for task in tasks:
        try:
            results[task] = train_qst_model(task, parameters)
            print(f"\n✅ 任务 {task} 训练成功!")
            print(f"   准确率: {results[task].get('eval_accuracy', results[task].get('eval_pearson', 'N/A'))}")
        except Exception as e:
            print(f"\n❌ 任务 {task} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*60)
    print("训练完成! 结果:")
    print("="*60)
    for task, result in results.items():
        print(f"{task}: {result}")
