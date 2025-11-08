# QST GLUE Benchmark - 快速开始指南

## 📊 功能特性

本脚本 `glue_qst_llama3.py` 提供完整的GLUE benchmark测试，并自动收集以下统计数据：

1. **8个GLUE数据集的最佳准确率**
   - RTE, MRPC, STS-B, CoLA, SST-2, QNLI, QQP, MNLI (含MNLI-MM)

2. **参数统计**
   - 可训练参数数量
   - 总参数数量  
   - 可训练参数占比(%)

3. **显存使用**
   - 训练时显存峰值(GB)

4. **自动Excel导出**
   - 所有结果自动保存到带时间戳的Excel文件

## 🚀 使用方法

### 单个任务测试
```bash
# 快速测试RTE任务（1个epoch）
python glue_qst_llama3.py --task rte --epochs 1

# 指定模型训练
python glue_qst_llama3.py --model_checkpoint meta-llama/Llama-3.2-3B

# 完整训练SST-2任务
python glue_qst_llama3.py --task sst2

# 更多参数查看train_qst_with_stats.py中的 "__main__"
```

### 运行所有GLUE任务
```bash
# 使用任务专属超参数运行所有8个任务
python glue_qst_llama3.py
```

支持的任务：
- `rte` - Recognizing Textual Entailment
- `mrpc` - Microsoft Research Paraphrase Corpus
- `stsb` - Semantic Textual Similarity Benchmark
- `cola` - Corpus of Linguistic Acceptability
- `sst2` - Stanford Sentiment Treebank
- `qnli` - Question Natural Language Inference
- `qqp` - Quora Question Pairs
- `mnli` - Multi-Genre Natural Language Inference (自动包含MNLI-MM)

## 📈 输出说明

### 终端输出
脚本会实时显示：
- 训练进度和loss
- 每个任务的准确率
- 统计汇总表格

### Excel文件
自动生成文件名格式：`QST_GLUE_Results_YYYYMMDD_HHMMSS.xlsx`

包含列：
- **任务**: 任务名称
- **最佳准确率(%)**: 验证集准确率（×100）
- **可训练参数占比(%)**: 相对于总参数的百分比
- **显存峰值(GB)**: 训练时GPU显存使用峰值
- **总参数**: 模型总参数量
- **可训练参数**: QST侧网络可训练参数量
- **Loss**: 验证集损失
- **F1/Matthews/Pearson**: 任务特定指标

## ⚙️ 任务专属超参数

脚本使用以下优化后的超参数（来自论文和最佳实践）：

```python
TASK_HYPERPARAMS = {
    'cola': {'lr': 2e-4, 'epochs': 5, 'batch_size': 16},
    'sst2': {'lr': 2e-4, 'epochs': 3, 'batch_size': 32},
    'mrpc': {'lr': 3e-4, 'epochs': 5, 'batch_size': 16},
    'stsb': {'lr': 2e-4, 'epochs': 5, 'batch_size': 16},
    'qqp': {'lr': 2e-4, 'epochs': 3, 'batch_size': 32},
    'mnli': {'lr': 2e-4, 'epochs': 3, 'batch_size': 32},
    'qnli': {'lr': 2e-4, 'epochs': 3, 'batch_size': 32},
    'rte': {'lr': 3e-4, 'epochs': 5, 'batch_size': 16},
}
```

## 💾 环境要求

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- pandas, openpyxl
- NVIDIA GPU (8GB+ VRAM推荐)
- 见requirement_latest.txt

## 📋 示例输出

```
============================================================
✅ 结果已导出到: QST_GLUE_Results_20251108_032608.xlsx
============================================================

📊 统计汇总:
  完成任务数: 8
  平均准确率: 82.45%
  平均可训练参数占比: 3.9630%
  平均显存峰值: 1.59 GB
  最大显存峰值: 1.59 GB

详细表格:
   任务  最佳准确率(%)  可训练参数占比(%)  显存峰值(GB)
   RTE      60.65            3.96           1.59
  MRPC      85.29            3.96           1.59
  STSB      88.12            3.96           1.59
  COLA      75.36            3.96           1.59
  SST2      92.55            3.96           1.59
  QNLI      89.74            3.96           1.59
   QQP      87.23            3.96           1.59
  MNLI      83.41            3.96           1.59
MNLI-MM    82.95            3.96           1.59
```

## 🔧 技术细节

- **4-bit量化**: 使用NF4量化主网络，节省显存
- **QST架构**: 只训练3.96%的参数
- **优化器**: Cosine学习率调度，warmup=0.06
- **正则化**: weight_decay=0.01, max_grad_norm=1.0

## 📝 注意事项

1. 训练大型数据集(QQP, MNLI)需要更长时间
2. 建议按顺序运行任务，避免GPU内存碎片
3. Excel文件保存在当前目录
4. 每个任务的checkpoint保存在`llama3-qst-4bit-{task}/`目录

