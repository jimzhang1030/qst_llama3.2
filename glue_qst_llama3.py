#!/usr/bin/env python3
"""
QST GLUE Benchmark - 完整版
自动收集统计数据并导出到Excel
"""

import sys
import os

# 首先导入train_qst_with_stats中的函数
exec(open('train_qst_with_stats.py').read().split('if __name__')[0])

import pandas as pd
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QST GLUE Benchmark with Excel Export")
    parser.add_argument("--task", type=str, default=None, help="运行单个任务，或不指定则运行所有任务")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认epochs")
    args = parser.parse_args()
    
    # 任务列表
    if args.task:
        tasks = [args.task]
    else:
        # 默认运行所有8个GLUE任务
        tasks = ["rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"]
    
    parameters = {
        "model_checkpoint": "meta-llama/Llama-3.2-1B",
        "batch_size": 8,
        "max_len": 128,
        "epochs": args.epochs if args.epochs else 3,
        "r": 16,
        "alpha_r": 16,
    }
    
    results = {}
    print(f"\n🚀 开始运行 {len(tasks)} 个GLUE任务...")
    print("="*60)
    
    for idx, task in enumerate(tasks, 1):
        print(f"\n[{idx}/{len(tasks)}] 当前任务: {task.upper()}")
        try:
            metrics = train_qst_model(task, parameters)
            results[task] = metrics
            print(f"✅ {task.upper()} 完成 - 准确率: {metrics.get('eval_accuracy', 0)*100:.2f}%")
        except Exception as e:
            print(f"❌ {task.upper()} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印结果
    print("\n" + "="*60)
    print("训练完成! 结果汇总:")
    print("="*60)
    
    excel_data = []
    for task, result in results.items():
        print(f"\n{task.upper()}:")
        for metric, value in result.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        # 收集Excel数据
        row = {
            '任务': task.upper(),
            '最佳准确率(%)': result.get('eval_accuracy', result.get('eval_pearson', 0)) * 100,
            '可训练参数占比(%)': result.get('trainable_ratio', 0),
            '显存峰值(GB)': result.get('peak_memory_gb', 0),
            '总参数': result.get('total_params', 0),
            '可训练参数': result.get('trainable_params', 0),
            'Loss': result.get('eval_loss', 0),
        }
        if 'eval_f1' in result:
            row['F1'] = result['eval_f1']
        if 'eval_matthews_correlation' in result:
            row['Matthews相关系数'] = result['eval_matthews_correlation']
        if 'eval_pearson' in result:
            row['Pearson相关系数'] = result['eval_pearson']
        if 'eval_spearmanr' in result:
            row['Spearman相关系数'] = result['eval_spearmanr']
        excel_data.append(row)
    
    # 导出到Excel
    if excel_data:
        df = pd.DataFrame(excel_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"QST_GLUE_Results_{timestamp}.xlsx"
        df.to_excel(filename, index=False, engine='openpyxl')
        
        print(f"\n{'='*60}")
        print(f"✅ 结果已导出到: {filename}")
        print(f"{'='*60}")
        print(f"\n📊 统计汇总:")
        print(f"  完成任务数: {len(df)}")
        print(f"  平均准确率: {df['最佳准确率(%)'].mean():.2f}%")
        print(f"  平均可训练参数占比: {df['可训练参数占比(%)'].mean():.4f}%")
        print(f"  平均显存峰值: {df['显存峰值(GB)'].mean():.2f} GB")
        print(f"  最大显存峰值: {df['显存峰值(GB)'].max():.2f} GB")
        print("\n详细表格:")
        print(df[['任务', '最佳准确率(%)', '可训练参数占比(%)', '显存峰值(GB)']].to_string(index=False))

