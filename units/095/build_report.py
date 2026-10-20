"""Render the saved, complete run table without re-training or selecting runs."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / 'outputs/results.json').read_text())
s = '''# 第095讲 当前实验项目报告

## 问题与方法

比较合成线段方向分类中的线性、16隐藏单元MLP和小CNN。固定128训练图、300测试图，100次全批Adam更新、学习率0.02，五个配对种子。未使用预训练。固定更新数不等于固定计算量。

## 完整结果

| 种子 | 模型 | 准确率 | 类0 | 类1 | 参数 | 训练秒 |
|---|---|---|---|---|---|---|
'''
for a in r['runs']:
    s += f"| {a['seed']} | {a['model']} | {a['accuracy']:.4f} | {a['class0_accuracy']:.4f} | {a['class1_accuracy']:.4f} | {a['parameters']} | {a['train_seconds']:.3f} |\n"
s += f"\n## 配对分析与限制\n\nCNN−MLP均值差{r['summary']['mean_difference']:.4f}，4000次种子bootstrap百分位区间{r['summary']['seed_bootstrap_percentile_95']}。只有五种子，固定测试集，无超参数搜索，不能据此宣称总体覆盖保证。\n\n"
s += '''## 错误与下一步

每次错误索引与逐类准确率保留在results.json，预测在predictions.npz；全部训练曲线在training_curves.json。下一步可预先指定相同训练时间比较、扩大独立训练数据抽样及按线位置分组。当前比较无法把参数量、结构和优化适配的影响完全分开。
'''
(ROOT / 'project_report.md').write_text(s)
