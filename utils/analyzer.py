# utils/analyzer.py
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import Counter
from .data_loader import load_bio_data  # ✅ 改这里：read_raw_data → load_bio_data


def analyze_seq_lengths(data_path: str, save_path: str = None) -> dict:
    """
    分析数据集中序列长度的分布

    参数:
        data_path: 数据文件路径
        save_path: 保存图片路径（可选）

    返回:
        包含统计信息的字典
    """
    sentences, _ = load_bio_data(data_path)  
    lengths = [len(s) for s in sentences]

    stats = {
        'total_samples': len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'p95_length': np.percentile(lengths, 95),
        'p99_length': np.percentile(lengths, 99),
        'over_128': sum([l > 128 for l in lengths])
    }

    print(f"数据集: {os.path.basename(data_path)}")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  最小长度: {stats['min_length']}")
    print(f"  最大长度: {stats['max_length']}")
    print(f"  平均长度: {stats['mean_length']:.2f}")
    print(f"  中位数长度: {stats['median_length']:.2f}")
    print(f"  95% 分位数: {stats['p95_length']:.0f}")
    print(f"  99% 分位数: {stats['p99_length']:.0f}")
    print(f"  超过128的样本数: {stats['over_128']} ({stats['over_128'] / stats['total_samples'] * 100:.2f}%)")

    # 绘制分布直方图
    if save_path:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(128, color='red', linestyle='--', linewidth=2, label='max_len=128')
        plt.axvline(stats['p95_length'], color='green', linestyle=':', linewidth=2, label='95% 分位数')
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Sequence Length Distribution - {os.path.basename(data_path)}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  分布图保存至: {save_path}")

    return stats


def analyze_label_distribution(data_path: str) -> dict:
    """
    分析数据集中标签的分布

    参数:
        data_path: 数据文件路径

    返回:
        标签统计信息
    """
    sentences, labels = load_bio_data(data_path)

    # 统计所有标签
    all_labels = []
    for seq_labels in labels:
        all_labels.extend(seq_labels)

    label_counts = Counter(all_labels)
    total = sum(label_counts.values())

    print(f"标签分布 - {os.path.basename(data_path)}")
    print(f"  总标签数: {total}")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({count / total * 100:.2f}%)")

    return dict(label_counts)