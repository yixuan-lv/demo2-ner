# utils/__init__.py
"""
工具模块

包含:
    - data_loader: 数据加载与预处理
    - metrics: NER 评估指标
    - analyzer: 数据分析工具
"""

from .data_loader import (
    NERDataset,
    load_bio_data,
    align_labels_with_tokens,
    load_ner_data,
    dynamic_pad_collate_fn,
    create_dataloader,
    analyze_data_statistics
)

from .metrics import (
    extract_entities,
    compute_ner_f1,
    compute_ner_classification_report,
    get_ner_f1_score,
    align_predictions
)

from .analyzer import (
    analyze_seq_lengths,
    analyze_label_distribution
)

__all__ = [
    # data_loader
    'NERDataset',
    'load_bio_data',
    'align_labels_with_tokens',
    'load_ner_data',
    'dynamic_pad_collate_fn',
    'create_dataloader',
    'analyze_data_statistics',
    # metrics
    'extract_entities',
    'compute_ner_f1',
    'compute_ner_classification_report',
    'get_ner_f1_score',
    'align_predictions',
    # analyzer
    'analyze_seq_lengths',
    'analyze_label_distribution'
]