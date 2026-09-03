# utils/metrics.py
from typing import List, Tuple, Dict


def extract_entities_from_seq(seq: List[str]) -> List[Tuple[str, int, int]]:
    """
    从单条标签序列中提取实体
    输入: ['B-PER', 'I-PER', 'O', 'B-LOC']
    输出: [('PER', 0, 1), ('LOC', 3, 3)]
    """
    entities = []
    i = 0
    while i < len(seq):
        label = seq[i]
        if label.startswith('B-'):
            entity_type = label[2:]
            start = i
            i += 1
            while i < len(seq) and seq[i] == f'I-{entity_type}':
                i += 1
            entities.append((entity_type, start, i - 1))
        else:
            i += 1
    return entities


def compute_ner_f1(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
    """
    计算 NER 的 Precision, Recall, F1

    参数:
        y_true: 真实标签序列列表，如 [['B-PER', 'I-PER', 'O'], ['B-LOC', ...]]
        y_pred: 预测标签序列列表，格式同上

    返回:
        {'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'tp': 1, 'fp': 1, 'fn': 1}
    """
    tp = 0  # 正确预测的实体数
    fp = 0  # 预测多了的实体数（预测出来了但真实没有）
    fn = 0  # 漏掉的实体数（真实有但没预测出来）

    for true_seq, pred_seq in zip(y_true, y_pred):
        # 提取真实实体和预测实体
        true_entities = extract_entities_from_seq(true_seq)
        pred_entities = extract_entities_from_seq(pred_seq)

        # 转成集合方便比较
        true_set = set(true_entities)
        pred_set = set(pred_entities)

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)


    # 计算指标（防止除零）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def get_ner_f1_score(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    """
    获取 NER F1 分数（便捷函数）
    """
    result = compute_ner_f1(y_true, y_pred)
    return result['f1']


def align_predictions(predictions, labels, id2label: Dict[int, str]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    将预测的 ID 和标签 ID 对齐为标签字符串列表

    参数:
        predictions: 预测的 ID 数组
        labels: 真实的标签 ID 数组
        id2label: ID 到标签字符串的映射

    返回:
        (pred_labels, true_labels): 标签字符串列表
    """
    preds = []
    trues = []

    for pred, label in zip(predictions, labels):
        pred_seq = []
        true_seq = []

        for p, l in zip(pred, label):
            if l != -100:  # 跳过 padding 和特殊 token
                pred_seq.append(id2label.get(p, 'O'))
                true_seq.append(id2label.get(l, 'O'))

        preds.append(pred_seq)
        trues.append(true_seq)

    return preds, trues


def compute_ner_classification_report(y_true: List[List[str]], y_pred: List[List[str]]) -> str:
    """
    生成分类报告字符串（按实体类型分别统计）
    """
    # 收集所有实体类型
    all_types = set()
    for seq in y_true:
        for label in seq:
            if label.startswith('B-'):
                all_types.add(label[2:])
    for seq in y_pred:
        for label in seq:
            if label.startswith('B-'):
                all_types.add(label[2:])

    if not all_types:
        return "No entities found."

    # 按类型统计
    report_lines = []
    report_lines.append(f"{'Type':>12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    report_lines.append("-" * 54)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_support = 0

    for entity_type in sorted(all_types):
        tp = 0
        fp = 0
        fn = 0
        support = 0

        for true_seq, pred_seq in zip(y_true, y_pred):
            true_entities = [(t, s, e) for (t, s, e) in extract_entities_from_seq(true_seq) if t == entity_type]
            pred_entities = [(t, s, e) for (t, s, e) in extract_entities_from_seq(pred_seq) if t == entity_type]

            true_set = set(true_entities)
            pred_set = set(pred_entities)

            tp += len(true_set & pred_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)
            support += len(true_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report_lines.append(f"{entity_type:>12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_support += support

    # 加权平均
    if total_support > 0:
        weighted_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        weighted_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall) if (
                                                                                                                       weighted_precision + weighted_recall) > 0 else 0.0

        report_lines.append("-" * 54)
        report_lines.append(
            f"{'weighted_avg':>12} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {total_support:>10}")

    return "\n".join(report_lines)