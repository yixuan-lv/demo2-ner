"""Project utilities and a self-contained entity-level NER metric."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)
Entity = Tuple[str, int, int]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


class NERMetrics:
    """Lightning-style reset/update/compute metric using exact entity matches."""

    def __init__(self, id2label: Mapping[int, str]) -> None:
        self.id2label = dict(id2label)
        self.reset()

    def reset(self) -> None:
        self.predictions: List[List[str]] = []
        self.targets: List[List[str]] = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        predictions, labels = predictions.detach().cpu(), labels.detach().cpu()
        if predictions.shape != labels.shape:
            raise ValueError(f"Shape mismatch: {tuple(predictions.shape)} vs {tuple(labels.shape)}")
        for predicted_ids, target_ids in zip(predictions, labels):
            valid = target_ids.ne(-100)
            self.predictions.append([self.id2label[int(value)] for value in predicted_ids[valid]])
            self.targets.append([self.id2label[int(value)] for value in target_ids[valid]])

    def compute(self) -> Dict[str, object]:
        if not self.targets:
            raise RuntimeError("No batches have been added to NERMetrics")
        counts = self._count_entities()
        tp = sum(value[0] for value in counts.values())
        fp = sum(value[1] for value in counts.values())
        fn = sum(value[2] for value in counts.values())
        precision, recall, f1 = _prf(tp, fp, fn)
        return {"precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "report": self._report(counts),
                "predictions": self.predictions, "labels": self.targets}

    def _count_entities(self) -> Dict[str, Tuple[int, int, int]]:
        counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
        for target, prediction in zip(self.targets, self.predictions):
            gold = set(extract_entities(target)); predicted = set(extract_entities(prediction))
            for entity_type in {item[0] for item in gold | predicted}:
                actual = {item for item in gold if item[0] == entity_type}
                guess = {item for item in predicted if item[0] == entity_type}
                counts[entity_type][0] += len(actual & guess)
                counts[entity_type][1] += len(guess - actual)
                counts[entity_type][2] += len(actual - guess)
        return {key: tuple(value) for key, value in counts.items()}

    @staticmethod
    def _report(counts: Mapping[str, Tuple[int, int, int]]) -> str:
        if not counts:
            return "No entities found."
        lines = [f"{'Type':>12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}", "-" * 54]
        total = [0, 0, 0]
        for entity_type in sorted(counts):
            tp, fp, fn = counts[entity_type]; precision, recall, f1 = _prf(tp, fp, fn)
            lines.append(f"{entity_type:>12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {tp + fn:>10}")
            total = [total[0] + tp, total[1] + fp, total[2] + fn]
        precision, recall, f1 = _prf(*total)
        lines.extend(["-" * 54, f"{'micro_avg':>12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {total[0] + total[2]:>10}"])
        return "\n".join(lines)


def extract_entities(labels: Sequence[str]) -> List[Entity]:
    """Convert BIO labels to exact-match (type, start, end) chunks."""
    entities: List[Entity] = []; active_type = None; start = None
    def close(end: int) -> None:
        nonlocal active_type, start
        if active_type is not None and start is not None: entities.append((active_type, start, end))
        active_type, start = None, None
    for index, label in enumerate(labels):
        prefix, separator, entity_type = label.partition("-")
        if label == "O" or not separator or prefix not in {"B", "I"} or not entity_type:
            close(index - 1); continue
        if prefix == "B" or active_type != entity_type:
            close(index - 1); active_type, start = entity_type, index
    if labels: close(len(labels) - 1)
    return entities


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def build_label_mappings(dataset_name: str, train_path: str):
    from data_loader import NERDataset
    _, sequences = NERDataset.read_bio(train_path)
    labels = sorted({label for sequence in sequences for label in sequence})
    labels = (["O"] if "O" in labels else []) + [label for label in labels if label != "O"]
    label2id = {label: index for index, label in enumerate(labels)}
    return label2id, {index: label for label, index in label2id.items()}, len(labels)


def build_label_map(train_path: str):
    return build_label_mappings("msra", train_path)[:2]
