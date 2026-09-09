"""Project utilities and sequence-level NER metrics."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


class NERMetrics:
    """Accumulate predictions and score exact entity matches."""

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
        total = tuple(sum(values[index] for values in counts.values()) for index in range(3))
        precision, recall, f1 = _prf(*total)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total[0],
            "fp": total[1],
            "fn": total[2],
            "report": self._report(counts),
            "predictions": self.predictions,
            "labels": self.targets,
        }

    def _count_entities(self) -> Dict[str, Tuple[int, int, int]]:
        counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
        for target, prediction in zip(self.targets, self.predictions):
            actual = set(extract_entities(target))
            guessed = set(extract_entities(prediction))
            for entity_type in {item[0] for item in actual | guessed}:
                actual_type = {item for item in actual if item[0] == entity_type}
                guessed_type = {item for item in guessed if item[0] == entity_type}
                counts[entity_type][0] += len(actual_type & guessed_type)
                counts[entity_type][1] += len(guessed_type - actual_type)
                counts[entity_type][2] += len(actual_type - guessed_type)
        return {entity_type: tuple(values) for entity_type, values in counts.items()}

    @staticmethod
    def _report(counts: Mapping[str, Tuple[int, int, int]]) -> str:
        if not counts:
            return "No entities found."

        names = sorted(counts)
        width = max(12, max(map(len, names)))
        lines = [f"{'':>{width}}  {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}", ""]
        scores = []
        for name in names:
            tp, fp, fn = counts[name]
            precision, recall, f1 = _prf(tp, fp, fn)
            support = tp + fn
            scores.append((precision, recall, f1, support))
            lines.append(f"{name:>{width}}  {precision:>9.4f} {recall:>9.4f} {f1:>9.4f} {support:>9}")

        lines.append("")
        total = tuple(sum(values[index] for values in counts.values()) for index in range(3))
        micro = _prf(*total)
        macro = tuple(sum(score[index] for score in scores) / len(scores) for index in range(3))
        support_total = total[0] + total[2]
        if support_total:
            weighted = tuple(
                sum(score[index] * score[3] for score in scores) / support_total
                for index in range(3)
            )
        else:
            weighted = (0.0, 0.0, 0.0)
        lines.append(f"{'micro avg':>{width}}  {micro[0]:>9.4f} {micro[1]:>9.4f} {micro[2]:>9.4f} {support_total:>9}")
        lines.append(f"{'macro avg':>{width}}  {macro[0]:>9.4f} {macro[1]:>9.4f} {macro[2]:>9.4f} {support_total:>9}")
        lines.append(f"{'weighted avg':>{width}}  {weighted[0]:>9.4f} {weighted[1]:>9.4f} {weighted[2]:>9.4f} {support_total:>9}")
        return "\n".join(lines) + "\n"


Entity = Tuple[str, int, int]


def extract_entities(labels: Sequence[str]) -> List[Entity]:
    """Convert a label sequence to ``(type, start, end)`` chunks."""
    chunks: List[Entity] = []
    previous_tag, previous_type = "O", ""
    start = 0
    for index, label in enumerate(list(labels) + ["O"]):
        tag, entity_type = _split_tag(label)
        if _end_of_chunk(previous_tag, tag, previous_type, entity_type):
            chunks.append((previous_type, start, index - 1))
        if _start_of_chunk(previous_tag, tag, previous_type, entity_type):
            start = index
        previous_tag, previous_type = tag, entity_type
    return chunks


def _split_tag(label: str) -> Tuple[str, str]:
    if label == "O" or not label:
        return "O", "_"
    return label[0], label[1:].split("-", 1)[-1] or "_"


def _end_of_chunk(previous_tag: str, tag: str, previous_type: str, entity_type: str) -> bool:
    if previous_tag in {"E", "S"}:
        return True
    if (previous_tag, tag) in {("B", "B"), ("B", "S"), ("B", "O"),
                               ("I", "B"), ("I", "S"), ("I", "O")}:
        return True
    return previous_tag not in {"O", "."} and previous_type != entity_type


def _start_of_chunk(previous_tag: str, tag: str, previous_type: str, entity_type: str) -> bool:
    if tag in {"B", "S"}:
        return True
    if (previous_tag, tag) in {("E", "E"), ("E", "I"), ("S", "E"),
                               ("S", "I"), ("O", "E"), ("O", "I")}:
        return True
    return tag not in {"O", "."} and previous_type != entity_type


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
