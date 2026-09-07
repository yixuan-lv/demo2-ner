"""Project utilities and sequence-level NER metrics."""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Mapping

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
    """Lightning-style accumulator backed by :mod:`seqeval`.

    seqeval extracts BIO chunks and computes entity-level metrics directly;
    callers therefore do not need to maintain token/entity ``tp``, ``fp`` and
    ``fn`` counters themselves.
    """

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
        try:
            from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
        except ImportError as exc:
            raise RuntimeError(
                "seqeval is required to compute NER metrics; install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        return {
            "precision": float(precision_score(self.targets, self.predictions)),
            "recall": float(recall_score(self.targets, self.predictions)),
            "f1": float(f1_score(self.targets, self.predictions)),
            "report": classification_report(self.targets, self.predictions, digits=4),
            "predictions": self.predictions,
            "labels": self.targets,
        }


def build_label_mappings(dataset_name: str, train_path: str):
    from data_loader import NERDataset
    _, sequences = NERDataset.read_bio(train_path)
    labels = sorted({label for sequence in sequences for label in sequence})
    labels = (["O"] if "O" in labels else []) + [label for label in labels if label != "O"]
    label2id = {label: index for index, label in enumerate(labels)}
    return label2id, {index: label for label, index in label2id.items()}, len(labels)


def build_label_map(train_path: str):
    return build_label_mappings("msra", train_path)[:2]
