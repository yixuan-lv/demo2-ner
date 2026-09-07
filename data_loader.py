"""BIO data loading, tokenisation and batching for NER training.

``NERDataModule`` deliberately combines the old dataset and data-module
responsibilities.  It owns the parsed/tokenised samples for every split and
exposes the usual ``*_dataloader`` helpers, while still implementing the
``Dataset`` protocol for the training split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

logger = logging.getLogger(__name__)


class NERDataModule(Dataset):
    """Combined BIO dataset and data-module implementation."""

    REQUIRED_SPLITS = ("train", "dev", "test")

    def __init__(
        self,
        data_paths: Mapping[str, str],
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_length: int,
        num_workers: int = 0,
    ) -> None:
        missing = [split for split in self.REQUIRED_SPLITS if not data_paths.get(split)]
        if missing:
            raise ValueError(f"Missing data paths for: {', '.join(missing)}")

        self.data_paths = dict(data_paths)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.label2id, self.id2label = self._build_label_mappings(self.data_paths["train"])
        self.datasets: Dict[str, List[Dict[str, List[int]]]] = {}
        self._collator = self._build_collator()

    @property
    def num_labels(self) -> int:
        return len(self.label2id)

    def setup(self) -> None:
        self.datasets = {split: self._load_split(path) for split, path in self.data_paths.items()
                         if split in self.REQUIRED_SPLITS}
        logger.info(
            "Dataset sizes: train=%d, dev=%d, test=%d",
            *(len(self.datasets[split]) for split in self.REQUIRED_SPLITS),
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train", shuffle=True)

    def dev_dataloader(self) -> DataLoader:
        return self._dataloader("dev", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader("test", shuffle=False)

    def _dataloader(self, split: str, shuffle: bool) -> DataLoader:
        if split not in self.datasets:
            raise RuntimeError("Call setup() before requesting a data loader")
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collator,
            pin_memory=torch.cuda.is_available(),
        )

    def _build_collator(self):
        from transformers import DataCollatorForTokenClassification

        return DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

    def _load_split(self, data_path: str) -> List[Dict[str, List[int]]]:
        path = Path(data_path)
        sentences, labels = self.read_bio(path)
        if not sentences:
            raise ValueError(f"No valid samples found in {path}")

        samples: List[Dict[str, List[int]]] = []
        for words, tags in zip(sentences, labels):
            encoding = self.tokenizer(words, is_split_into_words=True,
                                      truncation=True, max_length=self.max_length,
                                      padding=False)
            samples.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": self._align_labels(encoding.word_ids(), tags, self.label2id),
            })

        lengths = [len(sentence) for sentence in sentences]
        logger.info("%s: %d samples, average length %.2f, max length %d, %d over max_length=%d",
                    path.name, len(samples), sum(lengths) / len(lengths), max(lengths),
                    sum(length > self.max_length for length in lengths), self.max_length)
        return samples

    def __len__(self) -> int:
        return len(self.datasets.get("train", ()))

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        if "train" not in self.datasets:
            raise RuntimeError("Call setup() before requesting a sample")
        return self.datasets["train"][index]

    @staticmethod
    def _build_label_mappings(train_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        _, sequences = NERDataModule.read_bio(train_path)
        unique_labels = {label for sequence in sequences for label in sequence}
        ordered_labels = (["O"] if "O" in unique_labels else []) + sorted(unique_labels - {"O"})
        if not ordered_labels:
            raise ValueError(f"No labels found in training data: {train_path}")
        label2id = {label: index for index, label in enumerate(ordered_labels)}
        return label2id, {index: label for label, index in label2id.items()}


# Backwards-compatible import for callers of the former two-class API.
NERDataset = NERDataModule
