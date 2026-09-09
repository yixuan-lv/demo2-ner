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


class DataCollator:
    """Pad tokenized samples and their token-level labels."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: bool | str = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("DataCollator received an empty batch")

        label_name = "labels" if "labels" in features[0] else "label"
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None
        model_features = [dict(feature) for feature in features]
        if labels is not None:
            for feature in model_features:
                feature.pop(label_name, None)

        batch = self.tokenizer.pad(
            model_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if labels is None:
            return batch

        input_ids = batch["input_ids"]
        sequence_length = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids[0])
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        padded_labels = []
        for label in labels:
            values = label.detach().cpu().tolist() if torch.is_tensor(label) else list(label)
            pad_size = sequence_length - len(values)
            if pad_size < 0:
                raise ValueError("A label sequence is longer than the padded input sequence")
            padding_values = [self.label_pad_token_id] * pad_size
            padded_labels.append(values + padding_values if padding_side == "right" else padding_values + values)

        if self.return_tensors == "pt":
            batch[label_name] = torch.tensor(padded_labels, dtype=torch.long)
        elif self.return_tensors == "np":
            import numpy as np

            batch[label_name] = np.asarray(padded_labels, dtype=np.int64)
        else:
            batch[label_name] = padded_labels
        return batch


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
        return DataCollator(
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
