"""BIO dataset and data-module abstractions for NER training."""

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


class NERDataset(Dataset):
    """A tokenized BIO dataset representing one train, dev, or test split."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        label2id: Mapping[str, int],
        max_length: int = 128,
    ) -> None:
        self.data_path = Path(data_path)
        sentences, labels = self.read_bio(self.data_path)
        if not sentences:
            raise ValueError(f"No valid samples found in {self.data_path}")

        self.samples: List[Dict[str, List[int]]] = []
        for words, tags in zip(sentences, labels):
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            self.samples.append(
                {
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "labels": self._align_labels(encoding.word_ids(), tags, label2id),
                }
            )

        lengths = [len(sentence) for sentence in sentences]
        logger.info(
            "%s: %d samples, average length %.2f, max length %d, %d over max_length=%d",
            self.data_path.name,
            len(self.samples),
            sum(lengths) / len(lengths),
            max(lengths),
            sum(length > max_length for length in lengths),
            max_length,
        )

    @staticmethod
    def read_bio(data_path: Path | str) -> Tuple[List[List[str]], List[List[str]]]:
        """Read a token-per-line BIO file separated by blank lines."""
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"Data file does not exist: {path}")

        sentences: List[List[str]] = []
        label_sequences: List[List[str]] = []
        words: List[str] = []
        labels: List[str] = []

        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    if words:
                        sentences.append(words)
                        label_sequences.append(labels)
                        words, labels = [], []
                    continue

                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid BIO row at {path}:{line_number}: {line!r}")
                word, label = parts
                words.append(word)
                labels.append(label)

        if words:
            sentences.append(words)
            label_sequences.append(labels)
        return sentences, label_sequences

    @staticmethod
    def _align_labels(
        word_ids: Sequence[Optional[int]],
        labels: Sequence[str],
        label2id: Mapping[str, int],
    ) -> List[int]:
        aligned: List[int] = []
        previous_word_id: Optional[int] = None
        for word_id in word_ids:
            if word_id is None or word_id == previous_word_id:
                aligned.append(-100)
            else:
                aligned.append(label2id[labels[word_id]])
            previous_word_id = word_id
        return aligned

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.samples[index]


class NERDataModule:
    """Own label discovery, split datasets, dynamic padding, and data loaders."""

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
        self.datasets: Dict[str, NERDataset] = {}
        self._collator = self._build_collator()

    @property
    def num_labels(self) -> int:
        return len(self.label2id)

    def setup(self) -> None:
        self.datasets = {
            split: NERDataset(path, self.tokenizer, self.label2id, self.max_length)
            for split, path in self.data_paths.items()
            if split in self.REQUIRED_SPLITS
        }
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

    @staticmethod
    def _build_label_mappings(train_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        _, sequences = NERDataset.read_bio(train_path)
        unique_labels = {label for sequence in sequences for label in sequence}
        ordered_labels = (["O"] if "O" in unique_labels else []) + sorted(unique_labels - {"O"})
        if not ordered_labels:
            raise ValueError(f"No labels found in training data: {train_path}")
        label2id = {label: index for index, label in enumerate(ordered_labels)}
        return label2id, {index: label for label, index in label2id.items()}
