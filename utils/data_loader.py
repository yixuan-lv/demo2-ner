# utils/data_loader.py
import os
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class NERDataset(Dataset):
    """NER 数据集，只负责存储和索引数据"""

    def __init__(self, input_ids: List[List[int]],
                 attention_masks: List[List[int]],
                 label_ids: List[List[int]]):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.label_ids = label_ids
        self.length = len(input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.label_ids[idx]
        }


def load_bio_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    读取 BIO 格式的原始数据

    参数:
        file_path: 数据文件路径

    返回:
        (sentences, labels): 每个句子由词列表和标签列表组成
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    sentences = []
    labels = []
    cur_words = []
    cur_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_words:
                    sentences.append(cur_words)
                    labels.append(cur_labels)
                    cur_words = []
                    cur_labels = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    cur_words.append(parts[0])
                    cur_labels.append(parts[1])
                else:
                    print(f"警告: 跳过格式异常的行: {line}")

    if cur_words:
        sentences.append(cur_words)
        labels.append(cur_labels)

    return sentences, labels


def align_labels_with_tokens(word_ids, tags, label2id):
    """
    将原始标签与 BERT tokenizer 分词后的 token 对齐

    参数:
        word_ids: tokenizer.word_ids() 返回的列表
        tags: 原始标签列表
        label2id: 标签到 ID 的映射字典

    返回:
        对齐后的 label_ids 列表（-100 表示忽略的 token）
    """
    label_ids = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label2id.get(tags[word_idx], 0))
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def load_ner_data(data_path: str, tokenizer: PreTrainedTokenizer,
                  label2id: Dict[str, int], max_len: int = 128) -> NERDataset:
    """
    从文件读取 NER 数据并进行预处理
    """
    sentences, labels = load_bio_data(data_path)

    if len(sentences) == 0:
        raise ValueError(f"数据文件 {data_path} 为空或格式不正确")

    input_ids_list = []
    attention_masks_list = []
    label_ids_list = []

    for words, tags in zip(sentences, labels):
        if len(words) != len(tags):
            print(f"警告: 词数({len(words)})与标签数({len(tags)})不匹配，跳过该样本")
            continue

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            padding="False"
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        word_ids = encoding.word_ids()

        label_ids = align_labels_with_tokens(word_ids, tags, label2id)

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        label_ids_list.append(label_ids)

    print(f"成功加载 {len(input_ids_list)} 个样本")

    return NERDataset(input_ids_list, attention_masks_list, label_ids_list)


def dynamic_pad_collate_fn(batch, pad_token_id: int = 0, label_pad_id: int = -100):
    """
    按 batch 内最大长度动态 padding
    """
    import torch

    max_len = max([len(item['input_ids']) for item in batch])

    input_ids_list = []
    attention_mask_list = []
    label_ids_list = []

    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len

        input_ids = item['input_ids'] + [pad_token_id] * pad_len
        attention_mask = item['attention_mask'] + [0] * pad_len
        labels = item['labels'] + [label_pad_id] * pad_len

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        label_ids_list.append(labels)

    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
        'labels': torch.tensor(label_ids_list, dtype=torch.long)
    }


def create_dataloader(data_path: str, tokenizer, label2id: Dict[str, int],
                      batch_size: int, max_len: int = 128, shuffle: bool = True,
                      use_dynamic_padding: bool = True):
    """
    创建 DataLoader（一站式便捷函数）
    """
    from torch.utils.data import DataLoader

    dataset = load_ner_data(data_path, tokenizer, label2id, max_len)

    if use_dynamic_padding:
        collate_fn = dynamic_pad_collate_fn
    else:
        collate_fn = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def analyze_data_statistics(data_path: str, tokenizer=None,
                            label2id: Dict[str, int] = None,
                            max_len: int = 128):
    """
    分析数据集统计信息
    """
    from collections import Counter

    sentences, labels = load_bio_data(data_path)
    orig_lengths = [len(s) for s in sentences]

    print(f"\n{'=' * 60}")
    print(f"📊 数据集统计: {os.path.basename(data_path)}")
    print(f"{'=' * 60}")
    print(f"总样本数: {len(sentences)}")
    print(f"平均长度: {sum(orig_lengths) / len(orig_lengths):.2f}")
    print(f"最大长度: {max(orig_lengths)}")
    exceeded = sum([1 for l in orig_lengths if l > max_len])
    print(f"超过 max_len={max_len} 的样本: {exceeded} ({exceeded / len(sentences) * 100:.2f})%")