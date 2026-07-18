
import torch
from transformers import AutoTokenizer


def build_label_map(train_data_path):
    """从训练数据中构建标签映射，自动识别所有标签"""
    labels = set()
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:  # 非空行且包含标签
                parts = line.split()
                if len(parts) == 2:
                    _, label = parts
                    labels.add(label)

    labels = sorted(list(labels))
    # 如果 'O' 在标签中，将其移到第一位
    if 'O' in labels:
        labels.remove('O')
        labels = ['O'] + labels

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def load_bio_data(file_path):
    """加载 BIO 格式的数据，返回句子列表和标签列表"""
    sentences = []
    labels_list = []
    words = []
    tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    word, tag = parts
                    words.append(word)
                    tags.append(tag)
            else:  # 空行表示一个句子结束
                if words:
                    sentences.append(words)
                    labels_list.append(tags)
                    words = []
                    tags = []
        # 处理文件末尾没有空行的情况
        if words:
            sentences.append(words)
            labels_list.append(tags)

    return sentences, labels_list


def get_entity_labels(label2id):
    """获取所有非 O 的标签"""
    entity_labels = []
    for label in label2id.keys():
        if label != 'O':
            entity_labels.append(label)
    return entity_labels