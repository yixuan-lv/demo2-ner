import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_bio_data(file_path):
    """读取 BIO 格式数据"""
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

    if cur_words:
        sentences.append(cur_words)
        labels.append(cur_labels)

    return sentences, labels


def extract_entities(labels):
    entities = set()
    for start, label in enumerate(labels):
        if label.startswith("B-"):
            entity_type = label[2:]
        elif label.startswith("I-"):
            entity_type = label[2:]
            # I-* 前面没有同类型实体时，按 B-* 处理
            if start > 0 and labels[start - 1] in {
                f"B-{entity_type}",
                f"I-{entity_type}"
            }:
                continue
        else:
            continue
        end = start
        while end + 1 < len(labels) and labels[end + 1] == f"I-{entity_type}":
            end += 1
        entities.add((entity_type, start, end))
    return entities


class NERMetrics:

    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, predictions, labels):
        predictions, labels = predictions.cpu(), labels.cpu()
        for predicted_ids, target_ids in zip(predictions, labels):
            valid = target_ids.ne(-100)
            self.predictions.append([self.id2label[int(value)] for value in predicted_ids[valid]])
            self.targets.append([self.id2label[int(value)] for value in target_ids[valid]])

    def compute(self):
        targets, predictions = set(), set()
        for sentence, (target, prediction) in enumerate(zip(self.targets, self.predictions)):
            targets.update((sentence, *entity) for entity in extract_entities(target))
            predictions.update((sentence, *entity) for entity in extract_entities(prediction))
        correct = len(targets & predictions)
        precision = correct / len(predictions) if predictions else 0
        recall = correct / len(targets) if targets else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        report = f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}"
        return {"precision": precision, "recall": recall, "f1": f1, "report": report}
