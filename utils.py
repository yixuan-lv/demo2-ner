
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        total = len(targets) + len(predictions)
        f1 = 2 * len(targets & predictions) / total
        return {"f1": f1, "report": f"Entity-level F1: {f1:.4f}"}


def extract_entities(labels):
    entities = set()
    for start, label in enumerate(labels):
        if not label.startswith("B-"):
            continue
        entity_type = label[2:]
        end = start
        while end + 1 < len(labels) and labels[end + 1] == f"I-{entity_type}":
            end += 1
        entities.add((entity_type, start, end))
    return entities
