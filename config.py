# config.py
import json
import torch
from dataclasses import dataclass
from typing import Dict


@dataclass
class Config:
    """配置类，从 JSON 文件加载，不写死任何参数"""

    # 所有字段都不设默认值，必须从 JSON 读取
    bert_path: str
    lstm_hidden_size: int
    lstm_layers: int
    dropout: float
    batch_size: int
    epochs: int
    warmup_ratio: float
    weight_decay: float
    max_seq_len: int
    max_grad_norm: float
    patience: int
    bert_lr: float
    lstm_lr: float
    classifier_lr: float
    seed: int
    output_dir: str
    log_dir: str

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """从 JSON 文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: str) -> None:
        """保存配置到 JSON 文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, **kwargs) -> None:
        """更新配置参数（用于命令行覆盖）"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")