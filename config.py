# config.py
import torch


class Config:
    # ========== 模型配置 ==========
    bert_path = "/root/demo2/bert_models/bert-base-chinese"  # 或 "chinese-bert-wwm"
    lstm_hidden_size = 256
    lstm_layers = 1
    dropout = 0.1

    # ========== 训练配置 ==========
    batch_size = 16
    learning_rate = 3e-5
    epochs = 15
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_seq_len = 128

    # ========== 数据配置 ==========
    data_path = {
        "msra": {
            "train": "data/MSRA/train.txt",
            "dev": "data/MSRA/dev.txt",
            "test": "data/MSRA/test.txt"
        },
        "weibo": {
            "train": "data/weibo/train.txt",
            "dev": "data/weibo/dev.txt",
            "test": "data/weibo/test.txt"
        }
    }

    # ========== 其他配置 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42