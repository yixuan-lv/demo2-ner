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
            "train": "data/ner/MSRA/train_5k.txt",
            "dev": "data/ner/MSRA/dev_1k.txt",
            "test": "data/ner/MSRA/test_1k.txt"
        },
        "weibo": {
            "train": "data/ner/weibo/train.txt",
            "dev": "data/ner/weibo/dev.txt",
            "test": "data/ner/weibo/test.txt"
        }
    }

    # ========== 其他配置 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42