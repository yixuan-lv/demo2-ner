# main.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
import numpy as np
import torch
import swanlab
from transformers import AutoTokenizer

from config import Config
from utils import build_label_map
from data_loader import create_dataloader
from model import BertBiLSTMCRF
from trainer import Trainer


def set_seed(seed):
    """设置随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 设置随机种子
    set_seed(Config.seed)

    # 选择数据集和模型
    dataset_name = "weibo"  # 可选: "msra" 或 "weibo"
    model_name = "/root/demo2/bert_models/bert-base-chinese"  # 可选: "bert-base-chinese" 或 "chinese-bert-wwm"

    print(f"数据集: {dataset_name}, 模型: {model_name}")

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. 构建标签映射
    train_path = Config.data_path[dataset_name]["train"]
    label2id, id2label = build_label_map(train_path)
    num_labels = len(label2id)
    print(f"标签数量: {num_labels}")
    print(f"标签映射: {label2id}")

    # 3. 创建 DataLoader

    from utils import load_bio_data

    sentences, labels = load_bio_data(Config.data_path["weibo"]["train"])

    print("句子数：", len(sentences))

    for i in range(3):
        print("=" * 40)
        print(sentences[i])
        print(labels[i])


    batch_size = Config.batch_size
    max_len = Config.max_seq_len

    train_loader = create_dataloader(
        data_path=Config.data_path[dataset_name]["train"],
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=True
    )

    dev_loader = create_dataloader(
        data_path=Config.data_path[dataset_name]["dev"],
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=False
    )

    test_loader = create_dataloader(
        data_path=Config.data_path[dataset_name]["test"],
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=False
    )

    # 4. 创建模型
    model = BertBiLSTMCRF(
        bert_path=model_name,
        num_labels=num_labels,
        lstm_hidden_size=Config.lstm_hidden_size,
        lstm_layers=Config.lstm_layers,
        dropout=Config.dropout
    )
    model.to(Config.device)

    # 5. 初始化 SwanLab
    swanlab.init(
        project="demo2-ner",
        experiment_name=f"{dataset_name}_{model_name}",
        config={
            "dataset": dataset_name,
            "model": model_name,
            "batch_size": batch_size,
            "learning_rate": Config.learning_rate,
            "epochs": Config.epochs,
            "max_seq_len": max_len,
            "lstm_hidden_size": Config.lstm_hidden_size,
            "lstm_layers": Config.lstm_layers,
            "dropout": Config.dropout,
            "num_labels": num_labels
        }
    )

    # 6. 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        config=Config,
        label2id=label2id,
        id2label=id2label
    )

    test_f1, test_report = trainer.train()

    print(f"\n===== 最终结果 =====")
    print(f"数据集: {dataset_name}, 模型: {model_name}")
    print(f"测试 F1: {test_f1:.4f}")
    print(len(train_loader.dataset))
    print(len(train_loader))

    swanlab.finish()


if __name__ == "__main__":
    main()