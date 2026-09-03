# main.py
"""
NER 模型训练入口文件
"""

import os
import sys
import random
import numpy as np
import torch
import swanlab
from transformers import AutoTokenizer

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 导入配置和工具
from config import Config
from model import NERModel
from trainer import Trainer

# 从 utils 统一导入
from utils import (
    create_dataloader,
    load_bio_data,
    analyze_data_statistics,
    get_ner_f1_score
)


def set_seed(seed: int) -> None:
    """设置随机种子，保证实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")


def build_label_mappings(dataset_name: str, train_path: str):
    """
    构建标签映射

    参数:
        dataset_name: 数据集名称 ("msra" 或 "weibo")
        train_path: 训练数据路径

    返回:
        (label2id, id2label, num_labels)
    """
    try:
        _, labels = load_bio_data(train_path)
        all_labels = set()
        for seq_labels in labels:
            all_labels.update(seq_labels)

        sorted_labels = sorted(all_labels)
        label2id = {label: idx for idx, label in enumerate(sorted_labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        print(f"从数据构建标签映射: {len(label2id)} 个标签")
        return label2id, id2label, len(label2id)

    except Exception as e:
        print(f"从数据构建标签映射失败: {e}")
        print("   使用预定义标签映射...")

    # 预定义映射
    if dataset_name == "msra":
        label2id = {
            'O': 0,
            'B-PER': 1, 'I-PER': 2,
            'B-LOC': 3, 'I-LOC': 4,
            'B-ORG': 5, 'I-ORG': 6
        }
    elif dataset_name == "weibo":
        label2id = {
            'O': 0,
            'B-GPE.NAM': 1, 'I-GPE.NAM': 2,
            'B-GPE.NOM': 3, 'I-GPE.NOM': 4,
            'B-LOC.NAM': 5, 'I-LOC.NAM': 6,
            'B-LOC.NOM': 7, 'I-LOC.NOM': 8,
            'B-ORG.NAM': 9, 'I-ORG.NAM': 10,
            'B-ORG.NOM': 11, 'I-ORG.NOM': 12,
            'B-PER.NAM': 13, 'I-PER.NAM': 14,
            'B-PER.NOM': 15, 'I-PER.NOM': 16
        }
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label, len(label2id)


def print_data_info(train_loader, dev_loader, test_loader, dataset_name: str) -> None:
    """打印数据信息"""
    print("\n" + "=" * 60)
    print(f"数据集信息: {dataset_name}")
    print("=" * 60)
    print(f"训练集: {len(train_loader.dataset)} 个样本, {len(train_loader)} 个批次")
    print(f"验证集: {len(dev_loader.dataset)} 个样本, {len(dev_loader)} 个批次")
    print(f"测试集: {len(test_loader.dataset)} 个样本, {len(test_loader)} 个批次")
    print("=" * 60 + "\n")


def main():
    """主函数"""

    # ============================================================
    # 1. 配置和实验设置
    # ============================================================

    dataset_name = "weibo"  # 可选: "msra" 或 "weibo"
    model_name = "bert-base-chinese"  # 可选: "bert-base-chinese" 或 "chinese-bert-wwm"

    model_path = f"./bert_models/{model_name}"

    print(f"\n启动实验")
    print(f"   数据集: {dataset_name}")
    print(f"   模型: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   设备: {device}")

    # 加载配置

    # 加载配置
    config = Config.from_json(f"configs/{dataset_name}_config.json")
    config.bert_path = model_path
    print(f"从 configs/{dataset_name}_config.json 加载配置")


    set_seed(config.seed)

    # ============================================================
    # 2. 数据加载
    # ============================================================

    data_paths = config.data_path.get(dataset_name, {})
    train_path = data_paths.get("train", "")
    dev_path = data_paths.get("dev", "")
    test_path = data_paths.get("test", "")

    for path_name, path in [("训练", train_path), ("验证", dev_path), ("测试", test_path)]:
        if not os.path.exists(path):
            print(f"{path_name}数据文件不存在: {path}")
            sys.exit(1)

    label2id, id2label, num_labels = build_label_mappings(dataset_name, train_path)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    analyze_data_statistics(
        train_path,
        tokenizer=tokenizer,
        label2id=label2id,
        max_len=config.max_seq_len
    )

    print("\n加载数据...")

    train_loader = create_dataloader(
        data_path=train_path,
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=config.batch_size,
        max_len=config.max_seq_len,
        shuffle=True,
        use_dynamic_padding=True
    )

    dev_loader = create_dataloader(
        data_path=dev_path,
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=config.batch_size,
        max_len=config.max_seq_len,
        shuffle=False,
        use_dynamic_padding=True
    )

    test_loader = create_dataloader(
        data_path=test_path,
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=config.batch_size,
        max_len=config.max_seq_len,
        shuffle=False,
        use_dynamic_padding=True
    )

    print_data_info(train_loader, dev_loader, test_loader, dataset_name)

    # ============================================================
    # 3. 模型
    # ============================================================

    print("构建模型...")

    model = NERModel(
        config=config,
        num_labels=num_labels,
        id2label=id2label
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型结构: BERT + BiLSTM + Linear")

    # ============================================================
    # 4. SwanLab 实验记录
    # ============================================================

    swanlab.init(
        project="demo2-ner",
        experiment_name=f"{dataset_name}_{model_name}_{config.bert_lr}",
        config={
            "dataset": dataset_name,
            "model": model_name,
            # 移除 use_crf 和 crf_lr
            "batch_size": config.batch_size,
            "bert_lr": config.bert_lr,
            "lstm_lr": config.lstm_lr,
            "classifier_lr": config.classifier_lr,
            "epochs": config.epochs,
            "max_seq_len": config.max_seq_len,
            "lstm_hidden_size": config.lstm_hidden_size,
            "lstm_layers": config.lstm_layers,
            "dropout": config.dropout,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "max_grad_norm": config.max_grad_norm,
            "num_labels": num_labels,
            "seed": config.seed
        }
    )

    print(f"SwanLab 已启动")

    # ============================================================
    # 5. 训练
    # ============================================================

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60 + "\n")

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        id2label=id2label
    )

    test_f1, test_report = trainer.train()

    # ============================================================
    # 6. 结果输出
    # ============================================================

    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"数据集: {dataset_name}")
    print(f"模型: {model_name}")
    print(f"模型结构: BERT + BiLSTM + Linear")
    print(f"测试 F1: {test_f1:.4f}")
    print("\n详细分类报告:")
    print(test_report)
    print("=" * 60)

    swanlab.finish()

    return test_f1, test_report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)