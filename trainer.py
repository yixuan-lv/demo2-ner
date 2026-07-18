# trainer.py
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import swanlab
from seqeval.metrics import classification_report, accuracy_score, f1_score


class Trainer:
    def __init__(self, model, train_loader, dev_loader, test_loader, config, label2id, id2label):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.config = config
        self.label2id = label2id
        self.id2label = id2label
        self.device = config.device

        # 优化器
        self.optimizer = AdamW(
            [
                {"params": model.bert.parameters(), "lr": 5e-5},
                {"params": model.bilstm.parameters(), "lr": 1e-3},
                {"params": model.classifier.parameters(), "lr": 1e-3},
                {"params": model.crf.parameters(), "lr": 1e-3},
            ],
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        total_steps = len(train_loader) * config.epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(config.warmup_ratio * total_steps),
            num_training_steps=total_steps
        )

        # 最佳验证 F1
        self.best_f1 = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss = self.model(input_ids, attention_mask, labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self, loader, stage="dev"):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {stage}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                predictions = self.model.predict(input_ids, attention_mask)

                # 将 attention_mask 转为 numpy 数组，方便逐样本处理
                mask_np = attention_mask.cpu().numpy()

                for i, (pred_seq, label_seq) in enumerate(zip(predictions, labels)):
                    # 正确获取当前样本的有效长度（非 padding 的长度）
                    valid_len = int(mask_np[i].sum())


                    pred_seq = pred_seq[1:valid_len - 1]
                    label_seq = label_seq[1:valid_len - 1]

                    pred_labels = [self.id2label[x] for x in pred_seq]
                    true_labels = [self.id2label[x.item()] for x in label_seq]

                    all_preds.append(pred_labels)
                    all_labels.append(true_labels)

        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        report = classification_report(all_labels, all_preds, zero_division=0)

        return f1, report

    def train(self):
        """完整训练流程"""
        for epoch in range(self.config.epochs):
            # 训练
            avg_loss = self.train_epoch(epoch)

            # 验证
            dev_f1, dev_report = self.evaluate(self.dev_loader, stage="dev")

            # 记录到 SwanLab
            swanlab.log({
                "train/loss": avg_loss,
                "dev/f1": dev_f1,
                "learning_rate": self.scheduler.get_last_lr()[0]
            })

            print(f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f}, Dev F1={dev_f1:.4f}")

            # 保存最佳模型
            if dev_f1 > self.best_f1:
                self.best_f1 = dev_f1
                torch.save(self.model.state_dict(), "best_model.pth")
                print(f"Best model saved with F1: {dev_f1:.4f}")

        # 测试最佳模型
        print("\n===== 加载最佳模型进行最终测试 =====")
        self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        test_f1, test_report = self.evaluate(self.test_loader, stage="test")
        print(f"Test F1: {test_f1:.4f}")
        print("\n测试集分类报告:")
        print(test_report)

        # 记录测试结果到 SwanLab
        swanlab.log({
            "test/f1": test_f1
        })

        return test_f1, test_report