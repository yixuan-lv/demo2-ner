# trainers/trainer.py


import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
import swanlab

from utils.metrics import align_predictions, get_ner_f1_score, compute_ner_classification_report


class Trainer:
    """NER 模型训练器"""

    def __init__(self, model: nn.Module, config,device,
                 train_loader: DataLoader, dev_loader: DataLoader,
                 test_loader: DataLoader, id2label: Dict[int, str]):
        """
        参数:
            model: NER 模型
            config: 配置对象
            device: 设备 (cuda/cpu)
            train_loader: 训练数据加载器
            dev_loader: 验证数据加载器
            test_loader: 测试数据加载器
            id2label: ID 到标签的映射
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.id2label = id2label
        self.device = device

        # ===== 优化器（分层学习率） =====
        optimizer_params = []

        # BERT 层
        if hasattr(model, 'bert'):
            optimizer_params.append({
                "params": model.bert.parameters(),
                "lr": config.bert_lr
            })
            print(f"   BERT 学习率: {config.bert_lr}")

        # BiLSTM 层
        if hasattr(model, 'bilstm'):
            optimizer_params.append({
                "params": model.bilstm.parameters(),
                "lr": config.lstm_lr
            })
            print(f"   BiLSTM 学习率: {config.lstm_lr}")

        # 分类器层
        if hasattr(model, 'classifier'):
            optimizer_params.append({
                "params": model.classifier.parameters(),
                "lr": config.classifier_lr
            })
            print(f"   分类器学习率: {config.classifier_lr}")

        self.optimizer = AdamW(optimizer_params, weight_decay=config.weight_decay)

        # ===== 学习率调度器 =====
        total_steps = len(train_loader) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # ===== Early Stopping =====
        self.best_dev_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_model_state = None

        # ===== 输出目录 =====
        os.makedirs(config.output_dir, exist_ok=True)

        print(f"✅ Trainer 初始化完成")
        print(f"   设备: {self.device}")
        print(f"   训练批次: {len(train_loader)}")
        print(f"   验证批次: {len(dev_loader)}")
        print(f"   测试批次: {len(test_loader)}")
        print(f"   总步数: {total_steps}")
        print(f"   预热步数: {warmup_steps}")

        # ===== SwanLab =====
        swanlab.init(
            project="NER-Demo2",
            experiment_name=f"{os.path.basename(config.bert_path)}_{os.path.basename(config.data_path.get('train', ''))}",
            config=config.__dict__
        )

    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # 数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']

            # 反向传播
            loss.backward()

            # ===== 梯度裁剪 =====
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 记录到 SwanLab（每10步记录一次）
            if batch_idx % 10 == 0:
                swanlab.log({
                    'train/loss': loss.item(),
                    'train/lr': self.scheduler.get_last_lr()[0]
                })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self, dataloader: DataLoader, desc: str = "Evaluating") -> Dict[str, Any]:
        """评估模型"""
        self.model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)

                # ===== 获取预测结果（无 CRF） =====
                preds = outputs['predictions'].cpu().numpy()

                # ===== 转换为标签字符串列表 =====
                batch_labels = labels.cpu().numpy()
                batch_attention = attention_mask.cpu().numpy()

                for i in range(len(preds)):
                    # 找到有效 token 的位置（attention_mask == 1）
                    valid_indices = batch_attention[i] == 1

                    # 只取有效位置的预测和标签
                    pred_seq = [self.id2label.get(p, 'O') for p in preds[i][valid_indices]]
                    label_seq = [self.id2label.get(l, 'O') for l in batch_labels[i][valid_indices]]

                    all_predictions.append(pred_seq)
                    all_labels.append(label_seq)

                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()

        # 计算指标
        f1 = get_ner_f1_score(all_predictions, all_labels)
        report = compute_ner_classification_report(all_predictions, all_labels)

        return {
            'f1': f1,
            'report': report,
            'predictions': all_predictions,
            'labels': all_labels,
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        }

    def train(self) -> Tuple[float, str]:
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("🎯 开始训练")
        print("=" * 60)

        for epoch in range(self.config.epochs):
            # ===== 训练 =====
            train_loss = self.train_epoch(epoch)

            # ===== 验证 =====
            dev_results = self.evaluate(self.dev_loader, desc="Validating")
            dev_f1 = dev_results['f1']

            # ===== 记录到 SwanLab =====
            swanlab.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'dev/f1': dev_f1,
                'dev/loss': dev_results['loss']
            })

            print(f"\n📊 Epoch {epoch + 1}/{self.config.epochs}")
            print(f"   训练损失: {train_loss:.4f}")
            print(f"   验证 F1: {dev_f1:.4f}")

            # ===== Early Stopping =====
            if dev_f1 > self.best_dev_f1:
                self.best_dev_f1 = dev_f1
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save_model(f"best_model_epoch_{epoch + 1}")
                print(f"   ✅ 新的最佳模型! F1: {dev_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"   ⚠️ F1 未提升, 耐心计数: {self.patience_counter}/{self.config.patience}")

            # 每 5 个 epoch 保存检查点
            if (epoch + 1) % 5 == 0:
                self._save_model(f"checkpoint_epoch_{epoch + 1}")

            # 判断是否 Early Stop
            if self.patience_counter >= self.config.patience:
                print(f"\n🛑 Early Stopping 触发于 epoch {epoch + 1}")
                break

        # ===== 加载最佳模型进行测试 =====
        print(f"\n📂 加载最佳模型 (epoch {self.best_epoch})")
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)

        # ===== 测试 =====
        print("\n🧪 测试评估...")
        test_results = self.evaluate(self.test_loader, desc="Testing")

        # 保存最终模型
        self._save_model("final_model")

        return test_results['f1'], test_results['report']

    def _save_model(self, name: str):
        """保存模型"""
        save_path = os.path.join(self.config.output_dir, f"{name}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'id2label': self.id2label,
            'best_dev_f1': self.best_dev_f1,
            'best_epoch': self.best_epoch
        }, save_path)
        print(f"   💾 模型已保存: {save_path}")

    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"✅ 模型已加载: {model_path}")
        return checkpoint