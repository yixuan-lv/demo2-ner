# model.py
"""
NER 模型定义

包含:
    - NERModel: BERT + BiLSTM + 线性分类层 命名实体识别模型
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, Dict, Any, List, Tuple



class NERModel(nn.Module):
    """
    BERT + BiLSTM + 线性分类层 命名实体识别模型

    结构:
        BERT (预训练) → BiLSTM (序列建模) → 线性层 (分类)

    损失函数:
        CrossEntropyLoss (忽略 -100)
    """

    def __init__(self, config, num_labels: int, id2label: Optional[Dict[int, str]] = None):
        """
        参数:
            config: 配置对象 (包含 bert_path, lstm_hidden_size, lstm_layers, dropout)
            num_labels: 标签类别数量
            id2label: ID 到标签的映射 (可选)
        """
        super(NERModel, self).__init__()

        self.num_labels = num_labels
        self.id2label = id2label

        # ===== 1. BERT 编码器 =====
        self.bert = AutoModel.from_pretrained(config.bert_path)
        bert_hidden_size = self.bert.config.hidden_size

        # ===== 2. Dropout =====
        self.dropout = nn.Dropout(config.dropout)

        # ===== 3. BiLSTM =====
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )
        lstm_output_size = config.lstm_hidden_size * 2  # 双向

        # ===== 4. 线性分类层 =====
        self.classifier = nn.Linear(lstm_output_size, num_labels)

        # ===== 5. 损失函数 =====
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        print(f"✅ 模型初始化完成")
        print(f"   BERT: {config.bert_path}")
        print(f"   BiLSTM: {config.lstm_hidden_size} (双向)")
        print(f"   标签数: {num_labels}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        前向传播

        参数:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] (训练时传入)

        返回:
            dict: 包含 loss, logits, predictions
        """
        # ===== 1. BERT 编码 =====
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # ===== 2. Dropout =====
        sequence_output = self.dropout(sequence_output)

        # ===== 3. BiLSTM =====
        lstm_output, _ = self.bilstm(sequence_output)  # [batch, seq_len, lstm_hidden*2]

        # ===== 4. 分类 =====
        logits = self.classifier(lstm_output)  # [batch, seq_len, num_labels]

        result = {'logits': logits}

        # ===== 5. 训练时计算损失 =====
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, self.num_labels),  # [batch*seq_len, num_labels]
                labels.view(-1)  # [batch*seq_len]
            )
            result['loss'] = loss

        # ===== 6. 推理时获取预测 =====
        else:
            predictions = torch.argmax(logits, dim=-1)  # [batch, seq_len]
            result['predictions'] = predictions

        return result

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        推理接口

        参数:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        返回:
            predictions: [batch_size, seq_len] 预测的标签 ID
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, labels=None)
            return outputs['predictions']

    def freeze_bert(self, freeze: bool = True):
        """
        冻结/解冻 BERT 参数

        参数:
            freeze: True 冻结, False 解冻
        """
        for param in self.bert.parameters():
            param.requires_grad = not freeze
        status = "冻结" if freeze else "解冻"
        print(f"   BERT 参数已 {status}")

    def freeze_embeddings(self, freeze: bool = True):
        """
        冻结/解冻 BERT Embedding 层

        参数:
            freeze: True 冻结, False 解冻
        """
        for param in self.bert.embeddings.parameters():
            param.requires_grad = not freeze
        status = "冻结" if freeze else "解冻"
        print(f"   BERT Embedding 层已 {status}")