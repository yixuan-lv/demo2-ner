# model.py
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertBiLSTMCRF(nn.Module):
    def __init__(self, bert_path, num_labels, lstm_hidden_size=128, lstm_layers=1, dropout=0.1):
        super().__init__()
        # 1. BERT 编码器
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size  # 通常是 768

        # 2. Dropout
        self.dropout = nn.Dropout(dropout)

        # 3. BiLSTM
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        lstm_output_size = lstm_hidden_size * 2  # 双向

        # 4. 线性分类层（发射概率）
        self.classifier = nn.Linear(lstm_output_size, num_labels)

        # 5. CRF 层
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] (可选)
        Returns:
            训练时返回 loss (标量)
            推理时返回 predictions (list of lists)
        """
        # 1. BERT 输出
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, bert_hidden_size]

        # 2. Dropout
        sequence_output = self.dropout(sequence_output)

        # 3. BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)  # [batch_size, seq_len, lstm_hidden_size*2]

        # 4. 线性分类层得到发射概率
        emissions = self.classifier(lstm_output)  # [batch_size, seq_len, num_labels]

        # 5. CRF 解码
        if labels is not None:
            # 训练模式：计算损失
            # attention_mask 是布尔类型用于 CRF
            mask = (attention_mask.bool())
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            # 推理模式：维特比解码得到最优路径
            mask = (attention_mask.bool())
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions

    def predict(self, input_ids, attention_mask):
        """便捷推理接口，返回预测标签序列"""
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids, attention_mask, labels=None)