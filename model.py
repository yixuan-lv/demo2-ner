# model.py


import torch
import torch.nn as nn
from transformers import AutoModel


class NERModel(nn.Module):
    """BERT + BiLSTM + 线性分类层 NER 模型"""

    def __init__(self, config, num_labels, id2label=None):
        super(NERModel, self).__init__()

        self.num_labels = num_labels
        self.id2label = id2label

        # BERT
        self.bert = AutoModel.from_pretrained(config.bert_path)
        bert_hidden_size = self.bert.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )
        lstm_output_size = config.lstm_hidden_size * 2

        # 线性分类层
        self.classifier = nn.Linear(lstm_output_size, num_labels)

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        print(f"   模型初始化完成")
        print(f"   BERT: {config.bert_path}")
        print(f"   BiLSTM: {config.lstm_hidden_size} (双向)")
        print(f"   标签数: {num_labels}")

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Dropout
        sequence_output = self.dropout(sequence_output)

        # BiLSTM
        lengths = attention_mask.sum(dim=1).cpu()
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            sequence_output,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, _ = self.bilstm(packed_input)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence_output.size(1)
        )

        # 分类
        logits = self.classifier(lstm_output)

        result = {'logits': logits}

        # 不管有没有 labels，都计算 predictions
        predictions = torch.argmax(logits, dim=-1)
        result['predictions'] = predictions

        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
            result['loss'] = loss

        return result

    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, labels=None)
            return outputs['predictions']

    def freeze_bert(self, freeze=True):
        for param in self.bert.parameters():
            param.requires_grad = not freeze
        status = "冻结" if freeze else "解冻"
        print(f"   BERT 参数已 {status}")

    def freeze_embeddings(self, freeze=True):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = not freeze
        status = "冻结" if freeze else "解冻"
        print(f"   BERT Embedding 层已 {status}")
