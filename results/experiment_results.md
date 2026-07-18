# Demo2 实验结果记录

实验一：bert-base-chinese + MSRA
测试 F1: 0.9114

分类报告:

实体类型	precision	recall	f1-score	support
LOC	0.91	0.91	0.91	632
ORG	0.85	0.88	0.86	268
PER	0.95	0.96	0.96	361
micro avg	0.91	0.92	0.91	1261
macro avg	0.90	0.92	0.91	1261
weighted avg	0.91	0.92	0.91	1261
超参数:

分层学习率: BERT层 2e-5, BiLSTM/Classifier/CRF层 1e-3
batch_size: 32
lstm_hidden_size: 256
lstm_layers: 1
dropout: 0.2
epochs: 10
warmup_ratio: 0.1
weight_decay: 0.01
max_seq_len: 128
- 图表: images/exp1_bert-base-msra


实验二：bert-base-chinese + Weibo
测试 F1: 0.6903

分类报告:

实体类型	precision	recall	f1-score	support
GPE.NAM	0.77	0.96	0.85	46
GPE.NOM	0.00	0.00	0.00	2
LOC.NAM	0.39	0.37	0.38	19
LOC.NOM	0.43	0.33	0.38	9
ORG.NAM	0.40	0.46	0.43	39
ORG.NOM	0.78	0.44	0.56	16
PER.NAM	0.78	0.78	0.78	112
PER.NOM	0.70	0.73	0.72	169
micro avg	0.69	0.70	0.69	412
macro avg	0.53	0.51	0.51	412
weighted avg	0.68	0.70	0.69	412
超参数:

分层学习率: BERT层 2e-5, BiLSTM/Classifier/CRF层 1e-3
batch_size: 16
lstm_hidden_size: 256
lstm_layers: 1
dropout: 0.2
epochs: 15
warmup_ratio: 0.1
weight_decay: 0.01
max_seq_len: 128
图表: images/exp2_bert-base-weibo/

实验三：chinese-bert-wwm + MSRA
测试 F1: 0.9106

分类报告:

实体类型	precision	recall	f1-score	support
LOC	0.93	0.90	0.92	632
ORG	0.86	0.86	0.86	268
PER	0.93	0.94	0.94	361
micro avg	0.92	0.90	0.91	1261
macro avg	0.91	0.90	0.91	1261
weighted avg	0.92	0.90	0.91	1261
超参数:

分层学习率: BERT层 2e-5, BiLSTM/Classifier/CRF层 1e-3

batch_size: 32
lstm_hidden_size: 256
lstm_layers: 1
dropout: 0.2
epochs: 15
warmup_ratio: 0.1
weight_decay: 0.01
max_seq_len: 128
图表: images/exp3_bert-wwm-msra/

完整学习率调优实验结果
BERT层学习率	Weibo 测试 F1
1e-5	    0.6710
2e-5	    0.6903 (基准)
3e-5	    0.6885
5e-5	    0.6833

实验四：chinese-bert-wwm + Weibo
测试 F1: 0.6586

分类报告:

实体类型	precision	recall	f1-score	support
GPE.NAM	0.74	0.87	0.80	46
GPE.NOM	0.00	0.00	0.00	2
LOC.NAM	0.25	0.37	0.30	19
LOC.NOM	0.29	0.22	0.25	9
ORG.NAM	0.44	0.44	0.44	39
ORG.NOM	0.64	0.44	0.52	16
PER.NAM	0.70	0.75	0.72	112
PER.NOM	0.70	0.72	0.71	169
micro avg	0.64	0.68	0.66	412
macro avg	0.47	0.48	0.47	412
weighted avg	0.64	0.68	0.66	412
超参数:

分层学习率: BERT层 2e-5, BiLSTM/Classifier/CRF层 1e-3

batch_size: 16
lstm_hidden_size: 256
lstm_layers: 1
dropout: 0.2
epochs: 15
warmup_ratio: 0.1
weight_decay: 0.01
max_seq_len: 128

图表:images/exp4_bert-wwm-weibo/

