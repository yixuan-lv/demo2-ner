基于BERT的中文命名实体识别（NER）实验
基于 bert-base-chinese 和 chinese-bert-wwm 预训练模型，在 MSRA 和 Weibo 数据集上完成中文命名实体识别任务。模型结构采用 BERT + BiLSTM + CRF，并进行了学习率调优实验。

项目结构
text
.
├── main.py           # 主程序入口，配置数据集和模型
├── model.py          # BERT + BiLSTM + CRF 模型定义
├── trainer.py        # 训练与评估逻辑
├── data_loader.py    # 数据加载与预处理
├── utils.py          # 工具函数（标签映射、数据读取）
├── config.py         # 超参数配置
├── requirements.txt  # 依赖列表
├── README.md
└── .gitignore
实验结果
1. 核心对比实验（2×2矩阵）
模型	数据集	测试 F1
bert-base-chinese	MSRA	91.14%
bert-base-chinese	Weibo	69.03%
chinese-bert-wwm	MSRA	91.06%
chinese-bert-wwm	Weibo	65.86%
两个模型在 MSRA 数据集上表现相近，F1 均超过 91%。但在 Weibo 数据集上，bert-base-chinese 的表现略优于 chinese-bert-wwm，差距约 3 个百分点。Weibo 整体性能低于 MSRA 约 20 个百分点，主要原因是社交媒体文本噪声更大、实体表达更不规范。

2. 超参数调优实验（Weibo + bert-base-chinese）
固定其他超参数，仅调整 BERT 层学习率：

BERT层学习率	测试 F1
1e-5	67.10%
2e-5	69.03%
3e-5	68.85%
5e-5	68.33%
学习率从 1e-5 提高到 2e-5 时，F1 上升了约 2 个百分点；继续增大学习率，性能逐渐下降。最优学习率为 2e-5。

3. 最优模型分类报告（bert-base-chinese + MSRA）
实体类型	precision	recall	f1-score	support
LOC	0.91	0.91	0.91	632
ORG	0.85	0.88	0.86	268
PER	0.95	0.96	0.96	361
加权平均	0.91	0.92	0.91	1261
人名（PER）识别效果最好，F1 达到 0.96；机构名（ORG）相对较弱，F1 为 0.86，可能与机构名形式更多样、长度更长有关。

可视化
训练过程中的验证 F1 曲线（bert-base-chinese + MSRA）
下图展示了模型在 MSRA 验证集上的 F1 变化趋势：

https://images/exp1_bert-base_msra_dev_f1.png

环境配置
bash
conda create -n demo2 python=3.10
conda activate demo2
pip install -r requirements.txt
运行方式
bash
python main.py
切换数据集或模型时，修改 main.py 中的 dataset_name 和 model_name 即可。

依赖
Python 3.10

PyTorch 2.8.0

Transformers 4.57.6

其他依赖见 requirements.txt

参考
bert-base-chinese

chinese-bert-wwm

MSRA & Weibo NER 数据集