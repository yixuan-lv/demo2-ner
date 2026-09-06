# 基于 BERT 的中文命名实体识别（NER）实验

本项目基于 **PyTorch** 和 **Hugging Face Transformers** 框架，实现了一个完整的中文命名实体识别系统。

模型采用：

> **BERT + BiLSTM + 线性分类层**

结构，通过 BERT 提取上下文语义表示，BiLSTM 建模序列依赖关系，线性层输出标签概率分布，最终输出最优实体标注序列。

项目使用：

- `bert-base-chinese`
- `chinese-bert-wwm`

两种中文预训练模型，在：

- MSRA 中文 NER 数据集
- Weibo 中文 NER 数据集

上进行实验，并进行了学习率调优分析。


---

## ✨ 项目特点

- ✅ 支持两种中文预训练模型
  - `bert-base-chinese`
  - `chinese-bert-wwm`

- ✅ 支持两个公开中文 NER 数据集
  - MSRA NER
  - Weibo NER

- ✅ 完整模型结构
BERT
↓
BiLSTM
↓
Linear
↓
NER 标签序列

text

- ✅ 分层学习率训练策略（BERT 层 2e-5，BiLSTM 层 1e-3，分类器层 1e-3）

- ✅ 完整实验流程
  - 模型训练
  - 验证评估
  - 测试集预测
  - 分类报告输出
  - SwanLab 实验记录


---

## 📂 数据集

### 1. MSRA NER 数据集

MSRA 是微软亚洲研究院发布的中文命名实体识别数据集，数据来源于新闻文本，包含三类实体：

- LOC（地点）
- ORG（组织）
- PER（人物）

共包含 7 个 BIO 标签。

| 标签 | 说明 | 示例 |
|---|---|---|
| O | 非实体 | — |
| B-LOC | 地名开始 | 北（京） |
| I-LOC | 地名内部 | 京 |
| B-ORG | 组织机构开始 | 清（华大学） |
| I-ORG | 组织机构内部 | 华 |
| B-PER | 人名开始 | 张（三） |
| I-PER | 人名内部 | 三 |


### 2. Weibo NER 数据集

Weibo 数据集来源于新浪微博，相比 MSRA，微博文本具有非正式表达、网络用语、实体边界复杂、噪声较大等特点。

该数据集包含 GPE、LOC、ORG、PER 4 类实体，每类进一步划分为 NAM（专有名词）和 NOM（普通名词），共 17 个标签。

| 标签 | 说明 | 示例 |
|---|---|---|
| O | 非实体 | — |
| B-GPE.NAM | 地缘政治实体-专有名词-开始 | 中（国） |
| I-GPE.NAM | 地缘政治实体-专有名词-内部 | 国 |
| B-GPE.NOM | 地缘政治实体-普通名词-开始 | 国（家） |
| I-GPE.NOM | 地缘政治实体-普通名词-内部 | 家 |
| B-LOC.NAM | 地点-专有名词-开始 | 故（宫） |
| I-LOC.NAM | 地点-专有名词-内部 | 宫 |
| B-LOC.NOM | 地点-普通名词-开始 | 这（里） |
| I-LOC.NOM | 地点-普通名词-内部 | 里 |
| B-ORG.NAM | 组织机构-专有名词-开始 | 阿（里巴巴） |
| I-ORG.NAM | 组织机构-专有名词-内部 | 里 |
| B-ORG.NOM | 组织机构-普通名词-开始 | 公（司） |
| I-ORG.NOM | 组织机构-普通名词-内部 | 司 |
| B-PER.NAM | 人名-专有名词-开始 | 马（云） |
| I-PER.NAM | 人名-专有名词-内部 | 云 |
| B-PER.NOM | 人名-普通名词-开始 | 这（个人） |
| I-PER.NOM | 人名-普通名词-内部 | 个人 |


---

## 📁 项目结构
demo2-ner/
│
├── main.py # 程序入口，配置数据集和模型
├── model.py # BERT + BiLSTM + Linear 模型定义
├── trainer.py # 训练与评估逻辑
├── config.py # 配置类（从 JSON 加载）
├── data_loader.py # 数据加载与预处理
├── utils.py # 工具函数与 NER 评估指标
├── requirements.txt # Python 依赖列表
├── README.md
│
├── configs/
│ ├── msra_config.json # MSRA 数据集配置
│ └── weibo_config.json # Weibo 数据集配置
│
├── data/
│ ├── MSRA/ # MSRA 数据集
│ └── weibo/ # Weibo 数据集
│
└── images/ # 实验结果图片
├── exp1_bert-base-msra_combined.png
├── exp2_bert-base-weibo_combined.png
├── exp3_bert-wwm-msra_combined.png
└── exp4_bert-wwm-weibo_combined.png

text


---

## ⚙️ 环境配置

### 创建环境

```bash
conda create -n demo2 python=3.10
conda activate demo2
安装 PyTorch
注意：本项目使用 RTX 5090 显卡，需要 PyTorch 2.8.0+ 及 CUDA 12.8 支持。

针对 RTX 50 系显卡（CUDA 12.8）：

bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
如果你的显卡不是 RTX 50 系或 CUDA 版本不同，请根据 PyTorch 官网 的指引选择合适的安装命令。

安装其他依赖
bash
pip install -r requirements.txt
核心依赖
软件	版本
Python	3.10
PyTorch	2.8.0+cu128
Transformers	4.57.6
SwanLab	0.7.14
🚀 运行方式
bash
python main.py
切换数据集或模型时，修改 main.py 中的参数：

python
dataset_name = "msra"      # 可选: "msra" 或 "weibo"
model_name = "bert-base-chinese"   # 可选: "bert-base-chinese" 或 "chinese-bert-wwm"
训练过程会自动记录到 SwanLab，可在浏览器中实时查看训练曲线和指标变化。

📊 实验结果
1. 核心对比实验（2×2 矩阵）
模型	数据集	测试 F1	最高验证 F1
chinese-bert-wwm	MSRA	91.21%	92.40%
bert-base-chinese	MSRA	90.39%	92.32%
chinese-bert-wwm	Weibo	67.16%	72.87%
bert-base-chinese	Weibo	66.42%	71.85%
实验分析
chinese-bert-wwm 在两个数据集上均略优于 bert-base-chinese：MSRA 上高出约 0.82 个百分点，Weibo 上高出约 0.74 个百分点。

MSRA 远容易于 Weibo：差距约 23-24 个百分点，验证了社交媒体文本的 NER 难度远大于规范新闻文本。

第三版代码效果提升明显：相比第二版，MSRA + chinese-bert-wwm 从 86.96% 提升到 91.21%，提升约 4.25 个百分点。

移除 CRF 后性能依然优秀：在 MSRA 上达到 90%+ F1，说明 BERT + BiLSTM + Linear 结构在规范文本上已足够强大；在 Weibo 上，移除 CRF 后性能基本持平，说明 CRF 在噪声数据上帮助有限。

2. 超参数调优实验（Weibo + bert-base-chinese）
固定其他超参数：

参数	设置
batch_size	16
hidden_size	256
dropout	0.2
仅调整 BERT 层学习率：

BERT 层学习率	测试 F1	最高验证 F1
1e-5	64.26%	72.82%
2e-5	69.24%	73.23%
3e-5	67.96%	71.99%
5e-5	67.59%	71.95%
实验分析
学习率从 1e-5 提高到 2e-5 时，测试 F1 上升约 4.98 个百分点（64.26% → 69.24%），提升显著。

最优学习率为 2e-5，验证集和测试集表现最平衡，泛化能力最好。

超过 2e-5 后，性能逐渐下降，3e-5 和 5e-5 的测试 F1 分别下降 1.28 和 1.65 个百分点。

1e-5 时验证 F1（72.82%）与测试 F1（64.26%）差距最大（8.56 个百分点），说明学习率偏小时模型在验证集上过拟合，泛化能力较弱。

3. 各实验详细分类报告
实验一：chinese-bert-wwm + MSRA（测试 F1: 91.21%）
实体类型	precision	recall	f1-score	support
LOC	0.9194	0.9209	0.9202	632
ORG	0.8577	0.8545	0.8561	268
PER	0.9342	0.9446	0.9394	361
加权平均	0.9107	0.9136	0.9121	1261
实验二：bert-base-chinese + MSRA（测试 F1: 90.39%）
实体类型	precision	recall	f1-score	support
LOC	0.8984	0.8956	0.8970	632
ORG	0.8664	0.8470	0.8566	268
PER	0.9452	0.9557	0.9504	361
加权平均	0.9053	0.9025	0.9039	1261
实验三：chinese-bert-wwm + Weibo（测试 F1: 67.16%）
实体类型	precision	recall	f1-score	support
GPE.NAM	0.8478	0.7358	0.7879	53
GPE.NOM	0.0000	0.0000	0.0000	0
LOC.NAM	0.2632	0.3846	0.3125	13
LOC.NOM	0.4444	0.6667	0.5333	6
ORG.NAM	0.4103	0.5714	0.4776	28
ORG.NOM	0.3750	0.6667	0.4800	9
PER.NAM	0.7545	0.6917	0.7217	120
PER.NOM	0.7246	0.6760	0.6994	179
加权平均	0.6716	0.6716	0.6716	408
实验四：bert-base-chinese + Weibo（测试 F1: 66.42%）
实体类型	precision	recall	f1-score	support
GPE.NAM	0.7600	0.8261	0.7917	46
GPE.NOM	0.0000	0.0000	0.0000	2
LOC.NAM	0.4583	0.5789	0.5116	19
LOC.NOM	0.5000	0.2222	0.3077	9
ORG.NAM	0.5200	0.3333	0.4063	39
ORG.NOM	0.5000	0.4375	0.4667	16
PER.NAM	0.7207	0.7143	0.7175	112
PER.NOM	0.6879	0.7041	0.6959	169
加权平均	0.6733	0.6553	0.6642	412
🔍 关键发现
chinese-bert-wwm 在两个数据集上均略优于 bert-base-chinese：MSRA 上高 0.82 个百分点，Weibo 上高 0.74 个百分点。

MSRA 远容易于 Weibo：F1 差距约 23-24 个百分点，验证了新闻文本与社交媒体文本的难度差异。

第三版代码效果提升明显：MSRA + chinese-bert-wwm 从第二版的 86.96% 提升到 91.21%。

人名（PER）识别效果最好：所有实验中 PER 的 F1 都是最高的。

机构名（ORG）识别难度最大：Weibo 上 ORG 的 F1 在 0.40-0.48 之间，远低于 PER 和 GPE。

样本量对性能影响显著：LOC.NOM、GPE.NOM 等少数类在测试集中样本极少，识别效果较差。

最优学习率为 2e-5：偏低或偏高都会损害模型性能。

⚙️ 参数配置详情
基础超参数
参数	MSRA 实验值	Weibo 实验值
batch_size	32	16
lstm_hidden_size	256	256
lstm_layers	1	1
dropout	0.2	0.2
epochs	10	15
warmup_ratio	0.1	0.1
weight_decay	0.01	0.01
max_seq_len	128	128
分层学习率
参数	学习率
BERT 层	2e-5
BiLSTM 层	1e-3
分类器层	1e-3
💻 实验环境
项目	配置
操作系统	Ubuntu 22.04
GPU	NVIDIA GeForce RTX 5090 (32GB)
Python	3.10
PyTorch	2.8.0+cu128
Transformers	4.57.6
📚 参考
bert-base-chinese

chinese-bert-wwm

MSRA NER 数据集

Weibo NER 数据集

SwanLab 实验可视化工具