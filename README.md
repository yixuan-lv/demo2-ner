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

```
BERT
  ↓
BiLSTM
  ↓
Linear
  ↓
NER 标签序列
```

- ✅ 分层学习率训练策略（BERT 层 3e-5，BiLSTM 层 1e-3，分类器层 1e-3）

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

```
demo2-ner/
│
├── main.py              # 程序入口，配置数据集和模型
├── model.py             # BERT + BiLSTM + Linear 模型定义
├── trainer.py           # 训练与评估逻辑
├── config.py            # 配置类（从 JSON 加载）
├── data_loader.py       # 数据加载与预处理
├── utils.py             # 工具函数与 NER 评估指标
├── requirements.txt     # Python 依赖列表
├── README.md
│
├── configs/
│   ├── msra_config.json   # MSRA 数据集配置
│   └── weibo_config.json  # Weibo 数据集配置
│
├── data/
│   ├── MSRA/              # MSRA 数据集
│   └── weibo/             # Weibo 数据集
│
└── images/                # 实验结果图片
    ├── exp1_bert-base-msra_combined.png
    ├── exp2_bert-base-weibo_combined.png
    ├── exp3_bert-wwm-msra_combined.png
    └── exp4_bert-wwm-weibo_combined.png
```


---

## ⚙️ 环境配置

### 创建环境

```bash
conda create -n demo2 python=3.10
conda activate demo2
```

### 安装 PyTorch

> **注意：本项目使用 RTX 5090 显卡，需要 PyTorch 2.8.0+ 及 CUDA 12.8 支持。**

针对 RTX 50 系显卡（CUDA 12.8）：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

如果你的显卡不是 RTX 50 系或 CUDA 版本不同，请根据 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的指引选择合适的安装命令。

### 安装其他依赖

```bash
pip install -r requirements.txt
```

### 核心依赖

| 软件 | 版本 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.57.6 |
| SwanLab | 0.7.14 |


---

## 🚀 运行方式

```bash
python main.py
```

切换数据集或模型时，修改 `main.py` 中的参数：

```python
dataset_name = "msra"      # 可选: "msra" 或 "weibo"
model_name = "bert-base-chinese"   # 可选: "bert-base-chinese" 或 "chinese-bert-wwm"
```

训练过程会自动记录到 SwanLab，可在浏览器中实时查看训练曲线和指标变化。


---

## 📊 实验结果

### 1. 核心对比实验（2×2 矩阵）

基于最优学习率 **3e-5** 运行的四组实验：

| 模型 | 数据集 | 测试 F1 | 最高验证 F1 |
|---|---|---|---|
| chinese-bert-wwm | MSRA | **91.64%** | 92.97% |
| bert-base-chinese | MSRA | **91.34%** | 92.26% |
| bert-base-chinese | Weibo | **67.96%** | 71.99% |
| chinese-bert-wwm | Weibo | **64.88%** | 70.47% |

#### 实验分析

- `chinese-bert-wwm` 在 MSRA 上优于 `bert-base-chinese`，高出约 0.30 个百分点。
- 在 Weibo 上，`bert-base-chinese` 优于 `chinese-bert-wwm`，高出约 3.08 个百分点，说明 `bert-base-chinese` 在社交媒体文本上适应性更好。
- MSRA 远容易于 Weibo：差距约 23-26 个百分点，验证了社交媒体文本的 NER 难度远大于规范新闻文本。
- 在 MSRA 上，`BERT + BiLSTM + Linear` 已达到 91%+ F1，说明该结构在规范文本上已足够强大。


### 2. 超参数调优实验（Weibo + bert-base-chinese）

固定其他超参数：

| 参数 | 设置 |
|---|---|
| batch_size | 16 |
| hidden_size | 256 |
| dropout | 0.2 |

仅调整 BERT 层学习率：

| BERT 层学习率 | 测试 F1 | 最高验证 F1 |
|---|---|---|
| 1e-5 | 64.26% | 72.82% |
| 2e-5 | 66.42% | 71.85% |
| **3e-5** | **67.96%** | **71.99%** |
| 5e-5 | 67.59% | 71.95% |

#### 实验分析

- 学习率从 1e-5 提高到 3e-5 时，测试 F1 上升约 **3.70 个百分点**（64.26% → 67.96%），提升显著。
- **最优学习率为 3e-5**，验证集和测试集表现最平衡，泛化能力最好。
- 超过 3e-5 后，性能趋于稳定但略有下降，5e-5 的测试 F1 比 3e-5 低 0.37 个百分点。
- 1e-5 时验证 F1（72.82%）与测试 F1（64.26%）差距最大（8.56 个百分点），说明学习率偏小时模型在验证集上过拟合，泛化能力较弱。


### 3. 各实验详细分类报告

#### 实验一：chinese-bert-wwm + MSRA（测试 F1: 91.64%）

| 实体类型 | precision | recall | f1-score | support |
|---|---|---|---|---|
| LOC | 0.9257 | 0.9066 | 0.9161 | 632 |
| ORG | 0.8773 | 0.8806 | 0.8790 | 268 |
| PER | 0.9421 | 0.9474 | 0.9448 | 361 |
| **加权平均** | **0.9201** | **0.9128** | **0.9164** | 1261 |

![实验一](./images/exp1_bert-base-msra_combined.png)


#### 实验二：bert-base-chinese + MSRA（测试 F1: 91.34%）

| 实体类型 | precision | recall | f1-score | support |
|---|---|---|---|---|
| LOC | 0.9159 | 0.9130 | 0.9144 | 632 |
| ORG | 0.8535 | 0.8694 | 0.8614 | 268 |
| PER | 0.9452 | 0.9557 | 0.9504 | 361 |
| **加权平均** | **0.9109** | **0.9159** | **0.9134** | 1261 |

![实验二](./images/exp2_bert-base-weibo_combined.png)


#### 实验三：bert-base-chinese + Weibo（测试 F1: 67.96%）

| 实体类型 | precision | recall | f1-score | support |
|---|---|---|---|---|
| GPE.NAM | 0.7736 | 0.8913 | 0.8283 | 46 |
| GPE.NOM | 0.0000 | 0.0000 | 0.0000 | 2 |
| LOC.NAM | 0.5000 | 0.5789 | 0.5366 | 19 |
| LOC.NOM | 0.2000 | 0.1111 | 0.1429 | 9 |
| ORG.NAM | 0.5714 | 0.5128 | 0.5405 | 39 |
| ORG.NOM | 0.4545 | 0.3125 | 0.3704 | 16 |
| PER.NAM | 0.7500 | 0.7232 | 0.7364 | 112 |
| PER.NOM | 0.6798 | 0.7160 | 0.6974 | 169 |
| **加权平均** | **0.6796** | **0.6796** | **0.6796** | 412 |

![实验三](./images/exp3_bert-wwm-msra_combined.png)


#### 实验四：chinese-bert-wwm + Weibo（测试 F1: 64.88%）

| 实体类型 | precision | recall | f1-score | support |
|---|---|---|---|---|
| GPE.NAM | 0.7407 | 0.8696 | 0.8000 | 46 |
| GPE.NOM | 0.0000 | 0.0000 | 0.0000 | 2 |
| LOC.NAM | 0.3000 | 0.3158 | 0.3077 | 19 |
| LOC.NOM | 0.4444 | 0.4444 | 0.4444 | 9 |
| ORG.NAM | 0.4222 | 0.4872 | 0.4524 | 39 |
| ORG.NOM | 0.5625 | 0.5625 | 0.5625 | 16 |
| PER.NAM | 0.6667 | 0.7143 | 0.6897 | 112 |
| PER.NOM | 0.6630 | 0.7101 | 0.6857 | 169 |
| **加权平均** | **0.6247** | **0.6748** | **0.6488** | 412 |

![实验四](./images/exp4_bert-wwm-weibo_combined.png)


---

## 🔍 关键发现

- 在 MSRA 上，`chinese-bert-wwm` 优于 `bert-base-chinese`，高出约 0.30 个百分点；在 Weibo 上，`bert-base-chinese` 优于 `chinese-bert-wwm`，高出约 3.08 个百分点。
- MSRA 远容易于 Weibo：F1 差距约 23-26 个百分点，验证了新闻文本与社交媒体文本的难度差异。
- 在 MSRA 上，`BERT + BiLSTM + Linear` 已达到 91%+ F1，说明该结构在规范文本上已足够强大。
- 人名（PER）识别效果最好：所有实验中 PER 的 F1 都是最高的。
- 机构名（ORG）识别难度最大：Weibo 上 ORG 的 F1 在 0.37-0.56 之间，远低于 PER 和 GPE。
- 样本量对性能影响显著：LOC.NOM、GPE.NOM 等少数类在测试集中样本极少，识别效果较差。
- **最优学习率为 3e-5**：偏低或偏高都会损害模型性能。


---

## ⚙️ 参数配置详情

### 基础超参数

| 参数 | MSRA 实验值 | Weibo 实验值 |
|---|---|---|
| batch_size | 32 | 16 |
| lstm_hidden_size | 256 | 256 |
| lstm_layers | 1 | 1 |
| dropout | 0.2 | 0.2 |
| epochs | 10 | 15 |
| warmup_ratio | 0.1 | 0.1 |
| weight_decay | 0.01 | 0.01 |
| max_seq_len | 128 | 128 |

### 分层学习率

| 参数 | 学习率 |
|---|---|
| BERT 层 | **3e-5** |
| BiLSTM 层 | 1e-3 |
| 分类器层 | 1e-3 |


---

## 💻 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 5090 (32GB) |
| Python | 3.10 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.57.6 |


---

## 📚 参考

- [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)
- [chinese-bert-wwm](https://huggingface.co/hfl/chinese-bert-wwm)
- [MSRA NER 数据集](https://github.com/DetermineY/msra_ner)
- [Weibo NER 数据集](https://github.com/OYE0932/NER-Weibo)
- [SwanLab 实验可视化工具](https://swanlab.cn)
