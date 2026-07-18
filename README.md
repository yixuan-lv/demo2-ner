# 基于 BERT 的中文命名实体识别（NER）实验

<p align="center">
  <b>BERT + BiLSTM + CRF 中文命名实体识别系统</b>
</p>

本项目基于 **PyTorch** 和 **Hugging Face Transformers** 框架，实现了一个完整的中文命名实体识别（Named Entity Recognition, NER）系统。

实验采用：

- 两种中文预训练模型：
  - `bert-base-chinese`
  - `chinese-bert-wwm`

- 两个中文 NER 数据集：
  - MSRA NER
  - Weibo NER

模型结构采用经典的：

```
BERT Encoder
      ↓
BiLSTM Context Encoder
      ↓
CRF Sequence Decoder
      ↓
BIO Entity Labels
```

同时结合 **SwanLab** 对训练过程进行实时监控和可视化，并进行了学习率等超参数调优实验。


---

# ✨ 项目特点

- ✅ 支持多种中文预训练模型
  - bert-base-chinese
  - chinese-bert-wwm

- ✅ 支持多个中文 NER 数据集
  - MSRA
  - Weibo

- ✅ 完整实现 BERT + BiLSTM + CRF 序列标注模型

- ✅ 支持 BIO 标签体系

- ✅ 支持自动构建标签映射

- ✅ 支持训练、验证、测试完整流程

- ✅ 使用 seqeval 计算实体级 Precision / Recall / F1

- ✅ 使用 SwanLab 可视化：
  - loss 曲线
  - F1 变化
  - 学习率变化


---

# 📂 项目结构

```
demo2-ner/
│
├── main.py                  # 主程序入口
├── model.py                 # BERT-BiLSTM-CRF模型定义
├── trainer.py               # 训练、验证、测试流程
├── data_loader.py           # 数据读取和预处理
├── utils.py                 # 标签处理、数据解析工具
├── config.py                # 超参数配置
│
├── requirements.txt         # 环境依赖
├── README.md
│
├── data/
│   └── ner/
│       ├── MSRA/
│       │   ├── train_5k.txt
│       │   ├── dev_1k.txt
│       │   └── test_1k.txt
│       │
│       └── weibo/
│           ├── train.txt
│           ├── dev.txt
│           └── test.txt
│
└── images/
    ├── exp1_bert-base-msra/
    ├── exp2_bert-base-weibo/
    ├── exp3_bert-wwm-msra/
    └── exp4_bert-wwm-weibo/
```


---

# 🛠️ 环境配置

## 创建环境

```bash
conda create -n demo2 python=3.10

conda activate demo2
```


## 安装依赖

```bash
pip install -r requirements.txt
```


主要依赖：

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.8.0 |
| Transformers | 4.57.6 |
| seqeval | 1.2.2 |
| torchcrf | 1.1.0 |
| SwanLab | 0.7.14 |


---

# 📚 数据集介绍


## 1. MSRA NER Dataset

MSRA 数据集由微软亚洲研究院发布，来源于新闻文本。

包含三类实体：

| 类型 | 含义 | 示例 |
|-|-|-|
| LOC | 地点 | 北京 |
| ORG | 组织机构 | 清华大学 |
| PER | 人名 | 张三 |


采用 BIO 标注：

| 标签 | 含义 |
|-|-|
| B-LOC | 地点开始 |
| I-LOC | 地点内部 |
| B-ORG | 组织开始 |
| I-ORG | 组织内部 |
| B-PER | 人名开始 |
| I-PER | 人名内部 |
| O | 非实体 |


---

## 2. Weibo NER Dataset

Weibo 数据集来自新浪微博文本。

相比 MSRA：

- 文本更加口语化
- 存在大量网络表达
- 实体边界更加复杂


包含：

- GPE
- LOC
- ORG
- PER


并进一步区分：

- NAM（专有名词）
- NOM（普通名词）


共 17 类标签：

```
O

B/I-GPE.NAM
B/I-GPE.NOM

B/I-LOC.NAM
B/I-LOC.NOM

B/I-ORG.NAM
B/I-ORG.NOM

B/I-PER.NAM
B/I-PER.NOM
```


---

# 🚀 运行方式


运行：

```bash
python main.py
```


修改模型和数据集：

```python
dataset_name = "msra"

model_name = "/root/demo2/bert_models/bert-base-chinese"
```


支持：

```python
dataset_name:

"msra"
"weibo"


model_name:

"bert-base-chinese"
"chinese-bert-wwm"
```


训练过程中会自动上传 SwanLab：

- Train Loss
- Dev F1
- Test F1
- Learning Rate


---

# 🧠 模型结构


## BERT Encoder

使用预训练语言模型获得上下文语义表示：

```
Input Tokens

↓

BERT

↓

768维隐藏表示
```


## BiLSTM

进一步学习序列上下文：

```
Forward LSTM
+
Backward LSTM

↓

Context Representation
```


## CRF

通过标签转移约束，提高 BIO 序列合法性：

例如：

```
B-PER → I-PER ✅

O → I-PER ❌
```


最终输出：

```
实体类别序列
```


---

# 📊 实验结果


## 1. 模型对比实验（2×2）


| 模型 | 数据集 | Test F1 | Best Dev F1 |
|-|-|-|-|
| bert-base-chinese | MSRA | **91.14%** | 93.20% |
| bert-base-chinese | Weibo | **69.03%** | 72.39% |
| chinese-bert-wwm | MSRA | **91.06%** | 93.20% |
| chinese-bert-wwm | Weibo | **65.86%** | 70.17% |


---

# 📈 实验分析


## 1. MSRA 数据集

两个模型表现非常接近：

```
bert-base-chinese:
91.14%

chinese-bert-wwm:
91.06%
```


说明：

- 新闻文本更加规范
- 实体边界明确
- 两种预训练模型均能很好适应


---

## 2. Weibo 数据集

性能明显下降：

```
MSRA ≈ 91%

Weibo ≈ 66~69%
```


原因：

- 微博文本噪声较大
- 网络语言丰富
- 实体表达不规范
- 小类别样本数量不足


---

## 3. 实体类别分析


效果最好：

```
PER（人物）
```

原因：

- 人名模式明显
- 数据量充足


困难类别：

```
ORG
LOC.NOM
GPE.NOM
```


主要原因：

- 样本数量少
- 类别分布不均衡


---

# 🔬 超参数调优实验


实验设置：

数据集：

```
Weibo
```

模型：

```
bert-base-chinese
```


固定：

| 参数 | 值 |
|-|-|
| batch_size | 16 |
| hidden_size | 256 |
| dropout | 0.2 |


调整 BERT 学习率：

| BERT Learning Rate | Test F1 | Best Dev F1 |
|-|-|-|
|1e-5|67.10%|72.34%|
|2e-5|69.03%|72.39%|
|3e-5|68.85%|70.29%|
|5e-5|68.33%|72.39%|


实验结论：

- 学习率过低，模型收敛不足
- 学习率过高，容易破坏预训练表示
- `2e-5` 获得最佳性能


---

# ⚙️ 最优参数配置


## MSRA

| 参数 | Value |
|-|-|
|Batch Size|32|
|Learning Rate|2e-5|
|Epoch|10|
|Hidden Size|256|
|Dropout|0.2|
|Max Length|128|


## Weibo

| 参数 | Value |
|-|-|
|Batch Size|16|
|Learning Rate|2e-5|
|Epoch|15|
|Hidden Size|256|
|Dropout|0.2|
|Max Length|128|


---

# 🔥 分层学习率策略


为了避免破坏 BERT 预训练知识，采用不同学习率：


| 模块 | Learning Rate |
|-|-|
|BERT Encoder|2e-5|
|BiLSTM|1e-3|
|Classifier|1e-3|
|CRF|1e-3|


---

# 📌 实验总结


通过本实验，实现了一个完整的中文 NER 系统。

主要结论：

1. BERT 能够有效提升中文实体识别能力。

2. BERT + BiLSTM + CRF 在 MSRA 数据集上达到约 91% F1。

3. 微博文本由于噪声和类别不平衡问题，性能明显低于新闻文本。

4. 不同预训练模型性能差异较小，但在微博场景中 bert-base-chinese 表现更优。

5. 学习率对模型性能影响明显，合理调整能够提升实体识别效果。


---

# 📊 可视化结果

训练过程通过 SwanLab 记录：

- Loss 曲线
- Dev F1 曲线
- Learning Rate 曲线
- 实验参数


实验图片：

```
images/

├── exp1_bert-base-msra_combined.png

├── exp2_bert-base-weibo_combined.png

├── exp3_bert-wwm-msra_combined.png

└── exp4_bert-wwm-weibo_combined.png
```


---

# 📚 References

- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- Chinese-BERT-wwm

- MSRA Chinese NER Dataset

- Weibo NER Dataset

- Hugging Face Transformers

- SwanLab Experiment Tracking

