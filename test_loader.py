# test_loader.py
from transformers import AutoTokenizer
from config import Config
from utils import build_label_map
from data_loader import create_dataloader

# 测试 MSRA 数据
print("=" * 50)
print("测试 MSRA 数据加载")
print("=" * 50)

train_path = Config.data_path["msra"]["train"]
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

label2id, id2label = build_label_map(train_path)
print(f"标签数量: {len(label2id)}")
print(f"标签映射: {label2id}")

dataloader = create_dataloader(
    data_path=train_path,
    tokenizer=tokenizer,
    label2id=label2id,
    batch_size=2,
    max_len=32
)

for batch in dataloader:
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"第一个样本的前10个标签: {batch['labels'][0][:10]}")
    break

print("\n数据加载成功！")