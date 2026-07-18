import os
from PIL import Image

# 实验配置：文件夹名，实际文件名的前缀
experiments = [
    ("exp1_bert-base-msra", "exp1_bert-base_msra"),
    ("exp2_bert-base-weibo", "exp2_bert-base_weibo"),
    ("exp3_bert-wwm-msra", "exp3_bert-wwm_msra"),
    ("exp4_bert-wwm-weibo", "exp4_bert-wwm_weibo"),
]

suffixes = ["train_loss", "dev_f1", "test_f1", "learning_rate"]

for folder, prefix in experiments:
    img_paths = []
    for suffix in suffixes:
        # 构造文件名：前缀 + _ + 后缀 + .png
        fname = f"{prefix}_{suffix}.png"
        path = f"images/{folder}/{fname}"
        img_paths.append(path)

    # 检查所有图片是否存在
    missing = [p for p in img_paths if not os.path.exists(p)]
    if missing:
        print(f"跳过 {folder}，缺少: {missing}")
        continue

    imgs = [Image.open(p) for p in img_paths]
    w, h = imgs[0].size
    result = Image.new('RGB', (w * 2, h * 2))
    result.paste(imgs[0], (0, 0))
    result.paste(imgs[1], (w, 0))
    result.paste(imgs[2], (0, h))
    result.paste(imgs[3], (w, h))
    result.save(f"images/{folder}_combined.png")
    print(f"已生成: {folder}_combined.png")