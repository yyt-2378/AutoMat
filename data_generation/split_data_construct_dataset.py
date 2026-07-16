#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py

按照 8:1:1 比例随机划分 (image, label) 对到
moe_train_data/SRDATA/{training,validation,test}/{LR_original,HR}

使用方法：
python split_dataset.py
"""
import shutil
import random
from pathlib import Path


# ----------- 用户可修改的路径 -----------
IMG_DIR   = Path("./data_generation/simulate_data/aug_img")
LABEL_DIR = Path("./data_generation/simulate_data/aug_label")
OUT_ROOT  = Path("./data_generation/moe_train_data") / "SRDATA"
SPLITS    = [("training", 0.8), ("validation", 0.1), ("test", 0.1)]
SEED      = 42  # 固定随机种子以复现
# ---------------------------------------

def main():
    random.seed(SEED)

    # 1) 收集配对文件
    img_files = sorted(IMG_DIR.glob("*"))
    pairs = []
    for img_path in img_files:
        label_path = LABEL_DIR / img_path.name
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            print(f"[警告] 找不到标签文件: {label_path}")

    if not pairs:
        print("未找到成对的 (image, label) 文件，脚本终止。")
        return

    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * SPLITS[0][1])
    n_val   = int(n_total * SPLITS[1][1])
    n_test  = n_total - n_train - n_val

    split_sizes = {"training": n_train, "validation": n_val, "test": n_test}
    print("总数:", n_total, " -> ", split_sizes)

    # 2) 建立目标目录
    for split_name, _ in SPLITS:
        (OUT_ROOT / split_name / "LR_original").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split_name / "HR").mkdir(parents=True, exist_ok=True)

    # 3) 复制文件
    idx = 0
    for split_name, n_split in split_sizes.items():
        for _ in range(n_split):
            img_src, label_src = pairs[idx]
            idx += 1

            img_dst   = OUT_ROOT / split_name / "LR_original" / img_src.name
            label_dst = OUT_ROOT / split_name / "HR" / label_src.name

            shutil.copy2(img_src, img_dst)
            shutil.copy2(label_src, label_dst)

    print("数据划分完成，输出目录:", OUT_ROOT)

if __name__ == "__main__":
    main()
