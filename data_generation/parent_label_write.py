#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd


def extract_parent_cif_name(image_filename):
    """
    通过图片文件名提取对应的父类别 cif 文件名。
    例如： "orthogonal_2dm-458_supercell_16x16x1_dose20000_sampling0.1_iDPC_V3_slide39_zoom.png"
    分割后取前4个部分拼接，并添加 ".cif"
    得到： "orthogonal_2dm-458_supercell_16x16x1.cif"
    """
    base_name = os.path.basename(image_filename)
    # 去除扩展名
    name_without_ext, _ = os.path.splitext(base_name)
    # 按下划线分割
    parts = name_without_ext.split('_')
    if len(parts) >= 4:
        parent_cif = "_".join(parts[:4]) + ".cif"
    else:
        # 不符合预期格式时直接加 .cif
        parent_cif = name_without_ext + ".cif"
    return parent_cif

def main():
    # 设置文件路径，根据实际情况修改
    input_csv_path = "./structure_recongnition/clustered_augmented_images.csv"           # 原始 CSV 文件路径
    json_mapping_path = "./data_generation/classification_parent/binary_cif_clusters.json"       # JSON 文件路径，文件中包含父类别映射
    output_csv_path = "./data_generation/classification_parent/augmented_dataset.csv"       # 输出新的 CSV 文件路径

    # 读取原始 CSV 文件
    df = pd.read_csv(input_csv_path)
    print("原始数据集记录数：", len(df))

    # 读取 JSON 映射文件
    with open(json_mapping_path, "r", encoding="utf-8") as f:
        parent_mapping = json.load(f)
    print("加载父类别映射，映射数：", len(parent_mapping))

    # 对每一行，根据 image_path 提取父类别名称，并查询 JSON 映射得到父类别标签
    def get_parent_label(row):
        img_path = row["image_path"]
        parent_cif = extract_parent_cif_name(img_path)
        # 如果在映射中找不到对应的父类别，可返回一个默认值（例如 -1）
        return parent_mapping.get(parent_cif, -1)

    # 新增一列 parent_label
    df["parent_label"] = df.apply(get_parent_label, axis=1)

    # 保存到新的 CSV 文件
    df.to_csv(output_csv_path, index=False)
    print("扩展后的数据集已保存到：", output_csv_path)

if __name__ == '__main__':
    main()
