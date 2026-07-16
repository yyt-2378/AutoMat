#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cif_dataset_filter.py

功能
-----
1. 从“总数据集”剔除：
   - “已生成数据”目录中已存在的 CIF
   - “二维材料测试集”目录中已存在的 CIF
   剩余文件复制到 --remain_dir 指定目录。

2. 统计“已生成数据”与“二维材料测试集”重合情况，并：
   - 打印重合 / 不重合数量
   - 将测试集中不与已生成数据重合的文件复制到 --test_not_overlap_dir。
"""
import argparse
import shutil
from pathlib import Path


def collect_cif_basenames(folder: str) -> set[str]:
    """返回 folder 下所有 *.cif 的文件名前缀（stem）集合"""
    return {p.stem for p in Path(folder).rglob("*.cif")}


def copy_by_basenames(src_dir: str, dst_dir: str, basenames: set[str]) -> None:
    """从 src_dir 递归查找 <stem>.cif 并复制到 dst_dir。"""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for stem in basenames:
        matches = list(src_dir.rglob(f"{stem}.cif"))
        if not matches:
            # 若需要，可取消下一行注释以提示缺失
            # print(f"警告: 未在 {src_dir} 找到 {stem}.cif")
            continue
        for file_path in matches:
            shutil.copy2(file_path, dst_dir / file_path.name)


def main():
    parser = argparse.ArgumentParser("CIF 数据集过滤脚本")
    parser.add_argument("--total_dir", default="./data_generation/modified_supercell_orthonalize_selected_cifs")
    parser.add_argument("--generated_dir", default="./data_generation/selected_samples")
    parser.add_argument("--test2d_dir", default="./data_generation/test_2d_materials_selected_cifs")
    parser.add_argument("--remain_dir", default="./data_generation/train_selected_samples")
    parser.add_argument("--test_not_overlap_dir", default="./data_generation/test_2d")
    args = parser.parse_args()

    # 收集 basenames
    total_set     = collect_cif_basenames(args.total_dir)
    generated_set = collect_cif_basenames(args.generated_dir)
    test2d_set    = collect_cif_basenames(args.test2d_dir)

    # ---------- 1. 生成剩余数据集 ----------
    excluded_set = generated_set | test2d_set
    remain_set   = total_set - excluded_set

    print(f"总数据集:                 {len(total_set)}")
    print(f"已生成数据:               {len(generated_set)}")
    print(f"二维材料测试集:           {len(test2d_set)}")
    print(f"需要剔除的文件数:         {len(excluded_set)}")
    print(f"剔除后剩余文件数:         {len(remain_set)}")

    copy_by_basenames(args.total_dir, args.remain_dir, remain_set)

    # ---------- 2. 测试集与已生成数据重合情况 ----------
    overlap_set      = generated_set & test2d_set
    test_unique_set  = test2d_set - generated_set

    print("\n====== 测试集与已生成数据重合统计 ======")
    print(f"重合文件数:               {len(overlap_set)}")
    if overlap_set:
        preview = ", ".join(sorted(overlap_set)[:20])
        print(f"重合文件列表(前20):        {preview}{' ...' if len(overlap_set) > 20 else ''}")
    print(f"测试集中独有文件数:       {len(test_unique_set)}")

    copy_by_basenames(args.test2d_dir, args.test_not_overlap_dir, test_unique_set)


if __name__ == "__main__":
    main()
