#!/usr/bin/env python3
import os
import argparse
import json
from collections import defaultdict
from pymatgen.core import Structure

def is_close(x, y, tol):
    return abs(x - y) / abs(x) < tol

def classify_lattice(structure, tol_angle=1.0, tol_ratio=0.05):
    """
    根据晶胞的参数分类：
    - 如果所有角度接近 90°（±tol_angle），则分类为“正交胞”
    - 如果 a≈b（相对误差小于 tol_ratio）、α≈β≈90° 且 γ 接近 60° 或 120°（±tol_angle），则分类为“六方胞”
    - 否则返回“其他”
    
    参数：
        structure : pymatgen Structure 对象
        tol_angle : 角度容差（单位：度），默认 3°
        tol_ratio : 长度比较容差（用于比较 a 与 b 是否相等），默认 5%
    
    返回：
        字符串，取值 “正交胞”、“六方胞” 或 “其他”
    """
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

    # 判断是否正交：所有角度均接近 90°
    if abs(alpha - 90) < tol_angle and abs(beta - 90) < tol_angle and abs(gamma - 90) < tol_angle:
        return "正交胞"
    # 判断是否为六方胞：要求 a≈b，α≈β≈90°，且 γ 接近 60° 或 120°
    elif is_close(a, b, tol_ratio) and abs(alpha - 90) < tol_angle and abs(beta - 90) < tol_angle \
         and (abs(gamma - 120) < tol_angle or abs(gamma - 60) < tol_angle):
        return "六方胞"
    else:
        return "其他"

def main():
    parser = argparse.ArgumentParser(description="分类输入文件夹中所有 CIF 文件的晶胞类型，并保存结果为 JSON 文件")
    parser.add_argument("--input_dir", type=str, default='./data_generation/selected_cif_files',
                        help="输入包含 CIF 文件的文件夹路径")
    parser.add_argument("--output_json", type=str, default="./data_generation/classification_result.json",
                        help="输出 JSON 文件的路径，默认 classification_result.json")
    args = parser.parse_args()
    
    # 使用 defaultdict 存储分类结果，键为晶胞类型，值为该类型下的文件名列表
    classifications = defaultdict(list)
    total_files = 0
    
    for filename in os.listdir(args.input_dir):
        if not filename.lower().endswith(".cif"):
            continue
        file_path = os.path.join(args.input_dir, filename)
        try:
            structure = Structure.from_file(file_path)
        except Exception as e:
            print(f"读取 {filename} 出错: {e}")
            continue
        
        cell_type = classify_lattice(structure)
        classifications[cell_type].append(filename)
        total_files += 1
    
    print(f"总共处理了 {total_files} 个 CIF 文件。")
    print("晶胞类型统计结果：")
    for cell_type, files in classifications.items():
        print(f"  {cell_type}: {len(files)} 个")
        for f in files:
            print("    - " + f)
    
    # 将分类结果保存为 JSON 文件
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(classifications, f, ensure_ascii=False, indent=4)
    
    print(f"\n分类结果已保存至 {args.output_json}")

if __name__ == "__main__":
    main()
