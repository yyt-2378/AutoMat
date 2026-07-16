#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本脚本实现对一个文件夹中多个 CIF 文件的处理，将六方晶胞转换为正交晶胞。
转换规则：
  - 正交晶胞（α, β, γ 均约为 90°）：不做转换，直接输出原结构。
  - 六方晶胞：
      * 当 γ≈60° 时，采用 T1 = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]；
      * 当 γ≈120° 时，采用 T2 = [[1, 2, 0], [0, 1, 0], [0, 0, 1]]。
用法:
    python script.py input_folder output_folder
"""

import os
import sys
import numpy as np
from ase.io import read, write
from ase.build.supercells import make_supercell

def process_file(input_filepath, output_folder, tolerance=2.0):
    """
    处理单个 CIF 文件，根据晶胞角度判断是否需要转换，
    如果为六方晶胞，则采用对应的转换矩阵将其转换为正交晶胞，
    否则直接输出原结构。
    
    参数:
        input_filepath: 输入 CIF 文件的完整路径。
        output_folder: 输出文件夹路径。
        tolerance: 用于判断角度近似的容差（单位：°）。
    """
    try:
        atoms = read(input_filepath)
    except Exception as e:
        print(f"读取文件 {input_filepath} 时出错: {e}")
        return

    # 获取晶胞参数：a, b, c, α, β, γ
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

    # 判断是否为正交晶胞（所有角度接近 90°）
    if abs(alpha - 90) < tolerance and abs(beta - 90) < tolerance and abs(gamma - 90) < tolerance:
        print(f"{os.path.basename(input_filepath)}: 晶胞为正交，跳过转换。")
        new_atoms = atoms
    elif abs(gamma - 60) < tolerance:
        # 六方晶胞，γ≈60°
        T = [
            [1, 1, 0],
            [-1, 1, 0],
            [0, 0, 1],
        ]
        print(f"{os.path.basename(input_filepath)}: 检测到六方晶胞（γ≈60°），应用 T1 转换矩阵。")
        new_atoms = make_supercell(prim=atoms, P=T)
    elif abs(gamma - 120) < tolerance:
        # 六方晶胞，γ≈120°
        T = [
            [1, -1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]
        print(f"{os.path.basename(input_filepath)}: 检测到六方晶胞（γ≈120°），应用 T2 转换矩阵。")
        new_atoms = make_supercell(prim=atoms, P=T)
    else:
        print(f"{os.path.basename(input_filepath)}: 晶胞角度不符合转换要求，跳过转换。")
        new_atoms = atoms

    # 构造输出文件路径，文件名前加前缀 "orthogonal_"
    filename = os.path.basename(input_filepath)
    output_filename = "orthogonal_" + filename
    output_filepath = os.path.join(output_folder, output_filename)

    try:
        write(output_filepath, new_atoms, format="cif")
        print(f"文件 {filename} 处理完成，输出保存为: {output_filepath}")
    except Exception as e:
        print(f"保存文件 {output_filepath} 时出错: {e}")

def process_folder(input_folder, output_folder, tolerance=2.0):
    """
    处理输入文件夹中所有的 CIF 文件，将其转换（如需要）为正交晶胞，
    并输出到指定的输出文件夹中。
    
    参数:
        input_folder: 包含 CIF 文件的输入文件夹路径。
        output_folder: 输出文件夹路径。
        tolerance: 判断角度近似的容差（单位：°）。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".cif"):
            input_filepath = os.path.join(input_folder, filename)
            process_file(input_filepath, output_folder, tolerance)

if __name__ == "__main__":
    input_folder = './data_generation/supercell_selected_cifs'
    output_folder = './data_generation/supercell_orthonalize_selected_cifs'
    
    process_folder(input_folder, output_folder)
