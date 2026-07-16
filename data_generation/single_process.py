#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本脚本实现对一个文件夹中多个 CIF 文件的处理，检查每个 CIF 文件晶胞参数中的 γ 角，
如果 |γ - 90| 小于指定容差（例如 0.1°），则将 γ 修改为 90°并更新晶胞矩阵；
如果 |γ - 90| 大于或等于该容差，则跳过该文件，不进行保存。
用法:
    python modify_gamma_folder.py input_folder output_folder
"""

import os
import sys
from ase.io import read, write
from ase.geometry import cellpar_to_cell

def process_file(input_filepath, output_filepath, tol=0.7):
    """
    处理单个 CIF 文件：
      - 如果晶胞参数中 γ 与 90° 的误差小于 tol，则将 γ 修改为 90°（更新晶胞矩阵后输出），
      - 否则（误差大于或等于 tol）跳过该文件，不进行保存。
    
    参数:
        input_filepath: 输入 CIF 文件的完整路径。
        output_filepath: 输出 CIF 文件的完整路径。
        tol: 判断 γ 与 90° 差值的容差（单位：°）。
    """
    try:
        atoms = read(input_filepath)
    except Exception as e:
        print(f"读取文件 {input_filepath} 时出错: {e}")
        return

    # 获取晶胞参数：a, b, c, α, β, γ
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    print(f"处理文件 {os.path.basename(input_filepath)}: 原始晶胞参数: "
          f"a={a:.4f}, b={b:.4f}, c={c:.4f}, α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}")

    if abs(gamma - 90) < tol:
        # 如果误差在容差内
        if abs(gamma - 90) > 0:  # 需要修改
            print(f"文件 {os.path.basename(input_filepath)}: γ={gamma:.4f} 与 90°的差值小于 {tol}°，修改 γ 为 90°。")
            new_cellpar = [a, b, c, alpha, beta, 90.0]
            new_cell = cellpar_to_cell(new_cellpar)
            atoms.set_cell(new_cell, scale_atoms=False)
            a_new, b_new, c_new, alpha_new, beta_new, gamma_new = atoms.get_cell_lengths_and_angles()
            print(f"文件 {os.path.basename(input_filepath)}: 修改后的晶胞参数: "
                  f"a={a_new:.4f}, b={b_new:.4f}, c={c_new:.4f}, α={alpha_new:.4f}, β={beta_new:.4f}, γ={gamma_new:.4f}")
        else:
            print(f"文件 {os.path.basename(input_filepath)}: γ 已经为 90°，无需修改。")
    else:
        print(f"文件 {os.path.basename(input_filepath)}: γ={gamma:.4f} 与 90°的差值大于或等于 {tol}°，跳过该文件。")
        return

    try:
        write(output_filepath, atoms, format="cif")
        print(f"文件 {os.path.basename(input_filepath)} 处理完成，保存为: {output_filepath}")
    except Exception as e:
        print(f"保存文件 {output_filepath} 时出错: {e}")

def process_folder(input_folder, output_folder, tol=0.7):
    """
    遍历输入文件夹中所有 CIF 文件，检查每个文件晶胞参数中的 γ 角，
    如果误差小于 tol，则修改 γ 为 90°后保存到输出文件夹；否则跳过该文件。
    
    参数:
        input_folder: 包含 CIF 文件的输入文件夹路径。
        output_folder: 输出文件夹路径。
        tol: 判断 γ 与 90° 差值的容差（单位：°）。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".cif"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            process_file(input_filepath, output_filepath, tol)


if __name__ == "__main__":
    input_folder = './data_generation/supercell_orthonalize_selected_cifs'
    output_folder = './data_generation/orthonalize_supercell_cifs'
    
    process_folder(input_folder, output_folder)
