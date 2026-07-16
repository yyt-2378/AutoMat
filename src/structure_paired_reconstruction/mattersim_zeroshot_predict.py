#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute single‑point energy with MatterSim

• **批量模式**  
    指定 `input_dir` → 逐个读取其中的 *.cif*，计算总能 & 逐原子能，  
    结果写入 `results.csv`（两列：*energy*、*energy_per_atom*）。

• **单文件调试**（保持原脚本逻辑）  
    直接给出 `input_file` 即可，只打印信息，不写 CSV。
"""

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"mattersim\.forcefield\.potential"
)

import torch
import numpy as np
import pandas as pd                              # ★ 新增
from pathlib import Path                         # ★ 新增
from ase.build import bulk
from ase.io import read, write
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running MatterSim on {device}")

# ---------- 配置区域 ----------
input_dir  = None   # ← ★ 改为包含 CIF 的文件夹
input_file = './relaxed.cif'                                # 单文件时填路径，否则设 None
out_csv    = Path("results.csv")                 # ★ 输出 CSV
# --------------------------------

def compute_energy(atoms):
    atoms.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth",
                                     device=device)
    e_tot  = atoms.get_potential_energy()
    e_atom = e_tot / len(atoms)
    return e_tot, e_atom


def run_single(path: Path):
    """原脚本的单文件流程（保持不变）。"""
    atoms = read(str(path))
    print(f"Loaded structure from {path.name}  (atoms={len(atoms)})")
    e_tot, e_atom = compute_energy(atoms)
    print(f"Energy (eV)               = {e_tot}")
    print(f"Energy per atom (eV/atom) = {e_atom}")
    print(f"Forces of first atom (eV/A) = {atoms.get_forces()[0]}")
    print(f"Stress[0][0] (eV/A^3)       = {atoms.get_stress(voigt=False)[0][0]}")
    print(f"Stress[0][0] (GPa)          = {atoms.get_stress(voigt=False)[0][0] / GPa}")
    return e_tot, e_atom


# ---------------- 主执行 ----------------
if input_dir and input_dir.is_dir():
    # ===== 批量模式 =====
    rows = []
    for cif in sorted(input_dir.glob("*.cif")):
        try:
            atoms = read(str(cif))
        except Exception as e:
            print(f"❌ {cif.name}: {e}")
            continue
        e_tot, e_atom = compute_energy(atoms)
        rows.append({"file": cif.name,
                     "energy": e_tot,
                     "energy_per_atom": e_atom})
        print(f"✔ {cif.name}: E = {e_tot:.6f}  E/atom = {e_atom:.6f}")

    if rows:
        pd.DataFrame(rows)[["file", "energy", "energy_per_atom"]].to_csv(
            out_csv, index=False
        )
        print(f"\nAll done → results saved to {out_csv}")
    else:
        print("No valid CIF processed.")

elif input_file:
    # ===== 单文件调试 =====
    run_single(Path(input_file))

else:
    print("❌ Please set either 'input_dir' or 'input_file' path.")
