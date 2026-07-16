#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relax a structure with MatterSim:

•  单文件模式  
       python relax_msim.py -f input.cif          (结果 → relaxed.cif)

•  批  量  模式  
       python relax_msim.py --input-dir folder_containing_cifs \
                            --out-dir   folder_to_save_results
       每个 CIF 依次处理，结果文件名为  <原名>_optimized.cif

其余原功能、函数与注释 **均保持不变**。
"""

import argparse
import numpy as np
from pathlib import Path
from ase.build import bulk
from ase.io import read, write
from mattersim.forcefield.potential import MatterSimCalculator
from mattersim.applications.relax import Relaxer
import warnings

# ignore mattersim.forcefield.potential inner FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"mattersim.forcefield.potential"
)

# ============ ★ 新增：原子数阈值，超过则跳过优化 ============
MAX_ATOMS = 1200          ### NEW
# ==========================================================


def axis_with_smallest_span(atoms):
    """
    返回坐标跨度（max - min）最小的轴索引 (0=x, 1=y, 2=z)。
    若有并列最小，取第一个出现的。
    """
    spans = atoms.positions.ptp(axis=0)  # ptp = max - min
    return int(np.argmin(spans))


def add_noise_on_axis(atoms, noise_scale=0.05):
    """
    在跨度最小的轴添加随机噪声。
    参数:
        atoms       : ASE Atoms 对象
        noise_scale : 噪声幅度 (Å)
    """
    axis = axis_with_smallest_span(atoms)
    atoms.positions[:, axis] += noise_scale * np.random.randn(len(atoms))
    return axis, atoms


# ------------------ CLI ------------------
parser = argparse.ArgumentParser(
    description="Relax CIF/XYZ structure(s) with MatterSim + BFGS"
)
# 单文件
parser.add_argument(
    "-f", "--file",
    default='./output_final_4803.cif',
    metavar="STRUCTURE.{xyz,cif}",
    help="input structure file (XYZ or CIF)",
)
# ☆ 批量新增
parser.add_argument(
    "--input-dir",
    default='./baseline/test_baseline_cif_ori',
    metavar="DIR",
    help="directory containing *.cif files (batch mode)",
)
parser.add_argument(
    "--out-dir",
    metavar="DIR",
    default="./baseline/relaxed_results",
    help="output directory for batch mode (default: ./relaxed_results)",
)
# 共同参数
parser.add_argument(
    "-n", "--steps",
    type=int,
    default=500,
    help="max optimisation steps (default 300)",
)
parser.add_argument(
    "--noise",
    type=float,
    default=0.05,
    help="random noise amplitude in Å (default 0.05 Å)",
)
args = parser.parse_args()

# ------------------ Calculator & Relaxer (一次即可) ------------------
calc = MatterSimCalculator(
    load_path="MatterSim-v1.0.0-5M.pth",
    device="cuda",
)
relaxer = Relaxer(
    optimizer="BFGS",        # 可改为 "FIRE" 等
    filter='ExpCellFilter',
    constrain_symmetry=True,
)


# ------------------ 处理逻辑 ------------------
def relax_single(in_path: Path, out_path: Path):
    """读取 -> 加噪 -> 松弛 -> 保存"""
    try:
        atoms = read(str(in_path))
    except Exception as err:
        print(f"❌ Cannot read {in_path.name}: {err}")
        return

    # ---------- ★ 阈值判断 ----------
    if len(atoms) > MAX_ATOMS:                                ### NEW
        print(f"⚠️  {in_path.name}  has {len(atoms)} atoms "   ### NEW
              f"(>{MAX_ATOMS}); skip relaxation.")            ### NEW
        write(str(out_path), atoms)                            ### NEW
        return                                                 ### NEW
    # --------------------------------

    axis, atoms = add_noise_on_axis(atoms, noise_scale=args.noise)
    print(f"Loaded {in_path.name} (atoms={len(atoms)}) "
          f"→ added noise on axis {['x','y','z'][axis]}")

    atoms.calc = calc
    converged, relaxed_atoms = relaxer.relax(atoms, steps=args.steps)

    write(str(out_path), relaxed_atoms)
    print(f"✔ Relaxed → {out_path.name}\n")


# ---------- 单文件或批量 ----------
if args.file:
    # ---- 单文件模式 ----
    input_file = Path(args.file).resolve()
    if not input_file.exists():
        raise SystemExit(f"❌ {input_file} not found")
    relax_single(input_file, Path("relaxed.cif"))

elif args.input_dir:
    # ---- 批量模式 ----
    in_dir  = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"❌ {in_dir} not found / not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(in_dir.glob("*.cif"))
    if not cif_files:
        raise SystemExit(f"❌ No *.cif files found in {in_dir}")

    for cif in cif_files:
        out_name = cif.stem + "_optimized.cif"
        relax_single(cif, out_dir / out_name)
else:
    # ---- fallback Si 测试 ----
    atoms = bulk("Si", "diamond", a=5.43)
    axis, atoms = add_noise_on_axis(atoms, noise_scale=0.1)
    print(f"No input supplied → using perturbed Si diamond (noise on {['x','y','z'][axis]})")
    atoms.calc = calc
    converged, relaxed_atoms = relaxer.relax(atoms, steps=args.steps)
    write("relaxed.cif", relaxed_atoms)
    print("Test cell relaxation complete → relaxed.cif")
