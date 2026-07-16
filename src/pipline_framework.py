#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_pipeline_ds.py  (updated)

STEM big‑image → patch inference → template match → structure reconstruction →
primitive‑cell reduction → MatterSim relaxation.

改动要点
--------

1. **`inference_large_image_cv2_pil`** 现在显式 **return**
   重建结果的 PNG 路径，脚本直接使用其返回值来确定 `recon_png`。

2. **当用户没有给元素组成**
   * 在选出最终 label 后，自动用 label 文件名里的 *material_id*
     去 `metadata_csv` 里查元素集合，供 `process_image()` 使用。
   * 同时 `refine_top1()` 保持"用户优先"的过滤逻辑：
     – 若用户给了元素 → 用用户元素过滤；
     – 否则直接取 `top_matches[0]`（最快）。

其余流程保持不变。
"""

import os, re, json, shutil, sys, time, uuid, warnings
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from ase.io import read, write
from ase.units import GPa
from pymatgen.core import Structure

# ===== 请确认以下自定义模块可 import =====
_ROOT_UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(_ROOT_UTILS) not in sys.path:
    sys.path.insert(0, str(_ROOT_UTILS))

from fft_convert import load_gray_image, analyze_image
from test_large_image_model_batch import inference_large_image_cv2_pil
from structure_recongnition.atom_ai_analysis_fft import generate_candidate_windows
from structure_recongnition.batch_structure_paired import match_one, load_metadata, extract_material_id
from structure_recongnition.batch_atoms_analysis import process_image, shrink_once
from structure_recongnition.justify_img_quality import imread_gray, assess_image
from structure_recongnition.utils_without_template import reconstruct_minimal_cell
from mattersim.forcefield import MatterSimCalculator
from mattersim.applications.relax import Relaxer

# ========================================

warnings.filterwarnings("ignore", category=FutureWarning, module=r"mattersim")

# ---------- 辅助函数 ---------- #
_ELEMENT_PATTERN = re.compile(r'(?:elements?|元素)\s*[:=]\s*([A-Za-z,\s]+)', re.I)


def assess_image_path(
        path: str,
        min_periodicity: float = 0.40,
        min_snr: float = 2.0,
        min_cnr: float = 1.5,
):
    """Assess a restored STEM image from a file path."""
    img = imread_gray(path)
    result = assess_image(
        img,
        min_periodicity=min_periodicity,
        min_snr=min_snr,
        min_cnr=min_cnr,
    )
    return {"image": str(path), **result}


def shrink_or_window(sup_cif, out_cif_path, MAX_NUM_ITER=4, MAX_ATOMS_NUM=50):
    cur = Structure.from_file(str(sup_cif))
    for step in range(1, MAX_NUM_ITER):
        if len(cur) <= MAX_ATOMS_NUM:
            print(f"≤{MAX_ATOMS_NUM} atoms，停止。")
            break
        nxt = shrink_once(cur)
        if nxt is None or len(nxt) >= len(cur):
            print("已无法进一步缩减。")
            break
        print(f"Step {step}: {len(cur)} → {len(nxt)} atoms")
        cur = nxt
    cur.to(filename=str(out_cif_path))


def parse_elements_from_text(text: str) -> Optional[List[str]]:
    m = _ELEMENT_PATTERN.search(text)
    if not m:
        return None
    return [e.strip().capitalize() for e in re.split(r'[,\s]+', m.group(1)) if e.strip()]


def span_min_axis(atoms):
    return int(np.argmin(atoms.positions.ptp(axis=0)))


def refine_top1(top_matches, user_elements, metadata):
    """
    若 `user_elements` 给定 → 用它们过滤；否则直接返回 top‑1
    """
    if not user_elements:
        return top_matches[0][0]

    target = set(user_elements)
    filtered = []
    for name, dist in top_matches:
        mid = extract_material_id(name)
        elems = metadata.get(mid, set())
        if elems == target:
            filtered.append((name, dist))
    return (filtered or top_matches)[0][0]


# ---------- 主流程 ---------- #
def run_agent_pipeline(
        image_path: str,
        user_message: str,
        *,
        work_root: str,
        weight_path: str,
        label_dir: str,
        metadata_csv: str,
        max_atoms: int = 50,
        max_shrink_iter: int = 4,
        relax_steps: int = 500,
        noise_amp: float = 0.05,
        device="cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    t0 = time.time()
    img_p = Path(image_path).expanduser().resolve(strict=True)
    root = Path(work_root).expanduser()
    d_recon = root / "01_recon"
    d_label = root / "02_label"
    d_cif = root / "03_recon_cif"
    d_relax = root / "04_relax"
    for d in (d_recon, d_label, d_cif, d_relax):
        d.mkdir(parents=True, exist_ok=True)

    # === 1. PATCH 推理重建 ===
    print(f"\n=== [1] PATCH infer  {img_p.name} ===")
    recon_arr = inference_large_image_cv2_pil(
        str(img_p), weight_path,
        crop_size=128, stride=64, batch_size=32, device=device
    )
    recon_png = d_recon / f"{img_p.stem}_recon.png"  # ① 定义保存路径
    cv2.imwrite(str(recon_png), recon_arr)  # recon 是 uint8 numpy(H×W)
    print("Reconstructed image saved:", recon_png)  # ③ 后续继续用 Path

    # === 2. 模板匹配 ===
    print(f"\n=== [2] Template matching ===")
    metadata = load_metadata(Path(metadata_csv))
    user_elems = parse_elements_from_text(user_message)
    if user_elems:
        print("User‑provided elements:", user_elems)
    else:
        print("⚠️No element info in message – will infer from label/CSV")

    top_matches = match_one(recon_png, Path(label_dir), topk=3, min_area=5, max_dist=None, bin_width=5.0)
    best_label_name = refine_top1(top_matches, user_elems, metadata)

    src_label = Path(label_dir) / best_label_name
    dst_label = d_label / best_label_name
    shutil.copy2(src_label, dst_label)
    print("Chosen label:", best_label_name)

    # 若用户没给元素 → 用 label 的 material_id 去 CSV 查
    if not user_elems:
        mid = extract_material_id(best_label_name)
        user_elems = sorted(metadata.get(mid, []))
        if user_elems:
            print(f"Inferred elements from CSV: {user_elems}")
        else:
            print("⚠️  Cannot infer elements from CSV – process_image will use default logic")

    # === 3. 图像 → CIF & 缩减 ===
    print(f"\n=== [3] Image→CIF & shrink (≤{max_atoms} atoms) ===")

    # 3‑1  从 label PNG 得到 Atoms
    atoms = process_image(dst_label, user_elems)  # ← 仍然只返回 atoms
    if atoms is None or len(atoms) == 0:
        raise RuntimeError("process_image failed to detect atoms!")

    # 3‑2  保存 super‑cell CIF
    mid_match = re.search(r"(2dm-\d+)", dst_label.stem, re.I)
    mid = mid_match.group(1).lower() if mid_match else f"tmp-{uuid.uuid4().hex[:6]}"
    cif_super = d_cif / f"{mid}_reconstructed.cif"
    write(cif_super, atoms, format="cif", wrap=False)
    print("Super‑cell CIF:", cif_super)

    # 3‑3  调你已有的 shrink_or_window → output_final.cif
    cif_final = d_cif / "output_final.cif"
    shrink_or_window(cif_super, cif_final)  # ★ 你的现成函数
    print(f"Reduced cell saved: {cif_final}")

    # === 4. MatterSim Relax ===
    print(f"\n=== [4] MatterSim relaxation (device={device}) ===")

    atoms_relax = read(cif_final)
    axis_min = span_min_axis(atoms_relax)
    atoms_relax.positions[:, axis_min] += noise_amp * np.random.randn(len(atoms_relax))

    atoms_relax.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
    relaxer = Relaxer(optimizer="BFGS", filter=None, constrain_symmetry=False)
    converged, atoms_relaxed = relaxer.relax(atoms_relax, steps=relax_steps)

    cif_relaxed = d_relax / "relaxed.cif"
    write(cif_relaxed, atoms_relaxed)

    E = atoms_relaxed.get_potential_energy()
    F0 = atoms_relaxed.get_forces()[0]
    sxx = atoms_relaxed.get_stress(voigt=False)[0][0]

    print(f"Converged: {converged}")
    print(f"Energy (eV)               = {E:.6f}")
    print(f"Energy per atom (eV/atom) = {E / len(atoms_relaxed):.6f}")
    print(f"First‑atom force (eV/Å)   = {F0}")
    print(f"Stress xx (GPa)           = {sxx / GPa:.6f}")
    print(f"Total elapsed             = {time.time() - t0:.1f} s")

    return {
        "relaxed_cif": str(cif_relaxed),
        "energy_eV": float(E),
        "energy_per_atom": float(E / len(atoms_relaxed)),
        "force_first_atom": F0.tolist(),
        "stress_xx_GPa": float(sxx / GPa),
        "converged": bool(converged),
    }


def denoise_patch_inference_tool(image_path, weight_path, work_root, device="cuda"):
    """Step 1: Patch推理重建，返回重建图片路径"""
    try:
        img_p = Path(image_path).expanduser().resolve(strict=True)
        d_recon = Path(work_root).expanduser() / "01_recon"
        d_recon.mkdir(parents=True, exist_ok=True)
        recon_arr = inference_large_image_cv2_pil(
            str(img_p), weight_path,
            crop_size=128, stride=64, batch_size=32, device=device
        )
        recon_png = d_recon / f"{img_p.stem}_recon.png"
        cv2.imwrite(str(recon_png), recon_arr)
        return {"success": True, "recon_png": str(recon_png)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def template_match_tool(recon_png, label_dir, metadata_csv, user_message, work_root):
    """Step 2: 模板匹配，返回最佳label路径和元素信息"""
    try:
        d_label = Path(work_root).expanduser() / "02_label"
        d_label.mkdir(parents=True, exist_ok=True)
        metadata = load_metadata(Path(metadata_csv))
        user_elems = parse_elements_from_text(user_message)
        top_matches = match_one(recon_png, Path(label_dir), topk=3, min_area=5, max_dist=None, bin_width=5.0)
        best_label_name = refine_top1(top_matches, user_elems, metadata)
        src_label = Path(label_dir) / best_label_name
        dst_label = d_label / best_label_name
        shutil.copy2(src_label, dst_label)
        # 若用户没给元素 → 用 label 的 material_id 去 CSV 查
        if not user_elems:
            mid = extract_material_id(best_label_name)
            user_elems = sorted(metadata.get(mid, []))
        return {"success": True, "label_path": str(dst_label), "elements": user_elems}
    except Exception as e:
        return {"success": False, "error": str(e)}


def stem2cif_tool(label_path, elements, work_root, max_atoms=50, max_shrink_iter=4):
    '''
    Step 3: 图像→CIF结构，返回重建的cif和最终shrink CIF路径
    '''
    try:
        d_cif = Path(work_root).expanduser() / "03_recon_cif"
        d_cif.mkdir(parents=True, exist_ok=True)
        atoms = process_image(label_path, elements)
        if atoms is None or len(atoms) == 0:
            return {"success": False, "error": "process_image failed to detect atoms!"}
        import re, uuid
        mid_match = re.search(r"(2dm-\d+)", Path(label_path).stem, re.I)
        mid = mid_match.group(1).lower() if mid_match else f"tmp-{uuid.uuid4().hex[:6]}"
        cif_super = d_cif / f"{mid}_reconstructed.cif"
        write(cif_super, atoms, format="cif", wrap=False)
        cif_final = d_cif / "output_final.cif"
        shrink_or_window(cif_super, cif_final, MAX_NUM_ITER=max_shrink_iter, MAX_ATOMS_NUM=max_atoms)
        # 只返回cif路径，不返回atoms对象
        return {"success": True, "cif_path": str(cif_final)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def property_prediction_tool(cif_path, work_root, noise_amp=0.05, relax_steps=500, device="cuda"):
    """Step 4: 物性预测，返回能量、力、应力等"""
    try:
        d_relax = Path(work_root).expanduser() / "04_relax"
        d_relax.mkdir(parents=True, exist_ok=True)
        atoms_relax = read(cif_path)
        axis_min = span_min_axis(atoms_relax)
        import numpy as np
        atoms_relax.positions[:, axis_min] += noise_amp * np.random.randn(len(atoms_relax))
        atoms_relax.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
        relaxer = Relaxer(optimizer="BFGS", filter=None, constrain_symmetry=False)
        converged, atoms_relaxed = relaxer.relax(atoms_relax, steps=relax_steps)
        cif_relaxed = d_relax / "relaxed.cif"
        write(cif_relaxed, atoms_relaxed)
        E = atoms_relaxed.get_potential_energy()
        F0 = atoms_relaxed.get_forces()[0]
        sxx = atoms_relaxed.get_stress(voigt=False)[0][0]
        from ase.units import GPa
        return {
            "success": True,
            "relaxed_cif": str(cif_relaxed),
            "energy_eV": float(E),
            "energy_per_atom": float(E / len(atoms_relaxed)),
            "force_first_atom": F0.tolist(),
            "stress_xx_GPa": float(sxx / GPa),
            "converged": bool(converged)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def reconstruct_from_denoised_img(
        denoised_img: str,
        user_elements: List[str],
        coord: Optional[dict] = None,
        pixel_size: Optional[float] = 0.1,
        top_n: int = 3,
        out_dir: str = "pipeline_out",
        snap_hex: int = 1,
        auto_basis: int = 1,
        vec_tol: float = 0.02,
        vec_maxlen: float = 0.45,
        vec_cross_min: float = 0.0002,
        dedup_xy: float = 0.02,
        dedup_z: float = 0.05,
) -> Tuple[str, dict, list]:
    """Reconstruct a minimal cell directly from a denoised STEM image."""
    os.makedirs(out_dir, exist_ok=True)

    image = load_gray_image(denoised_img)
    fft_out_dir = os.path.join(out_dir, "fft")
    os.makedirs(fft_out_dir, exist_ok=True)
    fft_result = analyze_image(
        image,
        pixel_size=pixel_size,
        show=False,
        save=True,
        out_dir=fft_out_dir,
    )
    lattice_a = float(fft_result.a_physical) if fft_result.a_physical is not None else None
    lattice_b = float(fft_result.b_physical) if fft_result.b_physical is not None else None
    lattice_gamma = float(fft_result.alpha_deg)

    candidate_prefix = os.path.join(out_dir, "candidate_window")
    candidate_paths = generate_candidate_windows(
        image_path=denoised_img,
        elements=user_elements,
        coord=coord,
        lattice_params=(
            lattice_a if lattice_a else 3.0,
            lattice_b if lattice_b else 3.0,
            lattice_gamma,
        ),
        top_n=top_n,
        out_prefix=candidate_prefix,
    )
    if not candidate_paths:
        raise RuntimeError("No candidate CIF windows were generated")

    window_1 = candidate_paths[0]
    window_2 = candidate_paths[1] if len(candidate_paths) >= 2 else window_1
    window_3 = candidate_paths[2] if len(candidate_paths) >= 3 else window_1
    output_path = os.path.join(out_dir, "minimal_cell_from_top3_pipeline.cif")
    basis = None if coord is None else ", ".join(f"{key}:{value}" for key, value in coord.items())

    final_path, new_cell, basis_atoms = reconstruct_minimal_cell(
        win1=window_1,
        win2=window_2,
        win3=window_3,
        vec_tol=vec_tol,
        vec_maxlen=vec_maxlen,
        vec_cross_min=vec_cross_min,
        dedup_xy=dedup_xy,
        dedup_z=dedup_z,
        snap_hex=snap_hex,
        basis=basis,
        out=output_path,
        auto_basis=auto_basis,
        lat_a=lattice_a,
        lat_b=lattice_b,
        lat_gamma=lattice_gamma,
    )
    serializable_basis = [
        {
            **atom,
            "frac": atom["frac"].tolist() if hasattr(atom.get("frac"), "tolist") else atom.get("frac"),
        }
        for atom in basis_atoms
    ]
    return final_path, new_cell, serializable_basis


# ---------- CLI ---------- #
if __name__ == "__main__":
    import argparse

    default_image = os.environ.get("AUTOMAT_IMAGE_PATH")
    parser = argparse.ArgumentParser(description="DeepSeek‑V3 STEM→Property agent")
    parser.add_argument("--image",
                        default=default_image,
                        required=default_image is None,
                        help="big STEM image path or attachment")
    parser.add_argument("--workdir", default=os.environ.get("AUTOMAT_WORK_ROOT", "harness_runs/current"),
                        help="working root dir")
    parser.add_argument("--weights",
                        default=os.environ.get("AUTOMAT_WEIGHT_PATH", "MOE_model_weights/moe_model.ckpt"),
                        help="patch‑inference weight")
    parser.add_argument("--label-dir",
                        default=os.environ.get("AUTOMAT_LABEL_DIR", "data_generation/label"),
                        help="template label directory (*.png)")
    parser.add_argument("--meta-csv",
                        default=os.environ.get("AUTOMAT_METADATA_CSV", "baseline/property.csv"),
                        help="material_id‑elements csv")
    parser.add_argument("--user-text", default="请帮我分析这张图，元素: Al,Sb，剂量 30k", help="original user message text")
    args = parser.parse_args()

    info = run_agent_pipeline(
        image_path=args.image,
        user_message=args.user_text,
        work_root=args.workdir,
        weight_path=args.weights,
        label_dir=args.label_dir,
        metadata_csv=args.meta_csv,
    )

    print("\n=== Pipeline Finished ===")
    print(json.dumps(info, indent=4))
