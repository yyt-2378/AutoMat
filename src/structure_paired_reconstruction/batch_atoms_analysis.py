#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_reconstruct.py

批量并行处理去噪后的 STEM 图像，自动重建周期性结构并导出 CIF 文件。
从图像文件名中提取 material_id，读取 CSV 中对应的元素列表，对每个图像并行执行：
- 聚类 + 拟合流程提取原子位置
- 合并近邻原子
- 尝试原胞缩减；若失败则提取最佳窗口
结果按 material_id 保存到指定输出文件夹。

用法示例：
  python batch_reconstruct.py \
    --input-dir examples/denoised_images \
    --elements-csv examples/materials.csv \
    --output-dir results/output_cifs \
    --workers 8
"""
import os
# 强制所有科学计算库只用 1 线程
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import sys
sys.path.append('./structure_recongnition')
import warnings
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import write
from pymatgen.core import Structure
from ase.neighborlist import neighbor_list
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from shrink_cif import load_structure, extract_and_write, find_best_window

warnings.filterwarnings("ignore", category=RuntimeWarning)

# 通用参数
PIXEL_SIZE = 0.10           # Å/pixel
MIN_MERGE_DISTANCE = 1.0    # Å
MAX_NUM_ITER = 4
MAX_ATOMS_NUM = 20


def _pbc_mean(cart_coords, cell):
    ref = cart_coords[0]
    shifted = []
    for p in cart_coords:
        d = p - ref
        for ax in range(3):
            L = cell[ax, ax]
            if L == 0: continue
            if d[ax] >  L/2: d[ax] -= L
            if d[ax] < -L/2: d[ax] += L
        shifted.append(ref + d)
    return np.mean(shifted, axis=0)


def merge_close_atoms(atoms, min_dist=1.0):
    idx_i, idx_j = neighbor_list("ij", atoms, cutoff=min_dist*0.999)
    parent = list(range(len(atoms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i,j):
        pi, pj = find(i), find(j)
        if pi != pj: parent[pj] = pi
    for i,j in zip(idx_i, idx_j):
        union(i, j)
    clusters = {}
    for i in range(len(atoms)):
        clusters.setdefault(find(i), []).append(i)
    pos = atoms.get_positions()
    syms = np.array(atoms.get_chemical_symbols())
    Z = np.array([atomic_numbers[s] for s in syms])
    cell = atoms.get_cell().array
    new_pos, new_sym = [], []
    for idxs in clusters.values():
        if len(idxs) == 1:
            i = idxs[0]
            new_pos.append(pos[i]); new_sym.append(syms[i])
        else:
            sub_syms = syms[idxs]
            if np.all(sub_syms == sub_syms[0]):
                merged = _pbc_mean(pos[idxs], cell)
                new_pos.append(merged); new_sym.append(sub_syms[0])
            else:
                heavy = idxs[np.argmax(Z[idxs])]
                new_pos.append(pos[heavy]); new_sym.append(syms[heavy])
    m = Atoms(symbols=list(new_sym), positions=new_pos)
    m.set_cell(atoms.get_cell()); m.set_pbc(atoms.get_pbc())
    return m


def gaussian_2d(xy, A, x0, y0, sx, sy, offset):
    x, y = xy
    return A * np.exp(-(((x-x0)**2)/(2*sx**2) + ((y-y0)**2)/(2*sy**2))) + offset


def process_image(image_path, elements_type):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"{image_path} not found")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() - img.min() < 1e-8:
        img_norm = np.zeros_like(img)
    else:
        img_norm = (img - img.min()) / (img.max() - img.min())
    img_den = img_norm  # 已去噪

    img_u8 = (img_den * 255).astype(np.uint8)
    _, binary = cv2.threshold(img_u8, 40, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    flat_lbl = labels.ravel(); flat_I = img_den.ravel()
    sums = np.bincount(flat_lbl, weights=flat_I); counts = np.bincount(flat_lbl)
    sums, counts = sums[1:], counts[1:]
    if len(sums) == 0:
        return None
    mean_intensity = sums / np.maximum(counts, 1)

    K = len(elements_type)
    vals = mean_intensity.reshape(-1,1)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(vals)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)[::-1]
    elems = sorted(elements_type, key=lambda s: atomic_numbers[s], reverse=True)
    cluster_to_elem = {int(cid): elems[i] for i, cid in enumerate(order)}

    coords, syms = [], []
    for lbl in range(1, num_labels):
        sym = cluster_to_elem[int(km.labels_[lbl-1])]
        x0, y0 = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP]
        w, h = stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
        sub = img_den[y0:y0+h, x0:x0+w]
        mask = (labels[y0:y0+h, x0:x0+w] == lbl)
        ys, xs = np.where(mask)
        I = sub[ys, xs]
        if I.sum() <= 0: continue
        xC = (xs*I).sum() / I.sum(); yC = (ys*I).sum() / I.sum()
        if len(xs) < 6:
            xf, yf = xC, yC
        else:
            p0 = (I.max()-I.min(), xC, yC, 1.5, 1.5, I.min())
            try:
                popt, _ = curve_fit(gaussian_2d, (xs, ys), I, p0=p0, maxfev=2000)
                _, xf, yf, _, _, _ = popt
            except:
                xf, yf = xC, yC
        coords.append((x0+xf, y0+yf))
        syms.append(sym)

    cell_c = 5.0; base_z = 0.5
    atoms_pos = [(x*PIXEL_SIZE, y*PIXEL_SIZE, base_z+cell_c/2) for x,y in coords]
    atoms = Atoms(symbols=syms, positions=atoms_pos)
    h, w = img.shape
    atoms.set_cell([[w*PIXEL_SIZE,0,0],[0,h*PIXEL_SIZE,0],[0,0,cell_c]])
    atoms.set_pbc((True, True, False))
    atoms = merge_close_atoms(atoms, MIN_MERGE_DISTANCE)
    return atoms


def try_primitive(struct):
    for tol in (0.1, 0.25):
        try:
            cand = struct.get_primitive_structure(tolerance=tol)
        except Exception:
            continue
        if len(cand) < len(struct):
            return cand
    return None


def shrink_once(struct):
    new = try_primitive(struct)
    return new


def shrink_or_window(atoms, out_cif_path):
    tmp = out_cif_path.with_suffix(".tmp.cif")
    write(tmp, atoms, format="cif", wrap=False)
    cur = Structure.from_file(str(tmp))
    for step in range(1, MAX_NUM_ITER):
        if len(cur) <= MAX_ATOMS_NUM:
            print(f"≤{MAX_ATOMS_NUM} atoms，停止。")
            cur.to(filename=str(out_cif_path))
            return
        nxt = shrink_once(cur)
        if nxt is None or len(nxt) >= len(cur):
            # fallback to window
            print("已无法进一步缩减。启用局部周期性结构寻找")
            atoms_ase, pos_xy, cell_abc = load_structure(str(tmp))
            a, b, _ = cell_abc
            area = a*b
            (bx, by), _ = find_best_window(pos_xy, (a, b), area)
            extract_and_write(atoms_ase, pos_xy, (bx, by), output_cif=str(out_cif_path))
            return
        cur = nxt
    # for step in range(1, MAX_NUM_ITER):
    #     if len(cur) <= MAX_ATOMS_NUM:
    #         print(f"≤{MAX_ATOMS_NUM} atoms，停止。")
    #         break
    #     nxt = shrink_once(cur)
    #     if nxt is None or len(nxt) >= len(cur):
    #         print("已无法进一步缩减。")
    #         break
    #     print(f"Step {step}: {len(cur)} → {len(nxt)} atoms")
    #     cur = nxt
    # cur.to(filename=str(out_cif_path))


def worker(task):
    img_path_str, elements_list, out_dir_str = task
    img_path = Path(img_path_str)
    out_dir = Path(out_dir_str)
    mid_match = re.search(r"(2dm-\d+)", img_path.stem)
    if not mid_match:
        return img_path.name, False, "bad filename"
    mid = mid_match.group(1)
    try:
        atoms = process_image(img_path, elements_list)
        if atoms is None or len(atoms) == 0:
            return img_path.name, False, "no atoms"
        out_cif = out_dir / f"{mid}_reconstructed.cif"
        shrink_or_window(atoms, out_cif)
        return img_path.name, True, out_cif.name
    except Exception as e:
        return img_path.name, False, str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="并行批量图像→CIF 重建")
    parser.add_argument("--input-dir",  required=True, help="去噪图像目录")
    parser.add_argument("--elements-csv", required=True, help="CSV，含 material_id,elements 列")
    parser.add_argument("--output-dir", required=True, help="输出 CIF 目录")
    parser.add_argument("--workers", type=int, default=48, help="并行进程数")
    args = parser.parse_args()
    print(multiprocessing.cpu_count()-1)

    df = pd.read_csv(args.elements_csv, dtype=str)
    mapping = {row.material_id: [e.strip() for e in row.elements.split(",")]
               for _, row in df.iterrows()}

    img_paths = list(Path(args.input_dir).glob("*.png"))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for p in img_paths:
        m = re.search(r"(2dm-\d+)", p.stem)
        if not m: continue
        mid = m.group(1)
        elems = mapping.get(mid)
        print(elems)
        if not elems: continue
        tasks.append((str(p), elems, str(out_dir)))

    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            name, ok, info = fut.result()
            if ok:
                print(f"✔ {name} → {info}")
            else:
                print(f"✖ {name}: {info}")


if __name__ == "__main__":
    main()
