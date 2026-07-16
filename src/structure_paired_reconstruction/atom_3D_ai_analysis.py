#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iDPC-STEM → 3 D CIF        （Si / O  + 质心合并 + 二次去重）

• 单独原子      → z = 1.5 Å
• Si/O 重叠同位 → Si:1.5 Å , O:3.0 Å
• 最后再次检查：两个原子 xy 距离 < TOL_ANG 时，保留较重元素
"""

import cv2, numpy as np, warnings, os, json
from scipy.ndimage import label
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN, KMeans
from ase import Atoms
from ase.io import write
from ase.data import atomic_numbers

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========= 可调参数 =========
PIXEL_SIZE   = 0.10          # Å / pixel
ELEMENTS     = ["Si", "O"]   # 降序写即可  (Si 原子序 14 > O 8)
CELL_C       = 5.0           # Å   (z 非周期)
TOL_PIX_Z    = 2.0           # 像素；Si/O 距离 ≤ 此值判定“重叠”
Z_DEFAULT    = 1.5           # Å
Z_OVERLAP    = {"Si": 1.5, "O": 3.0}

MIN_AREA_PIX = 5             # 连通域<5 像素跳过拟合
MERGE_PIX    = 10            # 像素；同元素质心 ≤ MERGE_PIX → 合并
# —— 二次去重阈值 (Å) ——
TOL_ANG      = MERGE_PIX * PIXEL_SIZE
# ============================


# ---------- K-Means 掩膜 ----------
def kmeans_masks(gray, n_elem):
    n_clusters = n_elem + 1
    flat = gray.reshape(-1, 1).astype(np.float32)
    lbl = KMeans(n_clusters, random_state=0).fit(flat).labels_.reshape(gray.shape)
    uniq, cnt = np.unique(lbl, return_counts=True)
    bg = uniq[np.argmax(cnt)]            # 像素最多的簇当背景
    masks, avg = [], []
    for lab in uniq:
        if lab == bg:
            continue
        m = (lbl == lab)
        masks.append(m)
        avg.append(gray[m].mean())
    order = np.argsort(avg)[::-1]        # 亮度降序 ↔ 元素序降序
    return [masks[i] for i in order]


# ---------- 连通域 → 质心 ----------
def gauss2d(coord, A, x0, y0, s, c):
    x, y = coord
    return c + A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * s**2))


def refine_centroids(mask, gray):
    lbl, n = label(mask)
    out = []
    for rid in range(1, n + 1):
        ys, xs = np.where(lbl == rid)
        ints = gray[ys, xs].astype(np.float64)
        if ints.sum() == 0:
            continue
        cx = np.sum(xs * ints) / ints.sum()
        cy = np.sum(ys * ints) / ints.sum()
        area = len(xs)
        if area >= MIN_AREA_PIX:
            p0 = [ints.ptp(), cx, cy, np.sqrt(area / np.pi), ints.min()]
            try:
                popt, _ = curve_fit(gauss2d, (xs, ys), ints.ravel(), p0=p0, maxfev=2000)
                _, cx, cy, _, _ = popt
            except Exception:
                pass
        out.append([cx, cy, ints.sum()])
    return out


def merge_close(points, eps_pix):
    if not points:
        return []
    xy   = np.array([[p[0], p[1]] for p in points])
    wgt  = np.array([p[2] for p in points])
    lab  = DBSCAN(eps=eps_pix, min_samples=1).fit(xy).labels_
    merged = []
    for l in np.unique(lab):
        idx = np.where(lab == l)[0]
        if len(idx) == 1:
            merged.append([*xy[idx[0]], wgt[idx[0]]])
        else:
            merged.append([
                np.average(xy[idx, 0], weights=wgt[idx]),
                np.average(xy[idx, 1], weights=wgt[idx]),
                wgt[idx].sum()
            ])
    return merged


# ---------- 质心 → Atoms ----------
def build_atoms(elem_pts, shape):
    Si_c, O_c = elem_pts["Si"], elem_pts["O"]
    used_o = set()
    sym, coord = [], []

    # 重叠 Si+O
    for sx, sy, _ in Si_c:
        found = False
        for j, (ox, oy, _) in enumerate(O_c):
            if j in used_o:
                continue
            if np.hypot(sx - ox, sy - oy) <= TOL_PIX_Z:
                xm, ym = (sx + ox)/2, (sy + oy)/2
                sym += ["Si", "O"]
                coord += [(xm, ym, Z_OVERLAP["Si"]),
                          (xm, ym, Z_OVERLAP["O"])]
                used_o.add(j)
                found = True
                break
        if not found:
            sym.append("Si")
            coord.append((sx, sy, Z_DEFAULT))

    # 剩余 O
    for j, (ox, oy, _) in enumerate(O_c):
        if j not in used_o:
            sym.append("O")
            coord.append((ox, oy, Z_DEFAULT))

    # 像素 → Å
    pos = [(x*PIXEL_SIZE, y*PIXEL_SIZE, z) for x, y, z in coord]
    h, w = shape
    cell = [w*PIXEL_SIZE, h*PIXEL_SIZE, CELL_C]
    return Atoms(symbols=sym, positions=pos, cell=cell, pbc=(True, True, False))


# ---------- 二次去重 ----------
def deduplicate_atoms(atoms, tol_ang=TOL_ANG):
    coords = atoms.get_positions()[:, :2]          # 只看 xy
    symbs  = np.array(atoms.get_chemical_symbols())
    order  = np.argsort([-atomic_numbers[s] for s in symbs])  # 重元素优先
    keep   = np.ones(len(atoms), dtype=bool)

    for i in order:                                # 遍历重→轻
        if not keep[i]:
            continue
        dxy = coords - coords[i]
        dist = np.linalg.norm(dxy, axis=1)
        clash = (dist < tol_ang) & keep            # 与已保留原子冲突
        clash[i] = False
        keep[clash] = False                       # 舍弃轻元素

    return atoms[keep]


# ---------- 主流程 ----------
def process_image(img_path, out_cif="reconstructed.cif"):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(img_path)
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    masks = kmeans_masks(img, len(ELEMENTS))

    elem_pts = {}
    for elem, mask in zip(ELEMENTS, masks):
        pts = refine_centroids(mask, img)
        pts = merge_close(pts, MERGE_PIX)
        elem_pts[elem] = pts

    atoms = build_atoms(elem_pts, img.shape)
    atoms = deduplicate_atoms(atoms)               # <-- ❹ 二次去重

    write(out_cif, atoms, format="cif", wrap=False)
    print(f"✔ {out_cif}  ({len(atoms)} atoms)")
    return atoms


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="STEM → 3D CIF (去重版)")
    ap.add_argument("--image", default='./large_area4.png', help="灰度图路径")
    ap.add_argument("-o", "--out", default="reconstructed.cif")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit("❌ image not found")
    atoms = process_image(args.image, args.out)
    u, c = np.unique(atoms.get_chemical_symbols(), return_counts=True)
    print(json.dumps(dict(zip(u, map(int, c))), indent=2))
