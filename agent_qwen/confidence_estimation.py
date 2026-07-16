#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
confidence_estimation.py

STEM2CIF 置信度估计模块
=======================
在原子峰定位和晶格拟合阶段，原子重叠严重的区域通常表现出更高的拟合方差（残差）。
本模块将这些局部残差显式用作结构不确定性的定量代理指标。

核心思路
--------
1. **原子峰定位置信度**：
   - 对每个原子执行 2D Gaussian 拟合，记录拟合残差（RSS / R²）
   - 统计局部原子拥挤度（nearest-neighbor distance）
   - 将残差 & 拥挤度映射为 [0,1] 置信度分数

2. **晶格拟合置信度**：
   - FFT 峰到理想整数倍格点的偏差（inlier 残差）
   - 倒空间峰强度集中度（前 N 峰能量占总谱能量的比例）

3. **综合置信度报告**：
   - 每个原子的置信度分数
   - 全局结构置信度
   - 高不确定性区域标记

输出
----
一个 JSON 文件，包含：
  - per_atom_confidence: [{atom_id, element, x, y, gauss_r2, nn_distance,
                           crowding_score, confidence}, ...]
  - lattice_confidence:  {peak_inlier_ratio, mean_residual, spectral_concentration,
                           lattice_confidence}
  - global_confidence:   float
  - high_uncertainty_atoms: [atom_id, ...]

示例用法
--------
  python confidence_estimation.py \\
    --image examples/denoised_stem.png \\
    --elements Zr O \\
    --output results/confidence_report.json
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..')
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)

from ase.data import atomic_numbers

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────── 参数 ───────────────────
PIXEL_SIZE = 0.10        # Å/pixel
MIN_COMPONENT_AREA = 3   # 像素，面积低于此值的连通域视为噪声
MIN_MERGE_DISTANCE = 1.0 # Å，近于此距离的原子将被合并
CONFIDENCE_LOW_THRESHOLD = 0.5   # 低于此值标记为高不确定性


# ─────────────────── 2D Gaussian 模型 ───────────────────
def gaussian_2d(xy, A, x0, y0, sx, sy, offset):
    """2D 高斯函数，用于原子峰的子像素拟合。"""
    x, y = xy
    return A * np.exp(-(((x - x0)**2) / (2 * sx**2) +
                        ((y - y0)**2) / (2 * sy**2))) + offset


def fit_gaussian_with_diagnostics(xs, ys, intensities, x_init, y_init):
    """
    对单个原子峰执行 2D Gaussian 拟合，并返回详细诊断信息。

    Returns
    -------
    dict with keys:
        x_fit, y_fit      : 拟合后的亚像素中心坐标
        amplitude         : 拟合振幅
        sigma_x, sigma_y  : 拟合宽度参数
        r_squared         : 拟合的 R² (决定系数)
        rss               : 残差平方和
        fit_success       : bool, 拟合是否收敛
        residual_std      : 拟合残差的标准差
    """
    result = {
        'x_fit': float(x_init),
        'y_fit': float(y_init),
        'amplitude': 0.0,
        'sigma_x': 0.0,
        'sigma_y': 0.0,
        'r_squared': 0.0,
        'rss': float('inf'),
        'fit_success': False,
        'residual_std': float('inf'),
    }

    if len(xs) < 6:
        return result

    I = intensities.astype(np.float64)
    I_max, I_min = I.max(), I.min()
    if I_max - I_min < 1e-10:
        return result

    p0 = (I_max - I_min, float(x_init), float(y_init), 1.5, 1.5, float(I_min))
    bounds_lo = (0,        x_init - 5, y_init - 5, 0.3, 0.3, -np.inf)
    bounds_hi = (np.inf,   x_init + 5, y_init + 5, 10,  10,   np.inf)

    try:
        popt, pcov = curve_fit(
            gaussian_2d, (xs.astype(np.float64), ys.astype(np.float64)),
            I, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=3000
        )
        A_fit, xf, yf, sx, sy, off = popt

        # 计算预测值与残差
        I_pred = gaussian_2d((xs.astype(np.float64), ys.astype(np.float64)), *popt)
        residuals = I - I_pred
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((I - I.mean())**2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        # 参数标准误差（从协方差矩阵对角元素）
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(6, np.inf)

        result.update({
            'x_fit': float(xf),
            'y_fit': float(yf),
            'amplitude': float(A_fit),
            'sigma_x': float(abs(sx)),
            'sigma_y': float(abs(sy)),
            'r_squared': float(max(r2, 0.0)),
            'rss': float(ss_res),
            'fit_success': True,
            'residual_std': float(np.std(residuals)),
        })
    except Exception:
        pass

    return result


# ─────────────────── 合并过近原子 ───────────────────

def _merge_close_atom_records(records: List[Dict],
                              min_dist: float) -> List[Dict]:
    """
    合并距离小于 min_dist 的原子记录，与 batch_atoms_analysis.merge_close_atoms 对齐。
    合并规则：
      - 同元素：取位置平均，保留较高置信度指标
      - 异元素：保留重元素（Z 更大的）
    """
    if len(records) <= 1:
        return records
    positions = np.array([[r['x_angstrom'], r['y_angstrom']] for r in records])
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=min_dist)

    # Union-Find
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi
    for i, j in pairs:
        union(i, j)

    clusters = {}
    for i in range(len(records)):
        clusters.setdefault(find(i), []).append(i)

    merged = []
    for idxs in clusters.values():
        if len(idxs) == 1:
            merged.append(records[idxs[0]])
        else:
            # 取 Z 最大的元素
            best = max(idxs, key=lambda i: atomic_numbers.get(records[i]['element'], 0))
            rec = dict(records[best])
            # 位置取平均
            xs = [records[i]['x_angstrom'] for i in idxs]
            ys = [records[i]['y_angstrom'] for i in idxs]
            rec['x_angstrom'] = round(float(np.mean(xs)), 4)
            rec['y_angstrom'] = round(float(np.mean(ys)), 4)
            # R² 取最大值（最优拟合代表）
            rec['gauss_r2'] = max(records[i]['gauss_r2'] for i in idxs)
            rec['fit_success'] = any(records[i]['fit_success'] for i in idxs)
            merged.append(rec)

    # 重编号
    for i, rec in enumerate(merged):
        rec['atom_id'] = i
    return merged


# ─────────────────── 原子级置信度 ───────────────────

def compute_atom_level_confidence(image_path: str,
                                  elements_type: List[str]) -> List[Dict]:
    """
    对 STEM 图像中的每个检测到的原子峰计算置信度。

    置信度由三个分量组成：
      1. Gaussian 拟合质量 (R²)：R² 越高说明峰形越规则
      2. 拥挤度惩罚：最近邻距离越小，原子越拥挤，投影歧义越大
      3. 强度分类清晰度：单个原子的 KMeans 距离到聚类中心的远近

    Returns
    -------
    list of dict, 每个原子一个条目
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"{image_path} not found")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    rng = img.max() - img.min()
    if rng < 1e-8:
        return []
    img_norm = (img - img.min()) / rng
    img_u8 = (img_norm * 255).astype(np.uint8)

    # ── 连通域分析 ──
    _, binary = cv2.threshold(img_u8, 40, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return []

    flat_lbl = labels.ravel()
    flat_I = img_norm.ravel()
    sums_all = np.bincount(flat_lbl, weights=flat_I)[1:]
    counts_all = np.bincount(flat_lbl)[1:]

    # ── 面积过滤：去除噪声连通域 ──
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_mask = areas >= MIN_COMPONENT_AREA
    valid_labels = np.where(valid_mask)[0]  # 0-based index into [1..num_labels-1]
    if len(valid_labels) == 0:
        return []

    mean_intensity = sums_all[valid_labels] / np.maximum(counts_all[valid_labels], 1)

    # ── KMeans 元素分类 ──
    K = len(elements_type)
    vals = mean_intensity.reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(vals)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)[::-1]
    elems_sorted = sorted(elements_type, key=lambda s: atomic_numbers[s], reverse=True)
    cluster_to_elem = {int(cid): elems_sorted[i] for i, cid in enumerate(order)}

    # 计算每个原子到其所属聚类中心的距离（归一化），越小 → 分类越清晰
    dist_to_center = np.abs(vals.flatten() - centers[km.labels_])
    max_dist = dist_to_center.max() if dist_to_center.max() > 0 else 1.0
    classification_clarity = 1.0 - dist_to_center / max_dist  # [0, 1]

    # ── 逐原子 Gaussian 拟合 ──
    atom_records = []
    for vi, valid_idx in enumerate(valid_labels):
        lbl = valid_idx + 1  # label in [1..num_labels-1]
        elem = cluster_to_elem[int(km.labels_[vi])]
        x0 = stats[lbl, cv2.CC_STAT_LEFT]
        y0 = stats[lbl, cv2.CC_STAT_TOP]
        w  = stats[lbl, cv2.CC_STAT_WIDTH]
        h  = stats[lbl, cv2.CC_STAT_HEIGHT]
        sub = img_norm[y0:y0+h, x0:x0+w]
        mask = (labels[y0:y0+h, x0:x0+w] == lbl)
        ys_local, xs_local = np.where(mask)
        I_local = sub[ys_local, xs_local]

        if I_local.sum() <= 0:
            continue

        # 强度加权质心
        xC = float((xs_local * I_local).sum() / I_local.sum())
        yC = float((ys_local * I_local).sum() / I_local.sum())

        # Gaussian 拟合 + 诊断
        diag = fit_gaussian_with_diagnostics(xs_local, ys_local, I_local, xC, yC)

        # 物理坐标 (Å)
        x_phys = (x0 + diag['x_fit']) * PIXEL_SIZE
        y_phys = (y0 + diag['y_fit']) * PIXEL_SIZE

        atom_records.append({
            'atom_id': len(atom_records),
            'element': elem,
            'x_angstrom': round(x_phys, 4),
            'y_angstrom': round(y_phys, 4),
            'gauss_r2': round(diag['r_squared'], 4),
            'gauss_rss': round(diag['rss'], 6),
            'residual_std': round(diag['residual_std'], 6),
            'sigma_x': round(diag['sigma_x'], 4),
            'sigma_y': round(diag['sigma_y'], 4),
            'fit_success': diag['fit_success'],
            'classification_clarity': round(float(classification_clarity[vi]), 4),
        })

    if not atom_records:
        return []

    # ── 合并过近的原子（与原 pipeline 保持一致） ──
    atom_records = _merge_close_atom_records(atom_records, MIN_MERGE_DISTANCE)

    # ── 最近邻距离 → 拥挤度分数 ──
    positions = np.array([[a['x_angstrom'], a['y_angstrom']] for a in atom_records])
    if len(positions) >= 2:
        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=2)
        nn_dists = dists[:, 1]  # 第一列是自身 (=0)
    else:
        nn_dists = np.array([999.0])

    # 拥挤度评分：nn_distance 越小越拥挤，置信度越低
    # 使用 sigmoid 式映射：当 nn < 1.0 Å 时拥挤，nn > 3.0 Å 时充分分离
    nn_mid = 1.5   # Å, 中间点
    nn_scale = 1.0 # 斜率控制
    crowding_score = 1.0 / (1.0 + np.exp(-(nn_dists - nn_mid) / nn_scale))

    # ── 综合置信度 ──
    for i, rec in enumerate(atom_records):
        rec['nn_distance_angstrom'] = round(float(nn_dists[i]), 4)
        rec['crowding_score'] = round(float(crowding_score[i]), 4)

        # 加权综合: fit quality (0.4) + crowding (0.35) + classification clarity (0.25)
        r2_score = rec['gauss_r2'] if rec['fit_success'] else 0.0
        conf = (0.40 * r2_score +
                0.35 * crowding_score[i] +
                0.25 * rec['classification_clarity'])
        rec['confidence'] = round(float(np.clip(conf, 0.0, 1.0)), 4)

    return atom_records


# ─────────────────── 晶格拟合置信度 ───────────────────

def compute_lattice_confidence(image_path: str,
                               pixel_size: float = 0.10) -> Dict:
    """
    通过 FFT 分析评估晶格拟合的可靠性。

    指标：
      - peak_inlier_ratio : 拟合到整数倍格点内的峰占所有检测峰的比例
      - mean_residual     : 内点的平均残差（频域单位）
      - spectral_concentration : 前 N 峰能量占总谱能量的比例
      - lattice_confidence: 综合置信度 [0,1]
    """
    import importlib.util
    _fft_spec = importlib.util.spec_from_file_location(
        "fft_convert",
        os.path.join(_PROJECT_ROOT, "utils", "fft_convert.py"))
    _fft_mod = importlib.util.module_from_spec(_fft_spec)
    _fft_spec.loader.exec_module(_fft_mod)

    load_gray_image = _fft_mod.load_gray_image
    fft_power = _fft_mod.fft_power
    greedy_peak_pick = _fft_mod.greedy_peak_pick
    refine_peaks_subpixel = _fft_mod.refine_peaks_subpixel
    idx_to_freq = _fft_mod.idx_to_freq
    pair_half_plane = _fft_mod.pair_half_plane
    choose_two_directions = _fft_mod.choose_two_directions
    project_peaks_onto_line = _fft_mod.project_peaks_onto_line
    estimate_fundamental_spacing = _fft_mod.estimate_fundamental_spacing
    score_spacing_1d = _fft_mod.score_spacing_1d

    img = load_gray_image(str(image_path))
    P = fft_power(img)
    H, W = P.shape

    # 峰检测
    raw_peaks = greedy_peak_pick(P, k=80, r_exclude=8,
                                 center_exclude_radius=12,
                                 threshold_rel=0.12)
    if not raw_peaks:
        return _empty_lattice_result()

    refined = refine_peaks_subpixel(P, raw_peaks)
    freq_vecs = idx_to_freq(refined, (H, W))
    reps = pair_half_plane(freq_vecs)

    if reps is None or len(reps) < 2:
        return _empty_lattice_result()

    # 选两个方向
    u1, u2 = choose_two_directions(reps)
    if u1 is None or u2 is None:
        return _empty_lattice_result()

    # 沿每个方向投影，评估整数倍拟合残差
    proj1 = project_peaks_onto_line(reps, u1, perp_tol=0.02)
    proj2 = project_peaks_onto_line(reps, u2, perp_tol=0.02)

    k1 = estimate_fundamental_spacing(proj1) if proj1.size > 0 else None
    k2 = estimate_fundamental_spacing(proj2) if proj2.size > 0 else None

    total_peaks = len(reps)
    inlier_count_1, mean_res_1 = 0, 0.0
    inlier_count_2, mean_res_2 = 0, 0.0

    if k1 is not None and proj1.size > 0:
        tol1 = 0.12 * k1
        inlier_count_1, mean_res_1 = score_spacing_1d(proj1, k1, tol1)
    if k2 is not None and proj2.size > 0:
        tol2 = 0.12 * k2
        inlier_count_2, mean_res_2 = score_spacing_1d(proj2, k2, tol2)

    total_inliers = inlier_count_1 + inlier_count_2
    total_proj = len(proj1) + len(proj2)
    peak_inlier_ratio = total_inliers / max(total_proj, 1)
    mean_residual = (mean_res_1 + mean_res_2) / 2.0

    # 谱集中度：前 10 峰能量 / 总谱能量
    P_no_dc = P.copy()
    cy, cx = H // 2, W // 2
    P_no_dc[max(0, cy-12):cy+13, max(0, cx-12):cx+13] = 0
    total_energy = P_no_dc.sum()
    flat = P_no_dc.ravel()
    if flat.size > 10:
        top10 = np.partition(flat, -10)[-10:]
        spectral_concentration = float(top10.sum() / max(total_energy, 1e-12))
    else:
        spectral_concentration = 1.0

    # 综合晶格置信度
    lattice_conf = (0.5 * peak_inlier_ratio +
                    0.3 * spectral_concentration +
                    0.2 * max(0, 1.0 - mean_residual * 50))
    lattice_conf = float(np.clip(lattice_conf, 0.0, 1.0))

    return {
        'peak_inlier_ratio': round(peak_inlier_ratio, 4),
        'mean_residual': round(mean_residual, 6),
        'spectral_concentration': round(spectral_concentration, 4),
        'lattice_confidence': round(lattice_conf, 4),
        'num_peaks_detected': total_peaks,
        'num_inliers': total_inliers,
    }


def _empty_lattice_result():
    return {
        'peak_inlier_ratio': 0.0,
        'mean_residual': float('inf'),
        'spectral_concentration': 0.0,
        'lattice_confidence': 0.0,
        'num_peaks_detected': 0,
        'num_inliers': 0,
    }


# ─────────────────── 综合报告 ───────────────────

def generate_confidence_report(image_path: str,
                                elements_type: List[str],
                                pixel_size: float = 0.10,
                                output_json: Optional[str] = None) -> Dict:
    """
    生成完整的置信度报告。

    Parameters
    ----------
    image_path : str
        去噪后 STEM 图像路径
    elements_type : list of str
        元素列表, e.g. ['Mo', 'S']
    pixel_size : float
        像素尺寸 (Å/pixel)
    output_json : str or None
        若指定，将报告保存至此 JSON 文件

    Returns
    -------
    dict: 完整置信度报告
    """
    print(f"[Confidence] 分析图像: {image_path}")
    print(f"[Confidence] 元素列表: {elements_type}")

    # 1) 原子级置信度
    print("[Confidence] 步骤 1/3: 原子峰拟合置信度分析...")
    atom_records = compute_atom_level_confidence(image_path, elements_type)

    # 2) 晶格拟合置信度
    print("[Confidence] 步骤 2/3: 晶格拟合置信度分析...")
    lattice_result = compute_lattice_confidence(image_path, pixel_size)

    # 3) 综合
    print("[Confidence] 步骤 3/3: 汇总...")
    if atom_records:
        confs = [a['confidence'] for a in atom_records]
        atom_mean_conf = float(np.mean(confs))
        atom_median_conf = float(np.median(confs))
        atom_min_conf = float(np.min(confs))
    else:
        atom_mean_conf = atom_median_conf = atom_min_conf = 0.0

    lattice_conf = lattice_result.get('lattice_confidence', 0.0)

    # 全局置信度 = 原子置信度 (60%) + 晶格置信度 (40%)
    global_confidence = round(0.6 * atom_mean_conf + 0.4 * lattice_conf, 4)

    # 标记高不确定性原子
    high_uncertainty = [a['atom_id'] for a in atom_records
                        if a['confidence'] < CONFIDENCE_LOW_THRESHOLD]

    # 统计摘要
    summary = {
        'total_atoms_detected': len(atom_records),
        'atom_mean_confidence': round(atom_mean_conf, 4),
        'atom_median_confidence': round(atom_median_conf, 4),
        'atom_min_confidence': round(atom_min_conf, 4),
        'num_high_uncertainty_atoms': len(high_uncertainty),
        'fraction_high_uncertainty': round(len(high_uncertainty) / max(len(atom_records), 1), 4),
    }

    report = {
        'image_path': str(image_path),
        'elements': elements_type,
        'pixel_size_angstrom': pixel_size,
        'summary': summary,
        'global_confidence': global_confidence,
        'lattice_confidence': lattice_result,
        'high_uncertainty_atom_ids': high_uncertainty,
        'per_atom_confidence': atom_records,
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Confidence] 报告已保存至: {out_path}")

    print(f"[Confidence] ── 完成 ──")
    print(f"  全局置信度:        {global_confidence:.4f}")
    print(f"  原子平均置信度:    {atom_mean_conf:.4f}")
    print(f"  晶格置信度:        {lattice_conf:.4f}")
    print(f"  检测原子数:        {len(atom_records)}")
    print(f"  高不确定性原子数:  {len(high_uncertainty)} "
          f"({summary['fraction_high_uncertainty']*100:.1f}%)")

    return report


# ─────────────────── CLI ───────────────────

def main():
    parser = argparse.ArgumentParser(
        description='STEM2CIF 置信度估计：量化重建结构的不确定性')
    parser.add_argument('--image', required=True,
                        help='去噪后 STEM 图像路径')
    parser.add_argument('--elements', nargs='+', required=True,
                        help='元素列表，如 Mo S')
    parser.add_argument('--pixel-size', type=float, default=0.10,
                        help='像素尺寸 (Å/pixel), 默认 0.10')
    parser.add_argument('--output', default=None,
                        help='输出 JSON 路径 (可选)')
    args = parser.parse_args()

    report = generate_confidence_report(
        image_path=args.image,
        elements_type=args.elements,
        pixel_size=args.pixel_size,
        output_json=args.output,
    )

    # 打印 per-atom 摘要前 10 个
    print("\n─── 原子置信度示例 (前 10 个) ───")
    print(f"{'ID':>4s}  {'Elem':>4s}  {'x(Å)':>7s}  {'y(Å)':>7s}  "
          f"{'R²':>6s}  {'NN(Å)':>6s}  {'Crowd':>6s}  {'Conf':>6s}")
    for a in report['per_atom_confidence'][:10]:
        print(f"{a['atom_id']:4d}  {a['element']:>4s}  "
              f"{a['x_angstrom']:7.2f}  {a['y_angstrom']:7.2f}  "
              f"{a['gauss_r2']:6.3f}  {a['nn_distance_angstrom']:6.3f}  "
              f"{a['crowding_score']:6.3f}  {a['confidence']:6.3f}")


if __name__ == '__main__':
    main()
