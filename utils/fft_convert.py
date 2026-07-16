#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEM → FFT → 最小单胞 (supercell-aware) 参数估计
================================================
输入：去噪后的 STEM 灰度图（png/jpg/tif…）
输出（默认仅显示）：输入图、FFT(log)、峰标注；并在终端打印 a, b, α（像素及可选物理单位）
可选保存：加 --save 会额外保存 PNG/JSON/CSV 到 --out-dir

核心步骤：
1) 去均值 + 汉宁窗 → 2D FFT → log 功率谱
2) 功率谱峰检测 + 亚像素质心细化
3) 选两条不共线的“最近”倒易方向，并把所有峰投影到两方向直线
4) 沿每个方向做“整数倍对齐”投票，在 1×…K×中选能解释最多峰的候选，优先选“更大的间距”
   ——对应从 supercell 回退到**最小单胞**
5) 用 A = (G^{-1})^T 从倒易基 G 得到实空间晶格：a, b, α 以及方向角

用法：
  仅显示（推荐）：python stem_fft_inline.py --image your.png --pixel-size 0.205
  保存结果：       python stem_fft_inline.py --image your.png --pixel-size 0.205 --save --out-dir out_fft
无图时会跑合成 DEMO。
"""
import argparse, json, math, os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ------------------------------ 基础工具 ------------------------------
def load_gray_image(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode not in ('L', 'F', 'I;16', 'I'):
        im = im.convert('L')
    arr = np.array(im).astype(np.float32)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def hann2d(h: int, w: int) -> np.ndarray:
    hy = np.hanning(h)[:, None]
    hx = np.hanning(w)[None, :]
    return hy * hx

def fft_power(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    win = hann2d(h, w)
    wimg = (img - img.mean()) * win
    F = np.fft.fft2(wimg)
    Fs = np.fft.fftshift(F)
    P = (Fs * Fs.conj()).real
    return P

def log_image(P: np.ndarray) -> np.ndarray:
    Pn = P / (P.max() + 1e-12)
    return np.log1p(Pn)


# ------------------------------ 峰检测与细化 ------------------------------
def greedy_peak_pick(P: np.ndarray, k: int = 80, r_exclude: int = 8,
                     center_exclude_radius: int = 12, threshold_rel: float = 0.12) -> List[Tuple[int,int,float]]:
    H, W = P.shape
    cy, cx = H // 2, W // 2
    Pmask = P.copy()
    yy, xx = np.ogrid[:H, :W]
    mask_center = (yy - cy) ** 2 + (xx - cx) ** 2 <= center_exclude_radius ** 2
    Pmask[mask_center] = 0.0

    thr = float(Pmask.max()) * float(threshold_rel)
    if thr <= 0:
        return []
    coords = np.argwhere(Pmask >= thr)
    if coords.size == 0:
        return []
    vals = Pmask[coords[:, 0], coords[:, 1]]
    order = np.argsort(-vals)
    coords = coords[order]; vals = vals[order]

    picked: List[Tuple[int,int,float]] = []
    r2 = r_exclude * r_exclude
    for (r, c), v in zip(coords, vals):
        ok = True
        for (pr, pc, _) in picked:
            if (pr - r) ** 2 + (pc - c) ** 2 < r2:
                ok = False; break
        if ok:
            picked.append((int(r), int(c), float(v)))
        if len(picked) >= k:
            break
    return picked

def refine_peaks_subpixel(P: np.ndarray, peaks_rcv: List[Tuple[int,int,float]], win: int = 4):
    H, W = P.shape
    out = []
    for r, c, v in peaks_rcv:
        r0 = max(0, r - win); r1 = min(H, r + win + 1)
        c0 = max(0, c - win); c1 = min(W, c + win + 1)
        patch = P[r0:r1, c0:c1].astype(np.float64)
        if patch.size == 0 or patch.sum() <= 0:
            out.append((float(r), float(c), float(v)))
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        wsum = patch.sum()
        r_sub = (rr * patch).sum() / wsum
        c_sub = (cc * patch).sum() / wsum
        v_sub = patch.max()
        out.append((float(r_sub), float(c_sub), float(v_sub)))
    return out

def idx_to_freq(peaks, shape):
    H, W = shape
    cy, cx = H // 2, W // 2
    vecs = []
    for r, c, v in peaks:
        ky = (r - cy) / H
        kx = (c - cx) / W
        vecs.append([ky, kx, v])
    return np.array(vecs, dtype=np.float64)

def pair_half_plane(vecs, eps=1e-3):
    if vecs.size == 0:
        return vecs
    used = np.zeros(len(vecs), dtype=bool)
    reps = []
    for i in range(len(vecs)):
        if used[i]: continue
        vi = vecs[i, :2]
        for j in range(i + 1, len(vecs)):
            if used[j]: continue
            vj = vecs[j, :2]
            if np.linalg.norm(vi + vj) < eps:
                keep = i
                if vi[0] < 0 or (abs(vi[0]) < eps and vi[1] < 0):
                    keep = j
                reps.append(vecs[keep])
                used[i] = used[j] = True
                break
        else:
            if vi[0] > 0 or (abs(vi[0]) < eps and vi[1] >= 0):
                reps.append(vecs[i]); used[i] = True
    reps = np.array(reps)
    if reps.size == 0:
        return reps
    norms = np.linalg.norm(reps[:, :2], axis=1)
    order = np.argsort(norms)
    return reps[order]


# ------------------------------ 方向与超胞→最小单胞 ------------------------------
def choose_two_directions(reps: np.ndarray, min_angle_deg: float = 10.0):
    if reps is None or len(reps) < 2:
        return None, None
    V = reps[:, :2]
    norms = np.linalg.norm(V, axis=1)
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            v1 = V[i]; v2 = V[j]
            n1 = norms[i]; n2 = norms[j]
            if n1 == 0 or n2 == 0: continue
            cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            ang = math.degrees(math.acos(cosang))
            if min_angle_deg <= ang <= 180 - min_angle_deg:
                return v1, v2
    return None, None

def project_peaks_onto_line(vecs: np.ndarray, u: np.ndarray, perp_tol: float = 0.02) -> np.ndarray:
    V = vecs[:, :2]
    u = u / (np.linalg.norm(u) + 1e-12)
    t = V @ u
    perp = np.linalg.norm(V - np.outer(t, u), axis=1)
    keep = perp <= perp_tol
    vals = np.abs(t[keep])
    vals = vals[vals > 1e-6]
    return np.sort(vals)

def score_spacing_1d(projs: np.ndarray, delta: float, tol: float):
    if projs.size == 0 or delta <= 0:
        return 0, 1e9
    m = np.rint(projs / delta)      # 取最近整数倍
    fit = m * delta
    resid = np.abs(projs - fit)
    inliers = resid <= tol
    if inliers.sum() == 0:
        return 0, 1e9
    return int(inliers.sum()), float(resid[inliers].mean())

def estimate_fundamental_spacing(projs: np.ndarray, max_multiple: int = 8, tol_frac: float = 0.12):
    """
    从最小非零投影 s_min 出发，测试 k*s_min (k=1..max_multiple)：
    - 内点越多越好；
    - 内点并列时，优先“更大的间距”（→ 实空间更小单胞，去掉 supercell）
    - 再以残差更小优先
    """
    if projs.size == 0:
        return None
    smin = float(np.min(projs))
    best = None
    for k in range(1, max_multiple + 1):
        delta = k * smin
        tol = tol_frac * delta
        count, mean_res = score_spacing_1d(projs, delta, tol)
        if best is None:
            best = (count, -delta, mean_res, delta)
        else:
            if (count > best[0]) or (count == best[0] and delta > -best[1]) or (count == best[0] and delta == -best[1] and mean_res < best[2]):
                best = (count, -delta, mean_res, delta)
    return None if best is None else best[3]

def build_G_from_directions(u1: np.ndarray, u2: np.ndarray, k1: float, k2: float) -> np.ndarray:
    u1 = u1 / (np.linalg.norm(u1) + 1e-12)
    u2 = u2 / (np.linalg.norm(u2) + 1e-12)
    g1 = k1 * u1
    g2 = k2 * u2
    return np.stack([g1, g2], axis=1)   # 2x2，列为 g1,g2


# ------------------------------ 倒易→实空间 ------------------------------
@dataclass
class LatticeResult:
    a_pixels: float
    b_pixels: float
    alpha_deg: float
    a_dir_deg: float
    b_dir_deg: float
    pixel_size: Optional[float] = None
    a_physical: Optional[float] = None
    b_physical: Optional[float] = None
    meta: Optional[dict] = None

def reciprocal_to_real(G: np.ndarray, pixel_size: Optional[float]) -> LatticeResult:
    if abs(np.linalg.det(G)) < 1e-9:
        raise ValueError("Reciprocal basis is degenerate (det ≈ 0).")
    A = np.linalg.inv(G).T  # A = (G^{-1})^T
    a1 = A[:, 0]; a2 = A[:, 1]
    a = float(np.linalg.norm(a1)); b = float(np.linalg.norm(a2))
    cosal = float(np.clip(np.dot(a1, a2) / (a * b + 1e-12), -1.0, 1.0))
    alpha = math.degrees(math.acos(cosal))
    # 方向角（与频域坐标系对应）：以 +x（列）为 0°，逆时针为正
    a_dir = math.degrees(math.atan2(a1[0], a1[1]))
    b_dir = math.degrees(math.atan2(a2[0], a2[1]))
    res = LatticeResult(a_pixels=a, b_pixels=b, alpha_deg=alpha, a_dir_deg=a_dir, b_dir_deg=b_dir)
    if pixel_size is not None:
        res.pixel_size = float(pixel_size)
        res.a_physical = a * pixel_size
        res.b_physical = b * pixel_size
    return res


# ------------------------------ 可视化与主流程 ------------------------------
def show_image(img: np.ndarray, title: str):
    plt.figure(figsize=(6,6)); plt.imshow(img, cmap='gray'); plt.axis('off'); plt.title(title); plt.tight_layout(); plt.show()

def analyze_image(img: np.ndarray,
                  pixel_size: Optional[float],
                  peak_k: int = 80,
                  peak_dist: int = 8,
                  center_mask: int = 12,
                  peak_thr: float = 0.12,
                  perp_tol: float = 0.02,
                  max_multiple: int = 8,
                  tol_frac: float = 0.12,
                  show: bool = True,
                  save: bool = False,
                  out_dir: str = "fft_unitcell_out") -> LatticeResult:

    if show:
        show_image(img, 'Input STEM (denoised)')

    P = fft_power(img)
    P_log = log_image(P)
    if show:
        show_image(P_log, 'FFT power (log)')

    peaks = greedy_peak_pick(P, k=peak_k, r_exclude=peak_dist,
                             center_exclude_radius=center_mask, threshold_rel=peak_thr)
    peaks_ref = refine_peaks_subpixel(P, peaks, win=4)

    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(P_log, cmap='gray'); plt.axis('off'); plt.title('FFT peaks (refined)')
        for r, c, _ in peaks_ref:
            plt.plot(c, r, 'o', markersize=3)
        plt.tight_layout(); plt.show()

    vecs = idx_to_freq(peaks_ref, P.shape)          # (N,3): fy,fx,val
    reps = pair_half_plane(vecs, eps=2e-3)
    g1, g2 = choose_two_directions(reps, min_angle_deg=10.0)
    if g1 is None or g2 is None:
        raise RuntimeError("未找到两条不共线的倒易方向；请调小 peak_thr 或增大 center_mask。")

    u1 = g1[:2] / (np.linalg.norm(g1[:2]) + 1e-12)
    u2 = g2[:2] / (np.linalg.norm(g2[:2]) + 1e-12)

    projs1 = project_peaks_onto_line(vecs, u1, perp_tol=perp_tol)
    projs2 = project_peaks_onto_line(vecs, u2, perp_tol=perp_tol)
    k1 = estimate_fundamental_spacing(projs1, max_multiple=max_multiple, tol_frac=tol_frac)
    k2 = estimate_fundamental_spacing(projs2, max_multiple=max_multiple, tol_frac=tol_frac)
    if k1 is None or k2 is None or k1 <= 0 or k2 <= 0:
        raise RuntimeError("无法稳定估计两个方向的基准间距；请调整 max_multiple / tol_frac / perp_tol。")

    G = build_G_from_directions(u1, u2, k1, k2)
    res = reciprocal_to_real(G, pixel_size=pixel_size)

    # 终端打印
    print("=== 最小单胞估计 ===")
    print(f"a (px): {res.a_pixels:.6f}")
    print(f"b (px): {res.b_pixels:.6f}")
    print(f"alpha (deg): {res.alpha_deg:.6f}")
    print(f"a_dir (deg): {res.a_dir_deg:.3f}")
    print(f"b_dir (deg): {res.b_dir_deg:.3f}")
    if pixel_size is not None:
        print(f"a (phys): {res.a_physical:.6f}")
        print(f"b (phys): {res.b_physical:.6f}")
        print(f"pixel_size: {res.pixel_size} per px")

    # 可选保存
    if save:
        os.makedirs(out_dir, exist_ok=True)
        # 保存图
        plt.figure(figsize=(6,6)); plt.imshow(img, cmap='gray'); plt.axis('off'); plt.title('Input STEM'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'input.png')); plt.close()

        plt.figure(figsize=(6,6)); plt.imshow(P_log, cmap='gray'); plt.axis('off'); plt.title('FFT power (log)'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fft_log.png')); plt.close()

        plt.figure(figsize=(6,6)); plt.imshow(P_log, cmap='gray'); plt.axis('off'); plt.title('FFT peaks (refined)')
        for r, c, _ in peaks_ref: plt.plot(c, r, 'o', markersize=3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'fft_peaks.png')); plt.close()

        # 保存参数
        with open(os.path.join(out_dir, 'lattice_result.json'), 'w', encoding='utf-8') as f:
            json.dump(asdict(res), f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, 'lattice_result.csv'), 'w', encoding='utf-8') as f:
            f.write('a_pixels,b_pixels,alpha_deg,a_dir_deg,b_dir_deg,pixel_size,a_physical,b_physical\n')
            row = [res.a_pixels, res.b_pixels, res.alpha_deg, res.a_dir_deg, res.b_dir_deg,
                   res.pixel_size, res.a_physical, res.b_physical]
            f.write(','.join('' if v is None else f'{v:.6f}' for v in row) + '\n')
        with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
            f.write("=== STEM FFT Unit-Cell Estimation ===\n")
            f.write(f"a (px): {res.a_pixels:.6f}\n")
            f.write(f"b (px): {res.b_pixels:.6f}\n")
            f.write(f"alpha (deg): {res.alpha_deg:.6f}\n")
            f.write(f"a_dir (deg): {res.a_dir_deg:.3f}\n")
            f.write(f"b_dir (deg): {res.b_dir_deg:.3f}\n")
            if pixel_size is not None:
                f.write(f"a (phys): {res.a_physical:.6f}\n")
                f.write(f"b (phys): {res.b_physical:.6f}\n")
                f.write(f"pixel_size: {pixel_size} per px\n")

    return res


# ------------------------------ 合成 DEMO（无图时） ------------------------------
from math import cos, sin, radians
def make_synthetic_lattice(size=(512,512), a_px=22.0, b_px=28.0, alpha_deg=90.0, blur_sigma=1.0, seed=1):
    rng = np.random.default_rng(seed)
    H, W = size
    y = np.arange(H); x = np.arange(W)
    X, Y = np.meshgrid(x, y)
    a = np.array([a_px, 0.0])
    b = np.array([b_px*cos(radians(alpha_deg)), b_px*sin(radians(alpha_deg))])
    max_extent = int(2 * max(H / min(a_px, b_px), W / min(a_px, b_px)) + 4)
    img = np.zeros((H, W), dtype=np.float32)
    for i in range(-max_extent, max_extent+1):
        for j in range(-max_extent, max_extent+1):
            px, py = i*a[0] + j*b[0], i*a[1] + j*b[1]
            if -50 <= px <= W+50 and -50 <= py <= H+50:
                dx = X - px; dy = Y - py
                spot = np.exp(-(dx*dx + dy*dy) / (2.0*blur_sigma*blur_sigma))
                img += spot
    img = img / img.max()
    img += 0.05 * rng.standard_normal(img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    return img


# ------------------------------ CLI ------------------------------
def main():
    ap = argparse.ArgumentParser(description="STEM FFT 最小单胞参数估计（显示为主，可选保存）")
    ap.add_argument('--image', type=str, default=None, help='去噪后的 STEM 图像路径（若缺省，则跑合成 DEMO）')
    ap.add_argument('--pixel-size', type=float, default=0.1, help='每像素物理尺寸（如 0.205 表示 Å/px）')
    ap.add_argument('--save', action='store_true', default=True, help='保存 PNG/JSON/CSV 到 --out-dir')
    ap.add_argument('--out-dir', type=str, default='fft_unitcell_out_3763', help='保存目录（仅在 --save 时使用）')
    # 下面是可调参数（一般不用改）
    ap.add_argument('--peak-k', type=int, default=80)
    ap.add_argument('--peak-dist', type=int, default=8)
    ap.add_argument('--center-mask', type=int, default=12)
    ap.add_argument('--peak-thr', type=float, default=0.12)
    ap.add_argument('--perp-tol', type=float, default=0.02)
    ap.add_argument('--max-multiple', type=int, default=8)
    ap.add_argument('--tol-frac', type=float, default=0.12)
    args = ap.parse_args()

    if args.image is None:
        print("未提供 --image，运行合成 DEMO（正交格子 a=22px, b=28px, α=90°）")
        img = make_synthetic_lattice()
    else:
        img = load_gray_image(args.image)

    analyze_image(img,
                  pixel_size=args.pixel_size,
                  peak_k=args.peak_k,
                  peak_dist=args.peak_dist,
                  center_mask=args.center_mask,
                  peak_thr=args.peak_thr,
                  perp_tol=args.perp_tol,
                  max_multiple=args.max_multiple,
                  tol_frac=args.tol_frac,
                  show=True,
                  save=args.save,
                  out_dir=args.out_dir)

if __name__ == '__main__':
    main()
