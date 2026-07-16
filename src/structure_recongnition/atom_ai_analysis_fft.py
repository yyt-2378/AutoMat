import os
import sys
import cv2
import warnings
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import write, read
from ase.neighborlist import neighbor_list, natural_cutoffs
from pymatgen.core import Structure, Lattice
from mpl_toolkits.mplot3d import Axes3D
from shrink_cif import (load_structure, extract_and_write, find_multiple_best_windows, 
                       find_best_window, find_optimal_unit_cell,
                       extract_and_write_with_lattice_constraint, 
                       analyze_lattice_compatibility,
                       save_all_candidate_windows,
                       save_window_analysis_report,)
import argparse

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========= 可调参数 =========
PIXEL_SIZE = 0.10           # Å/pixel
ELEMENTS_TYPE = ['Zr', 'N', 'Cl']
MAX_NUM_ITER = 4
MAX_ATOMS_NUM = 20
MIN_MERGE_DISTANCE = 1.0    # Å
MIN_REGION_AREA = 10        # 噪声区最小像素面积阈值，不足则视为噪声

# ========= 新增：晶格参数（用户输入） =========
LATTICE_A = 3.452734        # 晶格常数 a (Å)
LATTICE_B = 3.451992        # 晶格常数 b (Å)
LATTICE_GAMMA = 90.0        # 夹角 gamma (度)
TOP_N_WINDOWS = 3           # 返回的最佳窗口数量

# ========= CLI 解析工具 =========

def parse_coordination_ratio(text: str):
    """将配位关系字符串解析为字典。
    支持的格式：
      - "Sn:1,S:2"
      - "Ga:2"
    返回: dict, 例如 {"Sn":1, "S":2}
    """
    if not text:
        return None
    parts = [p.strip() for p in text.split(',') if p.strip()]
    result = {}
    for part in parts:
        if ':' not in part:
            raise ValueError(f"无效的配位关系段: '{part}', 需为 'Elem:count' 格式")
        elem, cnt = part.split(':', 1)
        elem = elem.strip()
        try:
            cnt_val = int(cnt.strip())
        except ValueError:
            raise ValueError(f"元素 {elem} 的数量无效: '{cnt}' 需为整数")
        if cnt_val <= 0:
            raise ValueError(f"元素 {elem} 的数量需为正整数: {cnt_val}")
        result[elem] = cnt_val
    if not result:
        return None
    return result


def parse_elements(elements_text: Optional[str], coord: Optional[dict]):
    """解析元素列表。如果未显式提供，则从配位关系的键推断；否则回退默认。"""
    if elements_text:
        elems = [e.strip() for e in elements_text.split(',') if e.strip()]
        if not elems:
            raise ValueError("--elements 为空，请提供如 'Sn,S' 或省略该参数")
        return elems
    if coord:
        return list(coord.keys())
    return ELEMENTS_TYPE


# ------------------------------------------------------------
#  核心：原子去重 / 合并（支持周期性）
# ------------------------------------------------------------

def _pbc_mean(cart_coords, cell):
    """对跨边界坐标求均值：选第一点为参考，把其余点平移到最近镜像后再取均值。"""
    ref = cart_coords[0]
    shifted = []
    for p in cart_coords:
        d = p - ref
        # 最小镜像矢量
        for ax in range(3):
            L = cell[ax, ax]
            if L == 0:
                continue
            d_ax = d[ax]
            if d_ax > L / 2:
                d[ax] -= L
            elif d_ax < -L / 2:
                d[ax] += L
        shifted.append(ref + d)
    return np.mean(shifted, axis=0)


def merge_close_atoms(atoms: Atoms, min_dist: float = 1.0) -> Atoms:
    """合并距离 < *min_dist* Å 的原子（含跨 PBC）。

    规则：
      • 簇内元素一致 → 质心；
      • 元素不同     → 保留 Z 最大元素，坐标取该原子原位。
    """
    if len(atoms) == 0:
        return atoms.copy()

    # 1️⃣  构建近邻对（ASE neighbor_list 支持 PBC）
    idx_i, idx_j = neighbor_list("ij", atoms, cutoff=min_dist * 0.999)  # 乘 0.999 避免边界浮点误差
    pairs = list(zip(idx_i, idx_j))

    # 2️⃣  并查集聚类
    parent = list(range(len(atoms)))

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
    for idx in range(len(atoms)):
        clusters.setdefault(find(idx), []).append(idx)

    # 3. 生成新原子列表
    pos = atoms.get_positions()
    syms = np.array(atoms.get_chemical_symbols())
    Z = np.array([atomic_numbers[s] for s in syms])
    cell = atoms.get_cell().array  # 3×3

    new_pos, new_sym = [], []
    for idxs in clusters.values():
        if len(idxs) == 1:
            i = idxs[0]
            new_pos.append(pos[i])
            new_sym.append(syms[i])
            continue
        sub_syms = syms[idxs]
        cart_coords = pos[idxs]
        if np.all(sub_syms == sub_syms[0]):
            merged_coord = _pbc_mean(cart_coords, cell)
            new_pos.append(merged_coord)
            new_sym.append(sub_syms[0])
        else:
            heavy_idx = idxs[np.argmax(Z[idxs])]
            new_pos.append(pos[heavy_idx])
            new_sym.append(syms[heavy_idx])

    merged = Atoms(symbols=list(new_sym), positions=new_pos)
    merged.set_cell(atoms.get_cell())
    merged.set_pbc(atoms.get_pbc())
    return merged


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


# 2D高斯函数模型定义: 返回展开成一维的强度值数组
def gaussian_2d(xy, A, x0, y0, sx, sy, offset):
    x, y = xy
    return A * np.exp(-(((x - x0) ** 2) / (2 * sx ** 2) + ((y - y0) ** 2) / (2 * sy ** 2))) + offset


# 图像处理主函数
def process_image_with_lattice_params(image_path, lattice_params=None):
    """
    处理图像并返回原子结构，支持晶格参数输入
    lattice_params: (a, b, gamma) 晶格参数，如果为None则使用默认值
    """
    if lattice_params is None:
        lattice_params = (LATTICE_A, LATTICE_B, LATTICE_GAMMA)
    
    # 1. 读图 & 归一化
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将图像转换为浮点型并归一化到 [0,1]
    img = img.astype(np.float32)
    img_norm = (img - img.min())/(img.max()-img.min()+1e-8)

    # 2. 导入去噪后的图像
    img_denoised = img_norm  # 如果无可用模型，则跳过去噪（或选择其他去噪方法）

    # 3. 二值 & 连通域
    img_uint8 = (img_denoised*255).astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # binary 是 0/255 二值图
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # 可以根据噪声大小适当增大 kernel 大小，比如 (5,5)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = opened
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    flat_labels    = labels.flatten().astype(int)
    flat_intensity = img_denoised.flatten()

    # 计算每个标签的总强度 & 像素数
    sums   = np.bincount(flat_labels, weights=flat_intensity)
    counts = np.bincount(flat_labels)

    # 跳过背景 & 最小面积
    sums   = sums[1:]
    counts = counts[1:]
    areas  = stats[1:, cv2.CC_STAT_AREA]
    # 保留面积 ≥ MIN_REGION_AREA
    mask   = areas >= MIN_REGION_AREA
    sums   = sums[mask]
    counts = counts[mask]

    num_regions = len(sums)
    if num_regions == 0:
        print("No valid regions.")
        return None, None

    mean_intensity = sums/np.maximum(counts, 1)

    # 4. KMeans 聚类识别元素
    K = len(ELEMENTS_TYPE)
    brightness_values = mean_intensity.reshape(-1,1)
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0).fit(brightness_values)
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()

    brightness_order = np.argsort(centers)[::-1]
    elems_sorted_by_Z = sorted(ELEMENTS_TYPE, key=lambda s: atomic_numbers[s], reverse=True)
    cluster_to_elem = {int(cid): elems_sorted_by_Z[i] for i, cid in enumerate(brightness_order)}

    # 5. 拟合 & 提取像素中心
    atom_positions_pixels, atom_symbols = [], []
    # 构建"有效标签"列表：原 label 从1开始，取 mask True 的那些
    valid_labels = np.nonzero(mask)[0] + 1

    for label in valid_labels:
        # 取区域
        x0 = stats[label, cv2.CC_STAT_LEFT]
        y0 = stats[label, cv2.CC_STAT_TOP]
        w  = stats[label, cv2.CC_STAT_WIDTH]
        h  = stats[label, cv2.CC_STAT_HEIGHT]
        sub_img    = img_denoised[y0:y0+h, x0:x0+w]
        sub_labels = labels[y0:y0+h, x0:x0+w]
        ys, xs     = np.where(sub_labels==label)
        intensities= sub_img[ys, xs]

        # 质心
        sumI = intensities.sum()
        if sumI<=0:
            continue
        x0_init = (xs*intensities).sum()/sumI
        y0_init = (ys*intensities).sum()/sumI

        # 挑小斑点直接用质心
        if len(xs)<6:
            x_fit, y_fit = x0_init, y0_init
        else:
            # 拟合
            Imin = intensities.min(); Imax=intensities.max()
            init = (Imax-Imin, x0_init, y0_init, 1.5, 1.5, Imin)
            try:
                popt,_ = curve_fit(gaussian_2d, (xs, ys), intensities, p0=init, maxfev=2000)
                _, x_fit, y_fit, *_ = popt
            except:
                x_fit, y_fit = x0_init, y0_init

        atom_positions_pixels.append((x0 + x_fit, y0 + y_fit))
        # 元素类型
        idx = valid_labels.tolist().index(label)
        atom_symbols.append(cluster_to_elem[cluster_labels[idx]])

    # 6. 转 Å
    cell_c = 5.0
    atom_positions_ang = []
    for (xp, yp) in atom_positions_pixels:
        x_ang = xp*PIXEL_SIZE
        y_ang = yp*PIXEL_SIZE
        z_ang = 0.5 + np.random.uniform(-0.05,0.05) + cell_c/2
        atom_positions_ang.append((x_ang, y_ang, z_ang))

    atoms = Atoms(symbols=atom_symbols, positions=atom_positions_ang)
    h, w  = img.shape
    atoms.set_cell([[w*PIXEL_SIZE,0,0],[0,h*PIXEL_SIZE,0],[0,0,cell_c]])
    atoms.set_pbc((True,True,False))

    # 伪影过滤
    atoms = merge_close_atoms(atoms, min_dist=MIN_MERGE_DISTANCE)

    base = os.path.splitext(os.path.basename(image_path))[0]
    outp = base + "_reconstructed.cif"
    write(outp, atoms, format="cif", wrap=False)
    print(f"Processed {image_path}: {len(atom_symbols)} atoms → {outp}")
    return atoms, outp, lattice_params


def generate_candidate_windows(image_path: str,
                            elements: list,
                            coord: Optional[dict] = None,
                            lattice_params: Optional[tuple] = None,
                            top_n: int = 3,
                            out_prefix: str = "candidate_window"):
    """从图像生成候选窗口的 CIF（前 top_n 个），返回生成的文件路径列表。
    elements: 元素列表，例如 ['Zr','N','Cl']
    coord: 可选配位关系，如 {'Zr':1,'N':1,'Cl':1}，仅用于下游保存时的约束
    lattice_params: (a, b, gamma)；若为 None 则使用模块默认 LATTICE_A/B/GAMMA
    """
    global ELEMENTS_TYPE, TOP_N_WINDOWS
    if lattice_params is None:
        lattice_params = (LATTICE_A, LATTICE_B, LATTICE_GAMMA)
    # 设置元素与数量
    ELEMENTS_TYPE = list(elements)
    TOP_N_WINDOWS = int(top_n)

    # 图像处理得到初始 CIF
    atoms, output_path, lattice_params = process_image_with_lattice_params(image_path, lattice_params)

    # 读取结构并计算窗口
    atoms_pm, pos_xy, cell_abc = load_structure(output_path)
    a, b, c = cell_abc
    total_area = a * b

    print(f"使用晶格参数 {lattice_params} 寻找最佳窗口(top{top_n})...")
    analyze_lattice_compatibility(atoms_pm, pos_xy, lattice_params)
    best_windows = find_multiple_best_windows(pos_xy, lattice_params, total_area, top_n=top_n)

    if not best_windows:
        # 回退传统方法，仅保存一个窗口
        (bx, by), (p,u,d) = find_best_window(pos_xy, (a,b), total_area)
        print(f"传统方法回退：最佳窗口左下角({bx:.1f}, {by:.1f}) 评分(P,U,D)=({p:.3f},{u:.3f},{d:.3f})")
        extract_and_write_with_lattice_constraint(
            atoms_pm, pos_xy, (bx, by), lattice_params, f"{out_prefix}_01.cif", coord or {}
        )
        return [f"{out_prefix}_01.cif"]

    # 保存所有候选窗口
    save_all_candidate_windows(atoms_pm, pos_xy, best_windows, out_prefix)
    # 报告
    save_window_analysis_report(best_windows, f"{out_prefix}_analysis.txt")

    # 返回前 top_n 个路径
    paths = []
    for i in range(min(top_n, len(best_windows))):
        paths.append(f"{out_prefix}_{i+1:02d}.cif")
    return paths


if __name__ == '__main__':
    # 使用 argparse 统一输入
    parser = argparse.ArgumentParser(description='Atom AI FFT-based structure reconstruction')
    parser.add_argument('--image-path', type=str, required=True, help='输入的图像路径（PNG/TIF 等）')
    parser.add_argument('--lattice-a', type=float, default=LATTICE_A, help='晶格常数 a (Å)')
    parser.add_argument('--lattice-b', type=float, default=LATTICE_B, help='晶格常数 b (Å)')
    parser.add_argument('--lattice-gamma', type=float, default=LATTICE_GAMMA, help='夹角 gamma (度)')
    parser.add_argument('--coord', type=str, default='Zr:1, N:1, Cl:1', help="配位关系，如 'Sn:1,S:2' 或单元素 'Ga:2'")
    parser.add_argument('--elements', type=str, default='Zr, N, Cl', help="元素列表，如 'Sn,S'；缺省则从 --coord 推断或使用默认")
    parser.add_argument('--pixel-size', type=float, default=PIXEL_SIZE, help='像素尺寸 (Å/pixel)')
    parser.add_argument('--min-merge-distance', type=float, default=MIN_MERGE_DISTANCE, help='原子合并最小距离 (Å)')
    parser.add_argument('--min-region-area', type=int, default=MIN_REGION_AREA, help='连通域最小面积阈值 (pixels)')
    parser.add_argument('--max-num-iter', type=int, default=MAX_NUM_ITER, help='缩减晶胞的最大迭代次数')
    parser.add_argument('--max-atoms-num', type=int, default=MAX_ATOMS_NUM, help='停止缩减的原子数阈值')
    parser.add_argument('--top-n-windows', type=int, default=TOP_N_WINDOWS, help='多窗口候选数量')
    args = parser.parse_args()

    # 覆盖全局参数（仅在本进程内）
    PIXEL_SIZE = args.pixel_size
    MIN_MERGE_DISTANCE = args.min_merge_distance
    MIN_REGION_AREA = args.min_region_area
    MAX_NUM_ITER = args.max_num_iter
    MAX_ATOMS_NUM = args.max_atoms_num
    TOP_N_WINDOWS = args.top_n_windows

    # 解析配位与元素
    try:
        coordination_ratio = parse_coordination_ratio(args.coord) or {'Sn': 1, 'S': 2}
    except ValueError as e:
        print(f"配位关系解析失败: {e}")
        sys.exit(1)

    try:
        ELEMENTS_TYPE = parse_elements(args.elements, coordination_ratio)
    except ValueError as e:
        print(f"元素列表解析失败: {e}")
        sys.exit(1)

    lattice_params = (args.lattice_a, args.lattice_b, args.lattice_gamma)

    # 执行图像处理与结构重建
    image_path = args.image_path
    atoms, output_path, lattice_params = process_image_with_lattice_params(image_path, lattice_params)

    # 确定原始最小晶胞(缩减晶胞大小)流程：循环弛豫缩减
    cur = Structure.from_file(output_path)
    print(f"起始原子数: {len(cur)}")

    for step in range(1, MAX_NUM_ITER):
        if len(cur) <= MAX_ATOMS_NUM:
            print(f"≤{MAX_ATOMS_NUM}atoms，停止。")
            break
        nxt = shrink_once(cur)
        if nxt is None or len(nxt) >= len(cur):
            print("已无法进一步缩减，回退到最佳周期性区域提取。")

            # 将 pymatgen Structure 转为 ASE Atoms
            # 1. 读结构
            atoms, pos_xy, cell_abc = load_structure(output_path)
            a, b, c = cell_abc
            total_area = a * b

            # 2. 使用新的多窗口方法寻找最佳区域
            print(f"使用晶格参数 {lattice_params} 寻找最佳窗口...")
            
            # 首先分析晶格兼容性
            analyze_lattice_compatibility(atoms, pos_xy, lattice_params)
            
            best_windows = find_multiple_best_windows(pos_xy, lattice_params, total_area, top_n=TOP_N_WINDOWS)
            
            if not best_windows:
                print("未找到合适的窗口，使用传统方法...")
                # 回退到传统方法
                (bx, by), (p,u,d) = find_best_window(pos_xy, (a,b), total_area)
                print(f"最佳窗口左下角：({bx:.1f}, {by:.1f}) 评分(P,U,D)=({p:.3f},{u:.3f},{d:.3f})")
                
                # 应用配位关系约束
                extract_and_write_with_lattice_constraint(
                    atoms, pos_xy, (bx, by), lattice_params, "output_final.cif", coordination_ratio
                )
            else:
                print(f"找到 {len(best_windows)} 个候选窗口:")
                for i, ((bx, by), scores, combined_score) in enumerate(best_windows):
                    p, u, d, lattice_score = scores
                    print(f"  窗口 {i+1}: 左下角({bx:.1f}, {by:.1f}) 评分(P,U,D,Lattice,Combined)=({p:.3f},{u:.3f},{d:.3f},{lattice_score:.3f},{combined_score:.3f})")
                
                # 保存所有候选窗口的CIF文件
                save_all_candidate_windows(atoms, pos_xy, best_windows, "candidate_window")
                
                # 保存窗口分析报告
                save_window_analysis_report(best_windows, "window_analysis_report.txt")
                
                # 选择评分最高的窗口作为最终结果
                best_window = best_windows[0]
                (bx, by), scores, combined_score = best_window
                p, u, d, lattice_score = scores
                print(f"\n选择评分最高的窗口作为最终结果：")
                print(f"  窗口位置: 左下角({bx:.1f}, {by:.1f})")
                print(f"  评分(P,U,D,Lattice,Combined)=({p:.3f},{u:.3f},{d:.3f},{lattice_score:.3f},{combined_score:.3f})")
                
                # 保存最终结果，应用配位关系约束
                extract_and_write_with_lattice_constraint(
                    atoms, pos_xy, (bx, by), lattice_params, "output_final.cif", coordination_ratio
                )
            
            sys.exit(0)

        print(f"Step {step}: {len(cur)} → {len(nxt)} atoms")
        cur = nxt

    cur.to(filename="output_final_fft.cif")
    print(f"最终结构 output_final_fft.cif (atoms = {len(cur)})")
