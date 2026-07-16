#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_best_periodic_region.py

从一个 2D CIF 结构中自动提取一个 200×200 Å 的最佳周期性区域，
评分准则（优先级从高到低）：
 1. 周期性（FFT 谱中离散峰能量占比）
 2. 分布均匀性（最近邻距离变异系数）
 3. 原子密度与全局平均密度偏差

输出 shrink.cif，包括该区域内所有原子，晶胞尺寸为 200×200×原始 c。
"""

import numpy as np
from ase import io
from ase.data import atomic_numbers
from ase.io import write
from ase import Atoms
from ase.neighborlist import neighbor_list, natural_cutoffs
from scipy.spatial import cKDTree
import os
from collections import Counter


# -------- 用户配置 --------
INPUT_CIF   = "./best_reconstructed_image_reconstructed.cif"  # 输入CIF 文件路径
WIN_SIZE    = 15.0    # 窗口边长 (Å)
COARSE_STEP = 5.0     # 粗扫描步长 (Å)
FINE_STEP   = 1.0      # 细扫描步长 (Å)
GRID_RES    = 0.2      # 频域分析网格分辨率 (Å)
# --------------------------

def load_structure(path):
    """加载 ASE Atoms，对应2D结构，只取xy坐标。"""
    atoms = io.read(path)
    pos_xy = atoms.get_positions()[:, :2]
    a, b, c = atoms.cell.lengths()
    return atoms, pos_xy, (a, b, c)

def periodic_score(subset_xy, win_size, grid_res):
    """
    周期性评分：将原子投影到网格后做 FFT，
    取除 DC 外前若干峰能量 / 总谱能量。
    """
    # 构建网格
    nx = int(np.ceil(win_size/grid_res))
    ny = int(np.ceil(win_size/grid_res))
    grid = np.zeros((ny, nx), float)
    # 原子到网格索引
    idx = ((subset_xy) / grid_res).astype(int)
    idx[:,0] = np.clip(idx[:,0], 0, nx-1)
    idx[:,1] = np.clip(idx[:,1], 0, ny-1)
    grid[idx[:,1], idx[:,0]] = 1.0
    # FFT 谱
    F = np.fft.fft2(grid)
    P = np.abs(F)**2
    P[0,0] = 0.0
    flat = P.ravel()
    # 选前 10 个峰
    if flat.size <= 10:
        top = flat
    else:
        top = flat[np.argpartition(flat, -10)[-10:]]
    return top.sum() / (flat.sum() + 1e-12)

def uniform_score(subset_xy):
    """
    均匀性评分：最近邻距离的变异系数 CV = sigma/mu，
    返回 1 - CV，裁剪到 [0,1]。
    """
    tree = cKDTree(subset_xy)
    d, _ = tree.query(subset_xy, k=2)
    nn = d[:,1]
    mu, sigma = nn.mean(), nn.std()
    if mu < 1e-6:
        return 0.0
    cv = sigma/mu
    return max(0.0, 1 - cv)

def density_score(N, area, avg_density):
    """
    密度评分：1 - |ρ_region - ρ_avg|/ρ_avg，裁剪到 [0,1]。
    """
    rho = N/area
    diff = abs(rho - avg_density)/max(avg_density,1e-12)
    return max(0.0, 1 - diff)

def evaluate_region(x0, y0, pos_xy, total_area, avg_density):
    """计算窗口(x0,y0)->(x0+WIN_SIZE,y0+WIN_SIZE)的三项评分。"""
    mask = ((pos_xy[:,0]>=x0)&(pos_xy[:,0]<x0+WIN_SIZE) &
            (pos_xy[:,1]>=y0)&(pos_xy[:,1]<y0+WIN_SIZE))
    subset = pos_xy[mask]
    N = len(subset)
    if N == 0:
        return None  # 空区跳过
    # 周期性
    p_score = periodic_score(subset - [x0,y0], WIN_SIZE, GRID_RES)
    # 均匀性
    u_score = uniform_score(subset)
    # 密度
    d_score = density_score(N, WIN_SIZE*WIN_SIZE, avg_density)
    return (p_score, u_score, d_score)

def find_best_window(pos_xy, cell_ab, total_area):
    """粗细两阶段滑窗，返回最佳窗口左下角坐标及评分。"""
    avg_density = len(pos_xy)/total_area
    x_min, y_min = pos_xy.min(axis=0)
    x_max_start = pos_xy[:,0].max() - WIN_SIZE
    y_max_start = pos_xy[:,1].max() - WIN_SIZE

    best = ((-1,-1), (-1.0,-1.0,-1.0))
    # 粗扫描
    xs = np.arange(x_min, x_max_start+1e-6, COARSE_STEP)
    ys = np.arange(y_min, y_max_start+1e-6, COARSE_STEP)
    for x in xs:
        for y in ys:
            scores = evaluate_region(x, y, pos_xy, total_area, avg_density)
            if scores is None: continue
            if scores > best[1]:
                best = ((x,y), scores)

    # 细扫描
    (cx, cy), base_scores = best
    xs = np.arange(max(x_min, cx-COARSE_STEP),
                   min(x_max_start, cx+COARSE_STEP)+1e-6, FINE_STEP)
    ys = np.arange(max(y_min, cy-COARSE_STEP),
                   min(y_max_start, cy+COARSE_STEP)+1e-6, FINE_STEP)
    for x in xs:
        for y in ys:
            scores = evaluate_region(x, y, pos_xy, total_area, avg_density)
            if scores is None: continue
            if scores > best[1]:
                best = ((x,y), scores)

    return best

def find_multiple_best_windows(pos_xy, lattice_params, total_area, top_n=3):
    """
    改进版：返回多个最佳窗口，考虑晶格匹配度
    lattice_params: (a, b, gamma) 从FFT分析得到的晶格参数
    """
    a, b, gamma = lattice_params
    avg_density = len(pos_xy) / total_area
    
    # 计算扫描范围
    x_min, y_min = pos_xy.min(axis=0)
    x_max_start = pos_xy[:, 0].max() - WIN_SIZE
    y_max_start = pos_xy[:, 1].max() - WIN_SIZE
    
    # 初始化候选窗口列表
    candidate_windows = []
    
    # 粗扫描
    xs = np.arange(x_min, x_max_start + 1e-6, COARSE_STEP)
    ys = np.arange(y_min, y_max_start + 1e-6, COARSE_STEP)
    
    for x in xs:
        for y in ys:
            scores = evaluate_region(x, y, pos_xy, total_area, avg_density)
            if scores is None:
                continue
            
            # 计算晶格匹配度
            lattice_score = evaluate_lattice_match(x, y, pos_xy, a, b, gamma)
            
            # 综合评分：周期性 + 均匀性 + 密度 + 晶格匹配
            p_score, u_score, d_score = scores
            combined_score = (p_score * 0.4 + u_score * 0.2 + d_score * 0.2 + lattice_score * 0.2)
            
            candidate_windows.append(((x, y), (p_score, u_score, d_score, lattice_score), combined_score))
    
    # 按综合评分排序
    candidate_windows.sort(key=lambda x: x[2], reverse=True)
    
    # 返回前top_n个窗口
    return candidate_windows[:top_n]

def evaluate_lattice_match(x0, y0, pos_xy, a, b, gamma):
    """
    评估窗口内原子排列与给定晶格参数的匹配度
    """
    # 提取窗口内原子
    mask = ((pos_xy[:, 0] >= x0) & (pos_xy[:, 0] < x0 + WIN_SIZE) &
            (pos_xy[:, 1] >= y0) & (pos_xy[:, 1] < y0 + WIN_SIZE))
    subset = pos_xy[mask]
    
    if len(subset) < 4:  # 至少需要4个原子来评估周期性
        return 0.0
    
    # 将坐标平移到窗口原点
    subset_shifted = subset - [x0, y0]
    
    # 计算实际晶格参数
    actual_lattice = estimate_lattice_from_atoms(subset_shifted)
    if actual_lattice is None:
        return 0.0
    
    actual_a, actual_b, actual_gamma = actual_lattice
    
    # 计算参数匹配度
    a_match = 1.0 - abs(actual_a - a) / max(a, actual_a)
    b_match = 1.0 - abs(actual_b - b) / max(b, actual_b)
    gamma_match = 1.0 - abs(actual_gamma - gamma) / 180.0  # 角度差异
    
    # 综合匹配度
    match_score = (a_match + b_match + gamma_match) / 3.0
    return max(0.0, match_score)

def estimate_lattice_from_atoms(subset_xy):
    """
    从原子坐标估计晶格参数
    """
    if len(subset_xy) < 4:
        return None
    
    # 使用最近邻距离估计晶格参数
    from scipy.spatial import cKDTree
    tree = cKDTree(subset_xy)
    distances, indices = tree.query(subset_xy, k=2)
    nn_distances = distances[:, 1]  # 最近邻距离
    
    # 使用距离直方图找到主要周期
    hist, bins = np.histogram(nn_distances, bins=20)
    peak_indices = np.argsort(hist)[-3:]  # 前3个峰值
    
    if len(peak_indices) < 2:
        return None
    
    # 估计a和b
    a_est = bins[peak_indices[0]]
    b_est = bins[peak_indices[1]]
    
    # 估计gamma（简化版本）
    gamma_est = 90.0  # 默认直角，可以后续改进
    
    return (a_est, b_est, gamma_est)

def compare_substructure_similarity(window1, window2, pos_xy):
    """
    比较两个窗口的子结构相似性
    """
    (x1, y1), (x2, y2) = window1[0], window2[0]
    
    # 提取两个窗口的原子
    mask1 = ((pos_xy[:, 0] >= x1) & (pos_xy[:, 0] < x1 + WIN_SIZE) &
             (pos_xy[:, 1] >= y1) & (pos_xy[:, 1] < y1 + WIN_SIZE))
    mask2 = ((pos_xy[:, 0] >= x2) & (pos_xy[:, 0] < x2 + WIN_SIZE) &
             (pos_xy[:, 1] >= y2) & (pos_xy[:, 1] < y2 + WIN_SIZE))
    
    subset1 = pos_xy[mask1] - [x1, y1]  # 平移到原点
    subset2 = pos_xy[mask2] - [x2, y2]
    
    if len(subset1) < 3 or len(subset2) < 3:
        return 0.0
    
    # 使用结构因子比较相似性
    similarity = calculate_structure_factor_similarity(subset1, subset2)
    return similarity

def calculate_structure_factor_similarity(subset1, subset2):
    """
    通过结构因子计算两个子结构的相似性
    """
    # 将原子投影到网格
    grid_size = 0.5  # Å
    nx = int(WIN_SIZE / grid_size)
    ny = int(WIN_SIZE / grid_size)
    
    def atoms_to_grid(subset):
        grid = np.zeros((ny, nx))
        for atom in subset:
            i = int(atom[0] / grid_size)
            j = int(atom[1] / grid_size)
            if 0 <= i < nx and 0 <= j < ny:
                grid[j, i] = 1.0
        return grid
    
    grid1 = atoms_to_grid(subset1)
    grid2 = atoms_to_grid(subset2)
    
    # 计算互相关
    from scipy.signal import correlate2d
    correlation = correlate2d(grid1, grid2, mode='full')
    max_corr = np.max(correlation)
    
    # 归一化
    norm1 = np.sum(grid1**2)
    norm2 = np.sum(grid2**2)
    similarity = max_corr / np.sqrt(norm1 * norm2 + 1e-12)
    
    return similarity

def select_best_substructure_from_windows(windows, pos_xy, lattice_params):
    """
    从多个候选窗口中选择最佳子结构
    """
    if len(windows) < 2:
        return windows[0] if windows else None
    
    # 计算所有窗口对之间的相似性
    similarities = []
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            sim = compare_substructure_similarity(windows[i], windows[j], pos_xy)
            similarities.append((i, j, sim))
    
    # 找到最相似的窗口对
    if similarities:
        most_similar = max(similarities, key=lambda x: x[2])
        i, j, sim = most_similar
        
        # 选择评分更高的那个
        if windows[i][2] > windows[j][2]:  # combined_score
            return windows[i]
        else:
            return windows[j]
    
    # 如果没有相似性信息，返回评分最高的
    return max(windows, key=lambda x: x[2])

def extract_substructure_from_window(window, pos_xy):
    """
    从窗口提取子结构
    """
    (x, y), scores, combined_score = window
    mask = ((pos_xy[:, 0] >= x) & (pos_xy[:, 0] < x + WIN_SIZE) &
            (pos_xy[:, 1] >= y) & (pos_xy[:, 1] < y + WIN_SIZE))
    subset = pos_xy[mask] - [x, y]  # 平移到原点
    return subset

def align_substructure_to_lattice(subset, lattice_params):
    """
    将子结构对齐到给定的晶格参数
    """
    a, b, gamma = lattice_params
    
    # 计算旋转角度
    gamma_rad = np.radians(gamma)
    
    # 创建旋转矩阵
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)
    rotation_matrix = np.array([[cos_gamma, -sin_gamma], [sin_gamma, cos_gamma]])
    
    # 应用旋转
    aligned_subset = subset @ rotation_matrix.T
    
    return aligned_subset

def find_optimal_unit_cell(windows, pos_xy, lattice_params):
    """
    寻找最优单胞：结合多个候选窗口和晶格匹配
    """
    if not windows:
        return None
    
    # 1. 计算窗口间的相似性
    similarities = []
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            sim = compare_substructure_similarity(windows[i], windows[j], pos_xy)
            similarities.append((i, j, sim))
    
    # 2. 选择最相似的窗口对
    if similarities:
        most_similar = max(similarities, key=lambda x: x[2])
        i, j, sim = most_similar
        
        # 选择评分更高的窗口
        if windows[i][2] > windows[j][2]:
            best_window = windows[i]
        else:
            best_window = windows[j]
    else:
        # 如果没有相似性信息，选择评分最高的
        best_window = max(windows, key=lambda x: x[2])
    
    # 3. 提取子结构并对齐到晶格
    subset = extract_substructure_from_window(best_window, pos_xy)
    aligned_subset = align_substructure_to_lattice(subset, lattice_params)
    
    return best_window, aligned_subset

def extract_and_write(atoms, pos_xy, best_xy, output_cif):
    """根据 best_xy 提取子区域原子，重设晶胞并写出 shrink.cif"""
    x0, y0 = best_xy
    mask = ((pos_xy[:,0]>=x0)&(pos_xy[:,0]<x0+WIN_SIZE) &
            (pos_xy[:,1]>=y0)&(pos_xy[:,1]<y0+WIN_SIZE))
    sub = atoms[mask]             # ASE Atoms 支持布尔索引
    # 平移坐标到 (0,0)
    new_pos = sub.get_positions()
    new_pos[:,0] -= x0
    new_pos[:,1] -= y0
    sub.set_positions(new_pos)
    # 新晶胞
    a_new = [WIN_SIZE, 0, 0]
    b_new = [0, WIN_SIZE, 0]
    c_new = atoms.cell[2]  # 保留原 c 方向
    sub.set_cell([a_new, b_new, c_new])
    sub.set_pbc((False, False, False))
    sub = merge_close_atoms(sub)
    write(output_cif, sub, format="cif")
    print(f"最佳区域导出至 {output_cif}，包含 {len(sub)} 个原子。")


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

def _build_adjacency(positions_xy: np.ndarray, cutoff: float) -> list:
    """Build adjacency list for 2D positions using distance cutoff (Å)."""
    n = positions_xy.shape[0]
    if n == 0:
        return []
    tree = cKDTree(positions_xy)
    # query ball
    neighbors = tree.query_ball_point(positions_xy, r=cutoff)
    # ensure no self-dup complications; keep symmetric
    adj = []
    for i in range(n):
        # remove self
        adj.append([j for j in neighbors[i] if j != i])
    return adj


def _connected_components(adj: list) -> list:
    """Return list of lists of indices, each a connected component."""
    n = len(adj)
    visited = [False] * n
    comps = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _try_pick_by_ratio_greedy(component_idx: list,
                              positions_xy: np.ndarray,
                              symbols: list,
                              ratio: dict,
                              cutoff: float,
                              max_atoms: int = 20) -> list:
    """Greedy selection of a small connected subset within a component that satisfies the stoichiometric ratio.
    Returns indices (relative to positions_xy) or empty list if not found.
    Strategy: pick a seed of the most-demanding element, then grow by nearest neighbors of required types, keeping connectivity.
    """
    target_total = int(sum(ratio.values()))
    if target_total > max_atoms:
        return []

    # group indices in component by element
    elem_to_idxs = {}
    for idx in component_idx:
        s = symbols[idx]
        elem_to_idxs.setdefault(s, []).append(idx)

    # quick feasibility check
    for e, cnt in ratio.items():
        if len(elem_to_idxs.get(e, [])) < cnt:
            return []

    # choose seeds from the rarest required element to reduce branching
    req_sorted = sorted(ratio.items(), key=lambda kv: kv[1], reverse=True)
    # precompute distances within component
    comp_pos = positions_xy[component_idx]
    tree = cKDTree(comp_pos)

    def is_connected(subset_idxs: list) -> bool:
        if not subset_idxs:
            return False
        # build adjacency restricted to subset using cutoff
        pos = positions_xy[subset_idxs]
        t = cKDTree(pos)
        neigh = t.query_ball_point(pos, r=cutoff)
        # simple bfs
        seen = set([0])
        stack = [0]
        while stack:
            u = stack.pop()
            for v in neigh[u]:
                if v == u:
                    continue
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == len(subset_idxs)

    # heuristic: try each candidate seed of the first required element
    first_elem = req_sorted[0][0]
    seeds = elem_to_idxs[first_elem]
    # limit seeds to avoid explosion
    max_seeds = min(50, len(seeds))
    seeds = seeds[:max_seeds]

    for seed in seeds:
        selected = [seed]
        need = {e: ratio[e] for e in ratio}
        need[first_elem] -= 1

        # iterative grow: breadth from current selection, prefer nearest neighbors of needed elements
        while sum(need.values()) > 0 and len(selected) < target_total:
            # gather frontier neighbors within cutoff from current selected set
            frontier = set()
            for sidx in selected:
                # neighbors within cutoff from component indices
                local = tree.query_ball_point(positions_xy[sidx], r=cutoff)
                # map local indices back to global indices
                for li in local:
                    gi = component_idx[li]
                    if gi in selected:
                        continue
                    frontier.add(gi)
            if not frontier:
                break
            # choose best candidate among frontier that satisfies still-needed elements
            best_candidate = None
            best_dist = 1e9
            for gi in frontier:
                se = symbols[gi]
                if need.get(se, 0) <= 0:
                    continue
                # distance to current cluster centroid
                centroid = positions_xy[selected].mean(axis=0)
                d = np.linalg.norm(positions_xy[gi] - centroid)
                if d < best_dist:
                    best_dist = d
                    best_candidate = gi
            if best_candidate is None:
                # allow adding any element if still under target, but this likely breaks ratio; stop
                break
            selected.append(best_candidate)
            need[symbols[best_candidate]] -= 1

        # verify
        if sum(need.values()) == 0 and len(selected) == target_total and is_connected(selected):
            return selected

    return []


def select_connected_cluster_by_stoichiometry(positions_xy: np.ndarray,
                                              symbols: list,
                                              ratio: dict,
                                              cutoff: float,
                                              max_atoms: int = 20) -> list:
    """Find a connected cluster that satisfies the stoichiometric ratio within distance cutoff.
    Returns list of global indices; empty list if not found.
    """
    if positions_xy.shape[0] == 0:
        return []
    adj = _build_adjacency(positions_xy, cutoff)
    comps = _connected_components(adj)
    # try components by descending size but bounded
    comps_sorted = sorted(comps, key=len, reverse=True)
    for comp in comps_sorted:
        # prune big components by taking a working copy of indices
        sel = _try_pick_by_ratio_greedy(comp, positions_xy, symbols, ratio, cutoff, max_atoms=max_atoms)
        if sel:
            return sel
    return []

def extract_and_write_with_lattice_constraint(atoms, pos_xy, best_xy, lattice_params, output_cif, coordination_ratio=None, coord_cutoff: float = None):
    """
    根据晶格参数约束提取最佳子区域（保持几何不变，仅平移）。
    可选：coordination_ratio 例如 {'Sn':1,'S':2}，并通过邻接阈值 coord_cutoff 选择连通子簇。
    """
    a, b, gamma = lattice_params
    x0, y0 = best_xy

    # 提取窗口内原子（Cartesian 保持原样）
    mask = ((pos_xy[:, 0] >= x0) & (pos_xy[:, 0] < x0 + WIN_SIZE) &
            (pos_xy[:, 1] >= y0) & (pos_xy[:, 1] < y0 + WIN_SIZE))
    sub_atoms = atoms[mask]
    if len(sub_atoms) == 0:
        print("警告：窗口内没有原子")
        return

    # 平移坐标到窗口原点，保持几何不变
    sub_pos = sub_atoms.get_positions().copy()
    sub_pos[:, 0] -= x0
    sub_pos[:, 1] -= y0
    sub_syms = sub_atoms.get_chemical_symbols()

    keep_indices = list(range(len(sub_atoms)))

    if coordination_ratio is not None:
        # 估计默认 cutoff（Å）：窗口内最近邻中位数 * 1.4
        if coord_cutoff is None and len(sub_pos) >= 2:
            tree = cKDTree(sub_pos[:, :2])
            dists, _ = tree.query(sub_pos[:, :2], k=min(3, len(sub_pos)))
            # dists[:,1] is NN, guard zeros
            nn = dists[:, 1]
            nn = nn[nn > 1e-6]
            if len(nn) > 0:
                coord_cutoff = float(np.median(nn) * 1.4)
            else:
                coord_cutoff = float(min(a, b) * 0.6)
        elif coord_cutoff is None:
            coord_cutoff = float(min(a, b) * 0.6)

        sel = select_connected_cluster_by_stoichiometry(sub_pos[:, :2], sub_syms, coordination_ratio, coord_cutoff, max_atoms=20)
        if sel:
            keep_indices = sel
        else:
            print("[Warn] 未找到满足配位且连通的子簇，保留整个窗口原子。可调小/大 coord_cutoff 重试。")

    # 生成输出原子：仅保留所选，并保持 Cartesian 不变（只做平移）
    out_pos = sub_pos[keep_indices]
    out_syms = [sub_syms[i] for i in keep_indices]

    out = Atoms(symbols=out_syms, positions=out_pos)

    # 设置输出晶胞：使用确定好的 lattice_params，并将原子包裹进单胞（仅做按晶格矢量的平移，不改变几何）
    gamma_rad = np.radians(gamma)
    cell_latt = np.array([
        [a, 0.0, 0.0],
        [b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0.0],
        [0.0, 0.0, atoms.cell[2][2] if hasattr(atoms.cell[2], '__getitem__') else float(atoms.cell[2])]
    ], dtype=float)
    out.set_cell(cell_latt)
    out.set_pbc((True, True, False))

    # 将原子按 PBC 包裹到 0..1 分数坐标内（等同于平移整格矢量，不改变键长/角）
    sp = out.get_scaled_positions()
    sp[:, 0] = np.mod(sp[:, 0], 1.0)
    sp[:, 1] = np.mod(sp[:, 1], 1.0)
    out.set_scaled_positions(sp)

    write(output_cif, out, format="cif")
    print(f"[OK] 已导出窗口内几何保持的子结构至 {output_cif}，原子数={len(out)}，配位约束={coordination_ratio if coordination_ratio else '无'}")

def apply_coordination_constraint(positions, symbols, coordination_ratio):
    """
    根据配位关系约束筛选原子
    """
    # 按元素类型分组
    element_groups = {}
    for i, symbol in enumerate(symbols):
        if symbol not in element_groups:
            element_groups[symbol] = []
        element_groups[symbol].append(i)
    
    print(f"窗口内元素分布: {dict(Counter(symbols))}")
    
    # 检查是否满足配位关系要求
    for element, required_count in coordination_ratio.items():
        if element not in element_groups:
            print(f"警告：窗口内缺少元素 {element}")
            return positions, symbols
        if len(element_groups[element]) < required_count:
            print(f"警告：元素 {element} 数量不足 (需要{required_count}, 实际{len(element_groups[element])})")
            return positions, symbols
    
    # 根据配位关系选择原子
    selected_indices = []
    for element, required_count in coordination_ratio.items():
        if element in element_groups:
            # 选择指定数量的原子
            available_count = len(element_groups[element])
            select_count = min(required_count, available_count)
            
            # 选择前几个原子（可以根据需要改进选择策略）
            selected_indices.extend(element_groups[element][:select_count])
    
    # 限制在目标大小内
    selected_indices = selected_indices[:sum(coordination_ratio.values())]
    
    selected_pos = positions[selected_indices]
    selected_symbols = [symbols[i] for i in selected_indices]
    
    print(f"配位关系约束选择: {Counter(selected_symbols)}")
    
    return selected_pos, selected_symbols

def constrain_atoms_to_lattice(atoms, pos_xy, lattice_params):
    """
    根据晶格参数约束原子位置
    """
    a, b, gamma = lattice_params
    gamma_rad = np.radians(gamma)
    
    # 计算晶格矢量
    a_vec = np.array([a, 0])
    b_vec = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad)])
    
    # 将原子坐标转换为分数坐标
    # 构建变换矩阵：从笛卡尔坐标到分数坐标
    cell_matrix = np.array([a_vec, b_vec])
    inv_cell_matrix = np.linalg.inv(cell_matrix)
    
    # 转换到分数坐标
    frac_coords = pos_xy @ inv_cell_matrix.T
    
    # 筛选在[0,1)范围内的原子（单胞内）
    valid_mask = ((frac_coords[:, 0] >= 0) & (frac_coords[:, 0] < 1) &
                  (frac_coords[:, 1] >= 0) & (frac_coords[:, 1] < 1))
    
    if not np.any(valid_mask):
        print("警告：没有原子在单胞范围内")
        return Atoms()
    
    # 获取有效原子
    valid_atoms = atoms[valid_mask]
    valid_frac_coords = frac_coords[valid_mask]
    
    # 将分数坐标转换回笛卡尔坐标（在单胞内）
    valid_cart_coords = valid_frac_coords @ cell_matrix
    
    # 更新原子位置
    new_positions = valid_atoms.get_positions().copy()
    new_positions[:, :2] = valid_cart_coords
    valid_atoms.set_positions(new_positions)
    
    print(f"晶格约束：从 {len(atoms)} 个原子筛选出 {len(valid_atoms)} 个原子")
    
    return valid_atoms

def find_optimal_unit_cell_with_lattice_constraint(windows, pos_xy, lattice_params):
    """
    寻找最优单胞：结合多个候选窗口和晶格匹配，并进行晶格约束
    """
    if not windows:
        return None
    
    # 1. 计算窗口间的相似性
    similarities = []
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            sim = compare_substructure_similarity(windows[i], windows[j], pos_xy)
            similarities.append((i, j, sim))
    
    # 2. 选择最相似的窗口对
    if similarities:
        most_similar = max(similarities, key=lambda x: x[2])
        i, j, sim = most_similar
        
        # 选择评分更高的窗口
        if windows[i][2] > windows[j][2]:
            best_window = windows[i]
        else:
            best_window = windows[j]
    else:
        # 如果没有相似性信息，选择评分最高的
        best_window = max(windows, key=lambda x: x[2])
    
    # 3. 提取子结构并对齐到晶格
    subset = extract_substructure_from_window(best_window, pos_xy)
    aligned_subset = align_substructure_to_lattice(subset, lattice_params)
    
    return best_window, aligned_subset

def analyze_lattice_compatibility(atoms, pos_xy, lattice_params):
    """
    分析原子结构与晶格参数的兼容性
    """
    a, b, gamma = lattice_params
    gamma_rad = np.radians(gamma)
    
    # 计算晶格矢量
    a_vec = np.array([a, 0])
    b_vec = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad)])
    
    # 分析原子间距
    from scipy.spatial import cKDTree
    tree = cKDTree(pos_xy)
    distances, indices = tree.query(pos_xy, k=2)
    nn_distances = distances[:, 1]
    
    # 计算与晶格参数的匹配度
    a_match = 1.0 - abs(np.median(nn_distances) - a) / max(a, np.median(nn_distances))
    b_match = 1.0 - abs(np.median(nn_distances) - b) / max(b, np.median(nn_distances))
    
    print(f"晶格兼容性分析：")
    print(f"  平均最近邻距离: {np.median(nn_distances):.3f} Å")
    print(f"  目标晶格参数: a={a:.3f} Å, b={b:.3f} Å, gamma={gamma:.1f}°")
    print(f"  匹配度: a={a_match:.3f}, b={b_match:.3f}")
    
    return a_match, b_match

def save_all_candidate_windows(atoms, pos_xy, best_windows, output_prefix="candidate_window"):
    """
    保存所有候选窗口的CIF文件 - 保存大窗口视野，不使用lattice_params约束
    """
    if not best_windows:
        print("没有候选窗口可保存")
        return
    
    print(f"保存 {len(best_windows)} 个候选窗口的CIF文件...")
    print(f"每个窗口大小: {WIN_SIZE}×{WIN_SIZE} Å")
    
    for i, window in enumerate(best_windows):
        (bx, by), scores, combined_score = window
        p, u, d, lattice_score = scores
        
        # 创建输出文件名
        output_cif = f"{output_prefix}_{i+1:02d}.cif"
        
        # 提取窗口内原子
        mask = ((pos_xy[:, 0] >= bx) & (pos_xy[:, 0] < bx + WIN_SIZE) &
                (pos_xy[:, 1] >= by) & (pos_xy[:, 1] < by + WIN_SIZE))
        sub_atoms = atoms[mask]
        
        if len(sub_atoms) == 0:
            print(f"  窗口 {i+1}: 没有原子，跳过")
            continue
        
        # 平移坐标到原点
        new_pos = sub_atoms.get_positions().copy()
        new_pos[:, 0] -= bx
        new_pos[:, 1] -= by
        sub_atoms.set_positions(new_pos)
        
        # 设置晶胞为窗口大小（15×15 Å），而不是lattice_params
        a_new = [WIN_SIZE, 0, 0]  # 15 Å
        b_new = [0, WIN_SIZE, 0]  # 15 Å
        c_new = atoms.cell[2]     # 保留原c方向
        
        sub_atoms.set_cell([a_new, b_new, c_new])
        sub_atoms.set_pbc((True, True, False))
        
        # 合并相近原子
        sub_atoms = merge_close_atoms(sub_atoms)
        
        # 保存CIF
        write(output_cif, sub_atoms, format="cif")
        
        print(f"  窗口 {i+1}: 左下角({bx:.1f}, {by:.1f}) 窗口大小({WIN_SIZE}×{WIN_SIZE}) 原子数={len(sub_atoms)} 评分(P,U,D,Lattice,Combined)=({p:.3f},{u:.3f},{d:.3f},{lattice_score:.3f},{combined_score:.3f}) -> {output_cif}")


def save_window_analysis_report(best_windows, output_file="window_analysis_report.txt"):
    """
    保存窗口分析报告
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("候选窗口分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"窗口大小: {WIN_SIZE}×{WIN_SIZE} Å\n")
        f.write(f"总窗口数: {len(best_windows)}\n\n")
        
        f.write("注意：保存的CIF文件包含大窗口视野内的所有原子，晶胞大小为窗口大小\n\n")
        
        for i, window in enumerate(best_windows):
            (bx, by), scores, combined_score = window
            p, u, d, lattice_score = scores
            
            f.write(f"窗口 {i+1}:\n")
            f.write(f"  位置: 左下角({bx:.1f}, {by:.1f})\n")
            f.write(f"  窗口大小: {WIN_SIZE}×{WIN_SIZE} Å\n")
            f.write(f"  晶胞大小: {WIN_SIZE}×{WIN_SIZE} Å\n")
            f.write(f"  周期性评分: {p:.3f}\n")
            f.write(f"  均匀性评分: {u:.3f}\n")
            f.write(f"  密度评分: {d:.3f}\n")
            f.write(f"  晶格匹配评分: {lattice_score:.3f}\n")
            f.write(f"  综合评分: {combined_score:.3f}\n")
            f.write(f"  输出文件: candidate_window_{i+1:02d}.cif\n\n")
        
        # 统计信息
        scores_array = np.array([w[2] for w in best_windows])  # combined_score
        f.write("统计信息:\n")
        f.write(f"  最高评分: {scores_array.max():.3f}\n")
        f.write(f"  最低评分: {scores_array.min():.3f}\n")
        f.write(f"  平均评分: {scores_array.mean():.3f}\n")
        f.write(f"  评分标准差: {scores_array.std():.3f}\n")
    
    print(f"窗口分析报告已保存到: {output_file}")


if __name__ == "__main__":
    # 1. 读结构
    atoms, pos_xy, cell_abc = load_structure(INPUT_CIF)
    a, b, c = cell_abc
    total_area = a * b

    # 2. 寻找最佳窗口
    (bx, by), (p,u,d) = find_best_window(pos_xy, (a,b), total_area)
    print(f"最佳窗口左下角：({bx:.1f}, {by:.1f}) 评分(P,U,D)=({p:.3f},{u:.3f},{d:.3f})")

    # 3. 定义配位关系：Sn1S2
    coordination_ratio = {'Sn': 1, 'S': 2}
    
    # 4. 提取并写出 shrink.cif，应用配位关系约束
    extract_and_write_with_lattice_constraint(
        atoms, pos_xy, (bx, by), (a, b, 90.0), "shrink.cif", coordination_ratio
    )
