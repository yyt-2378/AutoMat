import os, re, math, argparse, numpy as np, shlex
from collections import defaultdict, Counter


def wrap01(x):
    return x - np.floor(x)

def wrap_pm05(d):
    return (d + 0.5) % 1.0 - 0.5

def read_cif(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f]
    cell = {}
    atoms = []
    for line in lines:
        if line.startswith('_cell_length_a'):
            cell['a'] = float(line.split()[-1].strip().strip("'").strip('"'))
        elif line.startswith('_cell_length_b'):
            cell['b'] = float(line.split()[-1].strip().strip("'").strip('"'))
        elif line.startswith('_cell_length_c'):
            cell['c'] = float(line.split()[-1].strip().strip("'").strip('"'))
        elif line.startswith('_cell_angle_alpha'):
            cell['alpha'] = float(line.split()[-1].strip().strip("'").strip('"'))
        elif line.startswith('_cell_angle_beta'):
            cell['beta'] = float(line.split()[-1].strip().strip("'").strip('"'))
        elif line.startswith('_cell_angle_gamma'):
            cell['gamma'] = float(line.split()[-1].strip().strip("'").strip('"'))
    # atom loop
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith('loop_'):
            i += 1
            headers = []
            while i < n and lines[i].startswith('_'):
                headers.append(lines[i]); i += 1
            if any(h.startswith('_atom_site_') for h in headers):
                def find_col(opts):
                    for idx, h in enumerate(headers):
                        for opt in opts:
                            if h.endswith(opt):
                                return idx
                    return None
                i_type = find_col(['_atom_site_type_symbol','_atom_site_type_symbo'])
                i_x = find_col(['_atom_site_fract_x'])
                i_y = find_col(['_atom_site_fract_y'])
                i_z = find_col(['_atom_site_fract_z'])
                while i < n and lines[i] and not lines[i].startswith('loop_') and not lines[i].startswith('_') and not lines[i].startswith('data_'):
                    parts = shlex.split(lines[i])
                    if i_type is not None and i_x is not None and i_y is not None and i_z is not None and len(parts) > i_z:
                        sym = parts[i_type].strip("'").strip('"')
                        try:
                            fx = float(parts[i_x]); fy = float(parts[i_y]); fz = float(parts[i_z])
                            frac = np.array([fx, fy, fz], dtype=float)
                            frac[:2] = wrap01(frac[:2])
                            atoms.append({'element': sym, 'frac': frac})
                        except:
                            pass
                    i += 1
            else:
                while i < n and lines[i] and not lines[i].startswith('loop_') and not lines[i].startswith('_') and not lines[i].startswith('data_'):
                    i += 1
        else:
            i += 1
    return cell, atoms

def write_cif(path, cell, atoms, spacegroup_name="P 1"):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("data_minimal_cell\n")
        formula = Counter([a['element'] for a in atoms])
        formula_str = " ".join(f"{el}{(str(cnt) if cnt>1 else '')}" for el, cnt in sorted(formula.items()))
        f.write(f'_chemical_formula_sum "{formula_str}"\n')
        f.write(f"_cell_length_a    {cell['a']:.8f}\n")
        f.write(f"_cell_length_b    {cell['b']:.8f}\n")
        f.write(f"_cell_length_c    {cell['c']:.8f}\n")
        f.write(f"_cell_angle_alpha {cell['alpha']:.2f}\n")
        f.write(f"_cell_angle_beta  {cell['beta']:.2f}\n")
        f.write(f"_cell_angle_gamma {cell['gamma']:.6f}\n\n")
        f.write(f'_space_group_name_H-M_alt "{spacegroup_name}"\n')
        f.write(f"_space_group_IT_number 1\n\n")
        f.write("loop_\n")
        f.write("  _space_group_symop_operation_xyz\n")
        f.write("  'x, y, z'\n\n")
        f.write("loop_\n")
        f.write("  _atom_site_type_symbol\n")
        f.write("  _atom_site_label\n")
        f.write("  _atom_site_fract_x\n")
        f.write("  _atom_site_fract_y\n")
        f.write("  _atom_site_fract_z\n")
        f.write("  _atom_site_occupancy\n")
        counters = defaultdict(int)
        for a in atoms:
            counters[a['element']] += 1
            lbl = f"{a['element']}{counters[a['element']]}".ljust(6)
            fx, fy, fz = a['frac']
            fx, fy = wrap01(fx), wrap01(fy)
            f.write(f"  {a['element']}  {lbl}  {fx:.10f}  {fy:.10f}  {fz:.10f}  1.0\n")

def count_matches(A, B, t, tol=0.02):
    bins = defaultdict(list)
    g = tol
    for a in A:
        key = (a['element'], int(np.floor(wrap01(a['frac'][0])/g)), int(np.floor(wrap01(a['frac'][1])/g)))
        bins[key].append(wrap01(a['frac'][:2]))
    cnt = 0
    for b in B:
        shifted = wrap01(b['frac'][:2] + t)
        key = (b['element'], int(np.floor(shifted[0]/g)), int(np.floor(shifted[1]/g)))
        found = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for axy in bins.get((b['element'], key[1]+dx, key[2]+dy), []):
                    diff = wrap_pm05(shifted - axy)
                    if np.linalg.norm(diff) <= tol:
                        found = True
                        break
                if found: break
            if found: break
        if found:
            cnt += 1
    return cnt

def best_translation(A, B, tol=0.02, max_candidates=200):
    cands = []
    elems = sorted(set([a['element'] for a in A]) | set([b['element'] for b in B]))
    for e in elems:
        Apos = [wrap01(a['frac'][:2]) for a in A if a['element']==e]
        Bpos = [wrap01(b['frac'][:2]) for b in B if b['element']==e]
        for ap in Apos:
            for bp in Bpos:
                t = wrap_pm05(ap - bp)
                cands.append(t)
    grid = defaultdict(int); grid_vec = {}
    for t in cands:
        key = (int(round(t[0]/tol)), int(round(t[1]/tol)))
        grid[key] += 1
        if key not in grid_vec: grid_vec[key] = t
    top_keys = sorted(grid, key=lambda k: grid[k], reverse=True)[:max_candidates]
    best_t, best_score = None, -1
    for k in top_keys:
        t = grid_vec[k]
        score = count_matches(A, B, t, tol=tol)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score

def shift_atoms(atoms, t):
    out = []
    for a in atoms:
        frac = a['frac'].copy()
        frac[:2] = wrap01(frac[:2] + t)
        out.append({'element': a['element'], 'frac': frac})
    return out

def dedup_atoms(atoms, xy_tol=0.02, z_tol=0.05):
    unique = []
    for a in atoms:
        a2 = {'element': a['element'], 'frac': a['frac'].copy()}
        a2['frac'][:2] = wrap01(a2['frac'][:2])
        found = False
        for u in unique:
            if a2['element'] == u['element']:
                diff = wrap_pm05(a2['frac'][:2] - u['frac'][:2])
                if np.linalg.norm(diff) < xy_tol and abs(a2['frac'][2]-u['frac'][2]) < z_tol:
                    u['frac'] = (u['frac'] + a2['frac'])/2.0
                    u['frac'][:2] = wrap01(u['frac'][:2])
                    found = True
                    break
        if not found:
            unique.append(a2)
    return unique

def candidate_vectors_pairdiff(atoms, tol=0.02, maxlen=0.5, max_cands=400):
    diffs = []
    by_elem = defaultdict(list)
    for a in atoms:
        by_elem[a['element']].append(wrap01(a['frac'][:2]))
    for e, poslist in by_elem.items():
        n = len(poslist)
        for i in range(n):
            for j in range(i+1, n):
                d = wrap_pm05(poslist[j] - poslist[i])
                L = np.linalg.norm(d)
                if 1e-6 < L < maxlen:
                    diffs.append(d); diffs.append(-d)
    grid = defaultdict(int); grid_vec = {}
    for d in diffs:
        key = (int(round(d[0]/tol)), int(round(d[1]/tol)))
        grid[key] += 1
        if key not in grid_vec: grid_vec[key] = d
    def score_vec(v): return count_matches(atoms, atoms, v, tol=tol)
    cands = list(grid_vec.values())
    scored = []
    for v in cands[:max_cands]:
        scored.append((score_vec(v), np.linalg.norm(v), v))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored

def select_two_basis(cands, min_cross=5e-4):
    if not cands: return None, None
    v1 = cands[0][2]
    for sc, L, v in cands[1:]:
        cross = abs(v1[0]*v[1] - v1[1]*v[0])
        if cross > min_cross:
            return v1, v
    return v1, None

def reduce_basis_2d(v1, v2):
    a = v1.copy(); b = v2.copy()
    def norm2(x): return float(x[0]*x[0] + x[1]*x[1])
    if norm2(b) < norm2(a):
        a, b = b, a
    changed = True
    while changed:
        changed = False
        mu = round((a[0]*b[0] + a[1]*b[1]) / max(1e-12, norm2(a)))
        cand = b - mu * a
        if norm2(cand) + 1e-12 < norm2(b):
            b = cand; changed = True
        if norm2(b) < norm2(a):
            a, b = b, a; changed = True
    a = wrap_pm05(a); b = wrap_pm05(b)
    return a, b

def transform_to_new_cell(atoms, v1, v2):
    M = np.column_stack([v1, v2])
    Minv = np.linalg.inv(M)
    new_atoms = []
    for a in atoms:
        p = wrap01(a['frac'][:2])
        uv = Minv @ p
        uv_mod = uv - np.floor(uv)
        new_atoms.append({'element': a['element'], 'frac': np.array([uv_mod[0], uv_mod[1], a['frac'][2]], dtype=float)})
    return new_atoms

def lattice_from_frac_vectors(v1, v2, a0, b0):
    ax = np.array([a0, 0.0]); ay = np.array([0.0, b0])
    a_cart = v1[0]*ax + v1[1]*ay
    b_cart = v2[0]*ax + v2[1]*ay
    la = float(np.linalg.norm(a_cart))
    lb = float(np.linalg.norm(b_cart))
    cosg = float(np.dot(a_cart, b_cart)/(max(1e-12, la*lb)))
    cosg = max(-1.0, min(1.0, cosg))
    gamma = float(math.degrees(math.acos(cosg)))
    return la, lb, gamma, a_cart, b_cart

# ---------- 周期聚类（无依赖 DBSCAN） ----------
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n)); self.s = [1]*n
    def find(self, x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self, a,b):
        ra, rb = self.find(a), self.find(b)
        if ra==rb: return
        if self.s[ra]<self.s[rb]: ra,rb=rb,ra
        self.p[rb]=ra; self.s[ra]+=self.s[rb]

def periodic_distance(u, v):
    d = np.abs(u - v)
    d = np.minimum(d, 1.0 - d)
    return float(np.linalg.norm(d))

def periodic_cluster(points_uv, eps=0.06):
    """对二维分数坐标做周期 DBSCAN-like 聚类，返回簇标签和簇中心"""
    n = len(points_uv)
    if n==0: return [], []
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            if periodic_distance(points_uv[i], points_uv[j]) <= eps:
                uf.union(i,j)
    # 收集类别
    roots = [uf.find(i) for i in range(n)]
    clusters = defaultdict(list)
    for idx,r in enumerate(roots):
        clusters[r].append(points_uv[idx])
    centers = []
    for r, pts in clusters.items():
        pts = np.array(pts)
        # 平均时考虑周期性：把点相对第一个点展开到最近像，再平均
        base = pts[0]
        unwrapped = []
        for p in pts:
            d = p - base
            d = d - np.round(d)  # 映射到 [-0.5,0.5]
            unwrapped.append(base + d)
        mean = np.mean(np.array(unwrapped), axis=0)
        centers.append(wrap01(mean))
    labels = [list(clusters.keys()).index(uf.find(i)) for i in range(n)]
    return labels, centers

def parse_basis_spec(spec):
    # 解析形如 "Sn:1,S:2" 的字符串
    want = {}
    if not spec: return want
    for part in spec.split(','):
        if not part.strip(): continue
        el, k = part.split(':')
        want[el.strip()] = int(k)
    return want

def _reconstruct_with_args(args):
    # 读取
    paths = [args.win1, args.win2, args.win3]
    cells, wins = [], []
    for p in paths:
        c, a = read_cif(p)
        if not c: raise RuntimeError(f"无法读取晶格参数：{p}")
        cells.append(c); wins.append(a)
    a0, b0, c0 = cells[0]['a'], cells[0]['b'], cells[0]['c']
    if not (all(abs(c['a']-a0)<1e-6 for c in cells) and all(abs(c['b']-b0)<1e-6 for c in cells)):
        raise AssertionError("三个 CIF 的 in-plane 晶格参数不一致")
    print("输入晶格： a0=%.4f Å, b0=%.4f Å, c0=%.4f Å" % (a0, b0, c0))

    # 对齐
    t12, s12 = best_translation(wins[0], wins[1], tol=args.vec_tol)
    t13, s13 = best_translation(wins[0], wins[2], tol=args.vec_tol)
    print("最佳平移： t12=", t12, "score=", s12, " | t13=", t13, "score=", s13)
    win2 = shift_atoms(wins[1], t12) if t12 is not None else wins[1]
    win3 = shift_atoms(wins[2], t13) if t13 is not None else wins[2]
    merged = wins[0] + win2 + win3
    merged_dedup = dedup_atoms(merged, xy_tol=args.dedup_xy, z_tol=args.dedup_z)
    print("去重后原子数：", Counter([a['element'] for a in merged_dedup]))
    # 自动基元规格：当 --auto_basis=1 或 --basis 为空字符串时，按元素种类各取1
    dedup_counts = Counter([a['element'] for a in merged_dedup])
    basis_spec_eff = args.basis if (args.basis is not None and str(args.basis).strip()!='') else None
    if (getattr(args, 'auto_basis', 0) or not basis_spec_eff) and len(dedup_counts)>0:
        basis_spec_eff = ", ".join(f"{el}:1" for el in sorted(dedup_counts.keys()))
        print("采用自动基元规格（近似各取1）：", basis_spec_eff)

    # 候选基矢
    scored = candidate_vectors_pairdiff(merged_dedup, tol=args.vec_tol, maxlen=args.vec_maxlen, max_cands=600)
    v1, v2 = select_two_basis(scored, min_cross=args.vec_cross_min)
    if v2 is None:
        raise AssertionError("没有找到两条线性无关的平移基矢；请调参 --vec_maxlen/--vec_tol/--vec_cross_min")

    # 规整
    v1r, v2r = reduce_basis_2d(v1, v2)

    # 实空间参数
    la, lb, gamma, a_cart, b_cart = lattice_from_frac_vectors(v1r, v2r, a0=a0, b0=b0)

    # 六角吸附（可选，且不覆盖非六角/用户指定晶格）
    if args.snap_hex and (getattr(args, 'lat_gamma', None) is None) and (abs(gamma-90)>=5) and (abs(gamma-60)<8 or abs(180-gamma-60)<8 or abs(gamma-120)<8):
        print("检测到接近六角；执行 snap 到 |a|=|b|, gamma=60°")
        L = (la+lb)/2.0
        # 在旧的分数空间里把 v1r,v2r 重正交化到等长+60度：
        # 我们只重定向 v1r,v2r 的 2x2 矩阵到等长+60°的“等效”矩阵（保留原近似方向）
        # 方案：用当前 a_cart,b_cart 的单位向量，生成目标向量：
        ua = a_cart/np.linalg.norm(a_cart); ub = b_cart/np.linalg.norm(b_cart)
        # 让 ub' 为 ua 旋转 +60° 的方向（在 xy 平面）：
        rot60 = np.array([[np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60))],
                           [np.sin(np.deg2rad(60)),  np.cos(np.deg2rad(60))]])
        ua2 = ua
        ub2 = rot60 @ ua  # 近似
        a_cart_new = ua2*L
        b_cart_new = ub2*L
        # 反解成分数向量（由于原底面是矩形 a0,b0）
        # [a_cart_new] = [a0 0; 0 b0] * [v1x; v1y]
        v1r = np.array([a_cart_new[0]/a0, a_cart_new[1]/b0])
        v2r = np.array([b_cart_new[0]/a0, b_cart_new[1]/b0])
        la, lb, gamma, _, _ = lattice_from_frac_vectors(v1r, v2r, a0=a0, b0=b0)

    print("选用基矢（分数坐标）: v1=", v1r, " v2=", v2r)
    print("新单胞参数： a=%.5f Å, b=%.5f Å, γ=%.5f°" % (la, lb, gamma))

    # 映射到新单胞
    atoms_new = transform_to_new_cell(merged_dedup, v1r, v2r)

    # 周期聚类 -> 取期望基元数
    want = parse_basis_spec(basis_spec_eff if basis_spec_eff is not None else args.basis)
    eps = max(0.05, args.vec_tol*2.0)
    basis_atoms = []
    for el in sorted(set(a['element'] for a in atoms_new)):
        pts_uv = np.array([a['frac'][:2] for a in atoms_new if a['element']==el])
        zs = np.array([a['frac'][2] for a in atoms_new if a['element']==el])
        if len(pts_uv)==0: continue
        labels, centers = periodic_cluster(pts_uv, eps=eps)
        # 为每个簇统计权重（点数），并给出 z 的平均
        k = want.get(el, len(centers))  # 若未指定，则保留所有簇
        cluster_stats = defaultdict(list)
        for idx, lab in enumerate(labels):
            cluster_stats[lab].append((pts_uv[idx], zs[idx]))
        # 生成 (size, center_uv, mean_z)
        entries = []
        for lab, lst in cluster_stats.items():
            size = len(lst)
            uv_center = centers[lab]
            mean_z = float(np.mean([z for _,z in lst]))
            entries.append((size, uv_center, mean_z))
        # 选前 k 个最大簇
        entries.sort(key=lambda x: -x[0])
        chosen = entries[:k]
        for size, uv_center, mean_z in chosen:
            basis_atoms.append({'element': el, 'frac': np.array([uv_center[0], uv_center[1], mean_z])})

    # 写出 CIF
    new_cell = {'a': la, 'b': lb, 'c': cells[0]['c'], 'alpha': 90.0, 'beta': 90.0, 'gamma': gamma}
    # 覆盖导出晶格参数（若用户提供）
    if getattr(args, 'lat_a', None) is not None: new_cell['a'] = args.lat_a
    if getattr(args, 'lat_b', None) is not None: new_cell['b'] = args.lat_b
    if getattr(args, 'lat_c', None) is not None: new_cell['c'] = args.lat_c
    if getattr(args, 'lat_alpha', None) is not None: new_cell['alpha'] = args.lat_alpha
    if getattr(args, 'lat_beta', None) is not None: new_cell['beta'] = args.lat_beta
    if getattr(args, 'lat_gamma', None) is not None: new_cell['gamma'] = args.lat_gamma
    print("最终确定单胞参数： a=%.5f Å, b=%.5f Å, alpha=%.5f°" % (new_cell['a'], new_cell['b'], new_cell['alpha']))
    # 排序：金属在前
    order = {'Sn':0, 'Mo':0, 'W':0, 'S':1, 'Se':1, 'Te':1}
    basis_atoms.sort(key=lambda a: (order.get(a['element'], 9), a['element']))
    write_cif(args.out, new_cell, basis_atoms, spacegroup_name="P 1")
    print("输出最小基元（按 --basis 约束）：", Counter([a['element'] for a in basis_atoms]))
    print("已写出 ->", args.out)
    return args.out, new_cell, basis_atoms

def reconstruct_minimal_cell(win1, win2, win3,
                             vec_tol=0.02,
                             vec_maxlen=0.45,
                             vec_cross_min=0.0002,
                             dedup_xy=0.02,
                             dedup_z=0.05,
                             use_grid=1,
                             grid_step=0.04,
                             fine_step=0.01,
                             snap_hex=1,
                             basis=None,
                             out='minimal_cell_from_top3_v3.cif',
                             auto_basis=1,
                             lat_a=None,
                             lat_b=None,
                             lat_c=None,
                             lat_alpha=None,
                             lat_beta=None,
                             lat_gamma=None):
    args = argparse.Namespace(
        win1=win1,
        win2=win2,
        win3=win3,
        vec_tol=vec_tol,
        vec_maxlen=vec_maxlen,
        vec_cross_min=vec_cross_min,
        dedup_xy=dedup_xy,
        dedup_z=dedup_z,
        use_grid=use_grid,
        grid_step=grid_step,
        fine_step=fine_step,
        snap_hex=snap_hex,
        basis=basis,
        out=out,
        auto_basis=auto_basis,
        lat_a=lat_a,
        lat_b=lat_b,
        lat_c=lat_c,
        lat_alpha=lat_alpha,
        lat_beta=lat_beta,
        lat_gamma=lat_gamma
    )
    return _reconstruct_with_args(args)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--win1', type=str, default='candidate_window_01.cif')
    ap.add_argument('--win2', type=str, default='candidate_window_02.cif')
    ap.add_argument('--win3', type=str, default='candidate_window_03.cif')
    ap.add_argument('--vec_tol', type=float, default=0.02)
    ap.add_argument('--vec_maxlen', type=float, default=0.45)
    ap.add_argument('--vec_cross_min', type=float, default=0.0002)
    ap.add_argument('--dedup_xy', type=float, default=0.02)
    ap.add_argument('--dedup_z', type=float, default=0.05)
    ap.add_argument('--use_grid', type=int, default=1)
    ap.add_argument('--grid_step', type=float, default=0.04)
    ap.add_argument('--fine_step', type=float, default=0.01)
    ap.add_argument('--snap_hex', type=int, default=1, help='若 gamma 接近60/120度，吸附为完美六角并令 |a|≈|b|')
    ap.add_argument('--basis', type=str, default=None, help='期望的最小基元簇数，例如 "Sn:1,S:2"')
    ap.add_argument('--out', type=str, default='minimal_cell_from_top3_windows.cif')
    # 额外可选参数：自动基元推断与晶格参数覆盖
    ap.add_argument('--auto_basis', type=int, default=1, help='为1或 --basis 为空时，按元素种类各取1个簇')
    ap.add_argument('--lat_a', type=float, default=None, help='覆盖导出 CIF 的 a')
    ap.add_argument('--lat_b', type=float, default=None, help='覆盖导出 CIF 的 b')
    ap.add_argument('--lat_c', type=float, default=None, help='覆盖导出 CIF 的 c')
    ap.add_argument('--lat_alpha', type=float, default=None, help='覆盖导出 CIF 的 alpha')
    ap.add_argument('--lat_beta', type=float, default=None, help='覆盖导出 CIF 的 beta')
    ap.add_argument('--lat_gamma', type=float, default=None, help='覆盖导出 CIF 的 gamma')
    args = ap.parse_args()

    # 将 CLI 参数委托给可复用的核心函数
    _reconstruct_with_args(args)

if __name__ == "__main__":
    main()

