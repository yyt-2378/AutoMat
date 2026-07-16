from collections import Counter
import os
from ase.io import read
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict


def compare_elemental_composition(cif1, cif2):
    """Jaccard index: 比较两个结构的组成元素是否一致"""
    atoms1 = read(cif1)
    atoms2 = read(cif2)
    elems1 = set(atoms1.get_chemical_symbols())
    elems2 = set(atoms2.get_chemical_symbols())
    intersection = elems1 & elems2
    union = elems1 | elems2
    return len(intersection) / len(union) if union else 0.0


def compare_atom_counts(cif1, cif2):
    """余弦相似度：比较不同元素的数量分布是否一致"""
    atoms1 = read(cif1)
    atoms2 = read(cif2)
    count1 = Counter(atoms1.get_chemical_symbols())
    count2 = Counter(atoms2.get_chemical_symbols())
    all_elems = list(set(count1) | set(count2))
    vec1 = np.array([count1.get(elem, 0) for elem in all_elems])
    vec2 = np.array([count2.get(elem, 0) for elem in all_elems])
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return float(np.dot(vec1, vec2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0


def compare_lattice_2d(atoms1, atoms2, length_tol=0.05, angle_tol=5):
    """
    判断二维材料结构晶格是否相似（忽略c轴长度和z轴角度）。
    """
    cell1 = atoms1.cell.lengths()
    cell2 = atoms2.cell.lengths()
    angles1 = atoms1.cell.angles()
    angles2 = atoms2.cell.angles()

    # 只比较 a, b 和 α, β
    ab1, ab2 = cell1[:2], cell2[:2]
    rel_diff = np.abs((ab1 - ab2) / ab1)
    if np.any(rel_diff > length_tol):
        return False

    angle_diff = np.abs(np.array(angles1[:2]) - np.array(angles2[:2]))
    if np.any(angle_diff > angle_tol):
        return False

    return True


def compare_structure_fractional_2d(cif1_path, cif2_path, dist_thresh=0.2):
    """
    比较两个 CIF 结构是否在 2D 结构下相似（考虑 lattice + 元素分组相对坐标）。
    """
    atoms1 = read(cif1_path)
    atoms2 = read(cif2_path)

    # 1. 晶格判断（忽略c轴）
    if not compare_lattice_2d(atoms1, atoms2):
        print("Lattice mismatch.")
        return 0.0

    # 2. 化学组分判断
    symbols1 = atoms1.get_chemical_symbols()
    symbols2 = atoms2.get_chemical_symbols()
    unique1 = set(symbols1)
    unique2 = set(symbols2)
    if unique1 != unique2:
        print("Element mismatch.")
        return 0.0

    # 3. 获取归一化坐标并忽略z轴
    frac1 = atoms1.get_scaled_positions()[:, :2]
    frac2 = atoms2.get_scaled_positions()[:, :2]

    # 4. 元素分组后匹配坐标
    total_mse = 0.0
    total_atoms = 0

    for element in unique1:
        idx1 = [i for i, s in enumerate(symbols1) if s == element]
        idx2 = [i for i, s in enumerate(symbols2) if s == element]

        if len(idx1) != len(idx2):
            print(f"Atom count mismatch for element {element}")
            return 0.0

        pos1 = frac1[idx1]
        pos2 = frac2[idx2]

        tree = cKDTree(pos2)
        dists, _ = tree.query(pos1, k=1)

        if np.any(dists > dist_thresh):
            print(f"Distance too large for element {element}")
            return 0.0

        total_mse += np.sum(dists ** 2)
        total_atoms += len(pos1)

    avg_mse = total_mse / total_atoms
    similarity_score = np.exp(-avg_mse)
    return similarity_score


def overall_similarity(cif1, cif2, weights=(0.3, 0.3, 0.4)):
    comp_sim = compare_elemental_composition(cif1, cif2)
    count_sim = compare_atom_counts(cif1, cif2)
    spatial_sim = compare_structure_fractional_2d(cif1, cif2)
    overall = weights[0]*comp_sim + weights[1]*count_sim + weights[2]*spatial_sim
    return {
        "composition_similarity": round(comp_sim, 4),
        "count_similarity": round(count_sim, 4),
        "spatial_similarity": round(spatial_sim, 4),
        "overall_similarity": round(overall, 4)
    }


# 示例：替换为你自己的CIF路径
if __name__ == "__main__":
    cif1 = r"./orthogonal_2dm-767_supercell_12x12x1_dose60000_sampling0.1_iDPC_V3_reconstructed.cif"
    cif2 = r"./orthogonal_2dm-767_supercell_12x12x1.cif"

    if os.path.exists(cif1) and os.path.exists(cif2):
        sim = overall_similarity(cif1, cif2)
        print("结构相似度比较结果：")
        for k, v in sim.items():
            print(f"{k}: {v}")
        threshold = 0.8
        if sim['overall_similarity'] >= threshold:
            print("✅ 两个CIF结构被认为基本一致。")
        else:
            print("❌ 两个CIF结构相差较大。")
    else:
        print("请检查CIF文件路径是否正确。")
