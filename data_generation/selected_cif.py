import os
import shutil
from pymatgen.core import Structure

# -----------------------
# 参数设置
# -----------------------
source_dir = "./data_generation/all_cif_files"      # CIF 文件所在的文件夹路径，请替换为实际路径
target_dir = "./data_generation/selected_cif_files"    # 筛选后文件存放的目标文件夹路径，请替换为实际路径
thickness_threshold = 4.0   # z 轴原子分布范围阈值（单位 Å），例如小于 5 Å
neighbor_cutoff = 5.0       # 判定平面内连续性的最近邻距离阈值（单位 Å）
max_vacuum_gap = 1.0        # c 轴中真空层最小要求（单位 Å），用于判断是否为二维材料
max_unique_elements = 3     # 材料中允许的最大元素种类数

# 确保目标文件夹存在
os.makedirs(target_dir, exist_ok=True)

selected_files = []  # 用于保存符合条件的文件名列表

# -----------------------
# 遍历源文件夹中的所有 CIF 文件
# -----------------------
for filename in os.listdir(source_dir):
    if not filename.lower().endswith(".cif"):
        continue  # 非 CIF 文件跳过

    file_path = os.path.join(source_dir, filename)
    
    try:
        # 读取 CIF 结构
        structure = Structure.from_file(file_path)
    except Exception as e:
        print(f"读取 {filename} 时出错: {e}")
        continue
    
    # -----------------------
    # 新增条件：元素种类数不超过 max_unique_elements
    # -----------------------
    unique_elements = structure.composition.elements  # 获取组成元素列表
    if len(unique_elements) > max_unique_elements:
        # 如果元素种类超过限制，则跳过该文件
        continue
    
    # -----------------------
    # 判断 z 轴厚度：计算所有原子 z 坐标的范围
    # -----------------------
    z_coords = [site.coords[2] for site in structure.sites]
    if not z_coords:
        continue
    z_range = max(z_coords) - min(z_coords)
    
    if z_range > thickness_threshold:
        # 如果 z 轴厚度超过阈值，则认为不满足二维要求，跳过
        continue

    # -----------------------
    # 检查 c 轴是否有足够真空层（避免三维堆叠）：若 c 轴长度与 z_range 差值过小，则认为是三维结构
    # -----------------------
    c_axis = structure.lattice.c  # c 轴长度
    if (c_axis - z_range) < max_vacuum_gap:
        continue

    # -----------------------
    # 检查平面内连续性：判断结构在 xy 平面内是否连续延展
    # -----------------------
    frac_coords = [site.frac_coords for site in structure.sites]  # 分数坐标 (0~1)
    lattice_matrix = structure.lattice.matrix  # 晶胞矩阵
    in_plane_continuous = True
    for i, frac_i in enumerate(frac_coords):
        found_neighbor = False
        # 检查当前原子在本胞元及相邻 8 个胞元中是否存在邻居
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # 遍历其他所有原子
                for j, frac_j in enumerate(frac_coords):
                    if i == j:
                        continue
                    # 考虑周期性：在 x, y 方向分别加上 dx, dy
                    disp = [frac_j[0] + dx - frac_i[0], frac_j[1] + dy - frac_i[1], frac_j[2] - frac_i[2]]
                    # 将分数坐标差转换为笛卡尔坐标差
                    cart_disp = disp[0]*lattice_matrix[0] + disp[1]*lattice_matrix[1] + disp[2]*lattice_matrix[2]
                    dist = sum(comp**2 for comp in cart_disp) ** 0.5
                    if dist < neighbor_cutoff:
                        found_neighbor = True
                        break
                if found_neighbor:
                    break
            if found_neighbor:
                break
        if not found_neighbor:
            in_plane_continuous = False
            break
    
    if not in_plane_continuous:
        continue  # 若平面内不连续，则跳过

    # -----------------------
    # 该结构满足所有条件，记录并复制文件
    # -----------------------
    selected_files.append(filename)
    shutil.copy2(file_path, os.path.join(target_dir, filename))

# -----------------------
# 输出筛选结果
# -----------------------
print("筛选出满足要求的二维材料结构文件共 {} 个：".format(len(selected_files)))
for fname in selected_files:
    print(" - " + fname)
