import os
import sys
import cv2
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import write
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from ase.io import read
from mpl_toolkits.mplot3d import Axes3D  # 确保引入3D支持
from pymatgen.core import Structure, Lattice


# 可配置参数, 替换成你输入的电镜图像的组成元素和分辨率
PIXEL_SIZE = 0.1
ELEMENTS_TYPE = ['As', 'Si']
MAX_NUM_ITER = 4
MAX_ATOMS_NUM = 50


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
    return A * np.exp(-(((x - x0)**2) / (2 * sx**2) + ((y - y0)**2) / (2 * sy**2))) + offset


# 图像处理主函数
def process_image(image_path):
    # 1. 图像读入与归一化
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 读取图像（保留原始深度）
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found or cannot be opened.")
    # 如果读到彩色图，将其转换为灰度
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将图像转换为浮点型并归一化到 [0,1]
    img = img.astype(np.float32)
    # 防止除零，若图像平坦则直接返回
    if img.max() - img.min() < 1e-8:
        img_norm = np.zeros_like(img, dtype=np.float32)
    else:
        img_norm = (img - img.min()) / (img.max() - img.min())

    # 2. 图像去噪（使用预训练模型，如 MOE-DIVAE-SR）
    # 假定存在一个预训练模型可以调用，这里用占位符表示模型推理
    # 如: model = load_model(model_path); img_denoised = model.predict(img_norm)
    img_denoised = img_norm  # 如果无可用模型，则跳过去噪（或选择其他去噪方法）

    # 3. 图像分割（阈值分割找到原子亮斑）
    # 将浮点图像转换为8位以应用 Otsu 阈值
    img_uint8 = (img_denoised * 255).astype(np.uint8)
    # 使用 Otsu 方法自动求阈值进行二值化
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 连通区域标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # 如果存在背景标签0，跳过背景
    # 提取连通区域的像素强度总和和计数，用于计算平均亮度
    flat_labels = labels.flatten().astype(np.int32)
    flat_intensity = img_denoised.flatten()
    # 计算每个标签的强度总和和像素数
    sums = np.bincount(flat_labels, weights=flat_intensity)
    counts = np.bincount(flat_labels)
    # 去除背景 (label 0) 的计数避免除零
    sums = sums[1:]
    counts = counts[1:]
    num_regions = len(sums)  # 实际原子斑点数量
    if num_regions == 0:
        print(f"No atomic regions found in image {image_path}")
        return None, None  # 返回空结果
    mean_intensity = sums / np.maximum(counts, 1)  # 平均亮度

    # 可选：过滤掉非常小的区域（例如面积小于3像素），认为是噪声
    # 这里根据需要启用，默认保留所有区域
    # valid_labels = [i for i in range(num_regions) if stats[i+1, cv2.CC_STAT_AREA] >= 3]
    # (如需过滤，则需相应筛选 mean_intensity, stats, centroids 等列表)

    # 4. 平均亮度聚类识别元素类型
    K = len(ELEMENTS_TYPE)
    if K == 0:
        raise ValueError("known_elements list is empty. Please provide the element types present.")
    # 将平均亮度值进行 KMeans 聚类
    brightness_values = mean_intensity.reshape(-1, 1)
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0).fit(brightness_values)
    cluster_labels = kmeans.labels_  # 每个区域的聚类标签 (0 ~ K-1)
    centers = kmeans.cluster_centers_.flatten()

    # 根据亮度中心与原子序数的排序建立簇索引到元素符号的映射
    # 将聚类中心按亮度降序排序，同时将已知元素按原子序(Z)降序排序
    brightness_order = np.argsort(centers)[::-1]  # 聚类中心从高到低的索引顺序
    elems_sorted_by_Z = sorted(ELEMENTS_TYPE, key=lambda sym: atomic_numbers[sym], reverse=True)
    if len(elems_sorted_by_Z) != K:
        raise ValueError("Number of known elements does not match number of clusters.")
    # 创建聚类标签到元素的映射字典
    cluster_to_elem = {int(cluster_id): elems_sorted_by_Z[i]
                       for i, cluster_id in enumerate(brightness_order)}

    # 5. 对每个区域进行2D高斯拟合求中心坐标
    atom_positions_pixels = []  # 用于存储每个原子的中心像素坐标 (x, y)
    atom_symbols = []           # 用于存储每个原子的元素符号
    # 遍历每个原子区域（连通区域），获取精确中心和元素
    for label in range(1, num_labels):  # label从1开始到num_labels-1对应每个原子区域
        region_idx = label - 1  # 将label映射到mean_intensity的索引
        # 跳过已过滤的区域（如果有筛选条件，这里可根据有效标签列表判断）
        # if region_idx not in valid_labels: continue  # (如启用过滤)
        # 提取该区域的元素类型
        elem_cluster = cluster_labels[region_idx]
        elem_symbol = cluster_to_elem[int(elem_cluster)]
        # 从统计信息中取得该区域的边界框
        x_min = stats[label, cv2.CC_STAT_LEFT]
        y_min = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        # 提取子图和区域掩膜
        sub_img = img_denoised[y_min:y_min+h, x_min:x_min+w]
        sub_labels = labels[y_min:y_min+h, x_min:x_min+w]
        region_mask = (sub_labels == label)
        # 获取区域内的坐标（相对于子图）和强度值
        ys, xs = np.where(region_mask)
        intensities = sub_img[ys, xs]
        # 准备高斯拟合初始参数 (使用局部坐标系以减少数值偏移)
        # 强度加权质心作为初始中心
        sumI = intensities.sum()
        if sumI <= 0:
            # 若强度和为0（理论上不会发生，除非空区域），跳过
            continue
        x0_init = (xs * intensities).sum() / sumI
        y0_init = (ys * intensities).sum() / sumI
        # 初始幅值取区域内最大值减最小值，偏置取最小值
        I_min = intensities.min()
        I_max = intensities.max()
        A_init = I_max - I_min
        offset_init = I_min
        sigma_init = 1.5  # 初始标准差 (像素)，可根据需要调整
        initial_guess = (A_init, x0_init, y0_init, sigma_init, sigma_init, offset_init)
        try:
            # 执行非线性最小二乘高斯拟合
            popt, _ = curve_fit(gaussian_2d, (xs, ys), intensities, p0=initial_guess, maxfev=2000)
            # 拟合结果中心（相对于子图坐标）
            A_fit, x0_fit, y0_fit, sx_fit, sy_fit, offset_fit = popt
        except RuntimeError:
            # 若拟合未收敛，则使用强度质心作为中心
            x0_fit, y0_fit = x0_init, y0_init
        # 转换为全局像素坐标
        x_center = x_min + x0_fit
        y_center = y_min + y0_fit
        atom_positions_pixels.append((x_center, y_center))
        atom_symbols.append(elem_symbol)

    # 6. 像素坐标转换为物理坐标 (Å)
    cell_c = 5.0  # 适合2D材料投影，设定一个合适的厚度
    shifted_z = cell_c / 2
    base_z = 0.5
    atom_positions_ang = []
    for (x_pixel, y_pixel) in atom_positions_pixels:
        x_ang = x_pixel * PIXEL_SIZE
        y_ang = y_pixel * PIXEL_SIZE
        z_ang = base_z + np.random.uniform(-0.05, 0.05) + shifted_z  # ±0.05 Å 随机浮动
        atom_positions_ang.append((x_ang, y_ang, z_ang))

    # 7. 生成 ASE Atoms 对象并设置晶胞参数再导出结构
    atoms = Atoms(symbols=atom_symbols, positions=atom_positions_ang)

    # 设置正交晶胞尺寸，c轴固定为20Å，a和b由图像像素大小计算得出
    height, width = img.shape
    cell_a = width * PIXEL_SIZE
    cell_b = height * PIXEL_SIZE

    atoms.set_cell([[cell_a, 0, 0],
                    [0, cell_b, 0],
                    [0, 0, cell_c]])
    atoms.set_pbc((True, True, False))  # 设定周期性边界条件（xy周期，z非周期）

    # 根据输入图像文件名，生成输出结构文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = base_name + "_reconstructed.cif"

    write(output_file, atoms, format="cif", wrap=False)
    print(f"Processed {image_path}: detected {len(atom_symbols)} atoms, results saved to {output_file}")
    return atoms, output_file


def visualize_results(image_path, cif_file):
    """
    可视化流程：
    1. 动画前半段（约20秒）：在原图上逐步累积显示聚类结果——逐步增加显示原子点，
       呈现聚类趋势效果，每个点标记大小设为 s=8；
    2. 动画后半段（约20秒）：左侧展示原图空间中原子点由初始（缩小的映射）平滑过渡到最终结构坐标，
       右侧展示一个由读取的CIF结构进行绕z轴旋转的三维视图；
    3. 两部分动画拼接成一个整体动画，并保存为一个mp4文件。

    参数：
      image_path: 原图路径（用于聚类分析动画背景）
      cif_file: 保存的CIF文件路径，通过ASE读取结构，并用于旋转展示
    """
    # -------------------- 图像和结构数据读取 --------------------
    # 读取原始图像（灰度图）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image for visualization.")
        return
    img_height, img_width = img.shape

    # 读取CIF结构
    structure = read(cif_file)
    pos_ang = structure.get_positions()  # 期望 shape=(N_atoms, 3)，单位 Å
    # 将原子位置转换为图像像素坐标（反推：x_pixel = x_ang / PIXEL_SIZE）
    pos_pixel = pos_ang[:, :2] / PIXEL_SIZE
    # CIF结构中最终的平面坐标（单位 Å）
    pos_final = pos_ang[:, :2]
    symbols = structure.get_chemical_symbols()

    # 定义元素颜色映射（未指定元素默认用黑色）
    color_map = {'As': 'green', 'Si': 'red'}
    colors = [color_map.get(sym, 'black') for sym in symbols]

    # -------------------- 辅助函数：figure转为图像数组 --------------------
    def fig_to_array(fig):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        return arr

    # -------------------- 生成聚类结果的静态帧 --------------------
    fig_static = plt.figure(figsize=(6, 6))
    ax_static = fig_static.add_subplot(111)
    ax_static.imshow(img, cmap='gray', origin='upper')
    ax_static.scatter(pos_pixel[:, 0], pos_pixel[:, 1],
                      c=colors, s=8, edgecolor='yellow', linewidth=0.5)
    ax_static.axis('off')
    clustering_frame = fig_to_array(fig_static)
    plt.close(fig_static)

    # 计算左侧动画初始位置：原图映射位置的0.8倍（单位为 Å）
    pos_start = pos_pixel * PIXEL_SIZE * 0.8

    # -------------------- 动画参数设置 --------------------
    frames_cluster = 300   # 前半段帧数
    frames_transform = 200  # 后半段帧数
    total_frames = frames_cluster + frames_transform

    # -------------------- 创建动画画布和坐标系 --------------------
    fig_anim = plt.figure(figsize=(12, 6))
    # 左侧：2D坐标轴
    ax_left = fig_anim.add_subplot(121)
    ax_left.axis('off')
    # 在左侧显示原图背景，设定extent保证坐标映射正确
    ax_left.imshow(img, cmap='gray', origin='upper',
                   extent=[0, img_width, img_height, 0])
    # 初始化左侧散点图（以避免初始facecolors数组长度不匹配）
    scat_left = ax_left.scatter(np.empty((0, 2)), np.empty(0), s=8,
                                edgecolor='yellow', linewidth=0.5)

    # 右侧预留区域：创建两个重叠的坐标系
    # 2D轴用于显示聚类静态帧
    ax_right_2d = fig_anim.add_axes([0.55, 0.1, 0.4, 0.8])
    ax_right_2d.axis('off')
    ax_right_2d.imshow(clustering_frame, interpolation="nearest")
    ax_right_2d.set_title("CIF Reconstruction", fontsize=10)
    # 3D轴用于显示CIF结构旋转，初始设为隐藏
    ax_right_3d = fig_anim.add_axes([0.55, 0.1, 0.4, 0.8], projection='3d')
    ax_right_3d.set_visible(False)
    # 设置3D轴长宽比（需要 Matplotlib 3.3 及以上）
    try:
        ax_right_3d.set_box_aspect((1, 1, 0.5))
    except Exception:
        pass

    # -------------------- 更新函数 --------------------
    def update(frame):
        if frame < frames_cluster:
            # ----- Part 1: 聚类逐步累积显示 -----
            fraction = frame / (frames_cluster - 1)
            count = int(fraction * len(pos_pixel))
            if count > 0:
                new_offsets = pos_pixel[:count]
                new_colors = colors[:count]
            else:
                new_offsets = np.empty((0, 2))
                new_colors = []
            scat_left.set_offsets(new_offsets)
            scat_left.set_facecolors(new_colors)
            ax_left.set_title("Clustering Analysis on Input Image", fontsize=10)

            # 右侧：显示2D聚类静态帧
            ax_right_2d.set_visible(True)
            ax_right_3d.set_visible(False)
            ax_right_2d.clear()
            ax_right_2d.imshow(clustering_frame, interpolation="nearest")
            ax_right_2d.axis('off')
            ax_right_2d.set_title("CIF Reconstruction", fontsize=10)
        else:
            # ----- Part 2: 左侧平滑过渡，右侧显示3D旋转 -----
            t = (frame - frames_cluster) / (frames_transform - 1)
            # 左侧：点位从 pos_start 平滑插值到 pos_final
            interp_offsets = (1 - t) * pos_start + t * pos_final
            scat_left.set_offsets(interp_offsets)
            scat_left.set_facecolors(colors)
            ax_left.set_title("Image Space Transformation", fontsize=10)

            # 右侧：切换至3D显示
            ax_right_2d.set_visible(False)
            ax_right_3d.set_visible(True)
            ax_right_3d.cla()  # 清除当前内容
            ax_right_3d.set_title("Rotating CIF Structure", fontsize=10)
            # 绘制 3D 原子散点图
            ax_right_3d.scatter(pos_ang[:, 0], pos_ang[:, 1], pos_ang[:, 2],
                                c=colors, s=50, edgecolor='k')

            # 设置视角，利用旋转角度 t 计算 azimuth (绕z轴旋转一整圈)
            angle = 360 * t
            ax_right_3d.view_init(elev=30, azim=angle)

            # 调整轴范围：这里对 z 轴进行压缩，让结构更具立体感
            x_min, x_max = pos_ang[:, 0].min(), pos_ang[:, 0].max()
            y_min, y_max = pos_ang[:, 1].min(), pos_ang[:, 1].max()
            z_min, z_max = pos_ang[:, 2].min(), pos_ang[:, 2].max()
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            mid_z = (z_max + z_min) / 2
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
            factor_z = 0.5  # z轴压缩因子，可根据需要调整
            ax_right_3d.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_right_3d.set_ylim(mid_y - max_range, mid_y + max_range)
            ax_right_3d.set_zlim(mid_z - max_range * factor_z, mid_z + max_range * factor_z)
        return [scat_left]

    # -------------------- 创建动画并保存 --------------------
    ani = animation.FuncAnimation(fig_anim, update, frames=total_frames,
                                  interval=50, blit=True)
    writer = animation.FFMpegWriter(fps=20)
    ani.save("combined_animation.mp4", writer=writer)
    plt.close(fig_anim)


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # 重建流程：单张图像重建流程
    # ------------------------------------------------------------------
    image_path = "./data_generation/simulate_data/label/orthogonal_2dm-767_supercell_12x12x1_dose60000_sampling0.1_iDPC_V3.png"  # 请将此处替换为实际的输入PNG文件路径
    atoms, output_path = process_image(image_path)

    # ------------------------------------------------------------------
    # 确定原始最小晶胞(缩减晶胞大小)流程：循环弛豫缩减
    # ------------------------------------------------------------------
    cur = Structure.from_file(output_path)
    print(f"起始原子数: {len(cur)}")

    for step in range(1, MAX_NUM_ITER):
        if len(cur) <= MAX_ATOMS_NUM:
            print(f"≤{MAX_ATOMS_NUM}atoms，停止。")
            break
        nxt = shrink_once(cur)
        if nxt is None or len(nxt) >= len(cur):
            print("已无法进一步缩减。")
            break
        print(f"Step {step}:  {len(cur)} → {len(nxt)} atoms")
        cur = nxt

    cur.to(filename="output_final.cif")
    print(f"最终结构 output_final.cif (atoms = {len(cur)})")

    # 调用可视化函数
    # visualize_results(image_path, output_path)