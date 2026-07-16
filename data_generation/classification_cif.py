import os
import numpy as np
import json
from pymatgen.core.structure import Structure
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------
# 参数设置
# ---------------------------
# 输入扩胞后 CIF 文件所在文件夹路径（请替换为实际路径）
input_folder = './data_generation/supercell_cifs'
# 输出聚类结果的 JSON 文件路径
output_json = './data_generation/cif_clusters.json'
# 2D histogram 的网格分辨率，生成 grid_size x grid_size 的图像
grid_size = 50  # 可根据需要调整

# ---------------------------
# 提取每个 CIF 文件在 xy 平面的投影（生成 2D histogram 特征向量）
# ---------------------------
feature_list = []  # 用于存储每个 CIF 的特征向量
filenames = []     # 存储对应的文件名

for file in os.listdir(input_folder):
    if file.lower().endswith('.cif'):
        file_path = os.path.join(input_folder, file)
        try:
            # 读取 CIF 文件，生成结构对象
            structure = Structure.from_file(file_path)
            # 获取原子笛卡尔坐标（单位：Å）
            coords = structure.cart_coords  # shape: (N_atoms, 3)
            # 投影到 xy 平面（忽略 z 轴信息）
            xy = coords[:, :2]
            
            # 根据 xy 坐标确定边界，并稍微加一点 margin
            x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
            y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
            margin_x = 0.05 * (x_max - x_min)
            margin_y = 0.05 * (y_max - y_min)
            x_min -= margin_x
            x_max += margin_x
            y_min -= margin_y
            y_max += margin_y
            
            # 利用 numpy.histogram2d 生成二维直方图，作为该结构在 xy 平面的 pattern 表示
            H, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=grid_size, range=[[x_min, x_max], [y_min, y_max]])
            # 将二维 histogram 展平为一维特征向量
            feature_vector = H.flatten()
            feature_list.append(feature_vector)
            filenames.append(file)
            print(f'处理文件{file}')
        except Exception as e:
            print(f"处理文件 {file} 时出错：{e}")

# 将所有特征向量构成数组，shape 为 (N_files, grid_size*grid_size)
features = np.array(feature_list)

# ---------------------------
# 特征归一化和降维
# ---------------------------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 利用 PCA 降到 10 维（可以根据需要调整维数）
pca = PCA(n_components=10, random_state=42)
features_pca = pca.fit_transform(features_scaled)

# ---------------------------
# 聚类：利用 KMeans 将结构分为 8 类
# ---------------------------
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_pca)

# ---------------------------
# 保存聚类结果：生成字典，键为原始文件名，值为聚类标签
# ---------------------------
results = {}
for fname, cluster in zip(filenames, clusters):
    results[fname] = int(cluster)

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"聚类结果已保存至：{output_json}")

# ---------------------------
# 可选：绘制 PCA 降维后前两个主成分的散点图展示聚类情况
# ---------------------------
output_point_fig = os.path.join('./data_generation', 'plot_point.jpg')
plt.figure(figsize=(8, 6))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='tab10', s=20)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("CIF 结构投影降维及聚类结果")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(output_point_fig)
