#!/usr/bin/env python3
import os
import numpy as np
import json
from pymatgen.core.structure import Structure
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import glob

# ---------------------------
# 参数设置
# ---------------------------
# 输入扩胞后 CIF 文件所在文件夹路径（请替换为实际路径）
input_folder = './data_generation/modified_supercell_orthonalize_selected_cifs'
# 输出聚类结果的 JSON 文件路径
output_json = './data_generation/new_cif_clusters.json'
# 记录筛选20%样本所属类别信息的 JSON 文件路径
selected_info_json = './data_generation/selected_samples_info.json'
# 2D histogram 的网格分辨率，生成 grid_size x grid_size 的图像
grid_size = 50  # 可根据需要调整
# 选出样本存储路径
output_folder = './data_generation/selected_samples'

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

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
            print(f'处理文件 {file}')
        except Exception as e:
            print(f"处理文件 {file} 时出错：{e}")

# 将所有特征向量构成数组，shape 为 (N_files, grid_size*grid_size)
features = np.array(feature_list)

# ---------------------------
# 特征归一化和降维
# ---------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

from sklearn.decomposition import PCA
# 利用 PCA 降到 7 维（可以根据需要调整维数）
pca = PCA(n_components=10, random_state=42)
features_pca = pca.fit_transform(features_scaled)

# ---------------------------
# 自动选择最佳聚类数：使用肘部法则（Elbow Method）和轮廓系数（Silhouette Score）
# ---------------------------
def find_best_n_clusters(features_pca):
    inertia = []
    silhouette_scores = []
    max_clusters = 8  # 设置最大聚类数
    for n_clusters in range(3, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features_pca)
        inertia.append(kmeans.inertia_)  # 计算每个聚类数的误差平方和
        silhouette_scores.append(silhouette_score(features_pca, kmeans.labels_))  # 计算轮廓系数
    
    # 绘制肘部法则图
    plt.figure(figsize=(8, 6))
    plt.plot(range(3, max_clusters + 1), inertia, marker='o', label='Inertia (SSE)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('./data_generation/elbow_method.jpg')
    plt.close()

    # 绘制轮廓系数图
    plt.figure(figsize=(8, 6))
    plt.plot(range(3, max_clusters + 1), silhouette_scores, marker='o', color='b', label='Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.savefig('./data_generation/silhouette_score.jpg')
    plt.close()

    # 选择轮廓系数最高的聚类数
    best_n_clusters = np.argmax(silhouette_scores) + 3
    return best_n_clusters

# 获取最佳聚类数
n_clusters = find_best_n_clusters(features_pca)
print(f"最佳聚类数：{n_clusters}")

# ---------------------------
# 聚类：利用 KMeans 将结构分为最佳数量的类
# ---------------------------
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
# 选择每个类别中距离聚类中心最近的20%的样本，并保存它们，同时记录这些样本及其所属聚类
# ---------------------------
selected_info = {}  # 用于记录选中的样本：键为聚类标签，值为文件名列表

for cluster_id in range(n_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    cluster_center = kmeans.cluster_centers_[cluster_id]
    
    # 计算每个样本到该聚类中心的欧氏距离
    distances = np.linalg.norm(features_pca[cluster_indices] - cluster_center, axis=1)
    
    # 按照距离排序，选择前20%的样本
    num_samples = len(cluster_indices)
    num_selected = max(1, int(0.2 * num_samples))  # 至少选取一个样本
    sorted_indices = np.argsort(distances)
    selected_cluster_indices = cluster_indices[sorted_indices[:num_selected]]
    
    # 将选中的样本记录到字典中
    selected_files = [filenames[idx] for idx in selected_cluster_indices]
    selected_info[f"Cluster_{cluster_id}"] = selected_files

    # 将选中的样本移动到输出文件夹
    for idx in selected_cluster_indices:
        file_name = filenames[idx]
        original_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.rename(original_path, output_path)  # 移动文件到新目录

print(f"每个类别中距离聚类中心最近的20%样本已保存至：{output_folder}")

# 保存选中样本的记录信息
with open(selected_info_json, "w", encoding="utf-8") as f:
    json.dump(selected_info, f, indent=4, ensure_ascii=False)

print(f"选中的样本信息已保存至：{selected_info_json}")

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
plt.close()
print(f"PCA降维及聚类结果的散点图已保存至：{output_point_fig}")
