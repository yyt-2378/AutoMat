import json
import matplotlib.pyplot as plt
import os


# 指定存放聚类结果 JSON 文件的路径，请根据实际情况修改路径
json_file = './data_generation/selected_samples_info.json'
putput_distribution = output_point_fig = os.path.join('./data_generation', 'plot_distribution.jpg')

# 读取 JSON 文件，得到一个字典，键为 CIF 文件名，值为聚类标签（类别）
with open(json_file, 'r', encoding='utf-8') as f:
    cluster_data = json.load(f)

# 统计各个类别的数量
counts = {}
for label in cluster_data.values():
    counts[label] = counts.get(label, 0) + 1

# 按类别标签排序
sorted_labels = sorted(counts.keys())
sorted_counts = [counts[label] for label in sorted_labels]

# 绘制柱状图
plt.figure(figsize=(8, 6))
plt.bar([str(label) for label in sorted_labels], sorted_counts, color='skyblue')
plt.xlabel("Cluster Label")
plt.ylabel("Number of CIF Files")
plt.title("Cluster Distribution of CIF Files")
plt.tight_layout()
plt.savefig(output_point_fig)