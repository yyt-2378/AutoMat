import os
import json
import csv

# 输入路径
json_path = "./data_generation/selected_samples_info.json"
augmented_image_folder = "./data_generation/simulate_data/aug_img"

# 输出路径
output_csv = "./structure_recongnition/clustered_augmented_images.csv"

# 读取 JSON 文件
with open(json_path, 'r', encoding='utf-8') as f:
    cluster_data = json.load(f)

# 获取所有增强后的图像文件名
all_augmented_images = [f for f in os.listdir(augmented_image_folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

# 存储结果行（图像路径 + 数字类别标签）
final_records = []

# 用数字索引代替 cluster 名称
for cluster_idx, (cluster_name, cif_list) in enumerate(cluster_data.items()):
    selected_cif_files = cif_list[:10] if len(cif_list) > 10 else cif_list

    for cif_filename in selected_cif_files:
        cif_prefix = os.path.splitext(cif_filename)[0]
        matched_images = [img for img in all_augmented_images if cif_prefix in img]

        for matched_img in matched_images:
            full_img_path = os.path.join(augmented_image_folder, matched_img)
            final_records.append([full_img_path, cluster_idx])

# 写入 CSV 文件
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'cluster_label'])
    writer.writerows(final_records)

print(f"✅ 已成功生成 {output_csv}，共记录 {len(final_records)} 条图像-类别对应关系，类别使用数字编码。")
