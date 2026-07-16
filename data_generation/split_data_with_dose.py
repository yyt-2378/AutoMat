#!/usr/bin/env python3
import os
import shutil
import argparse

def separate_by_dose(in_folder, out_base_folder, doses):
    """
    根据文件名中包含的 dose 关键字，将文件从 in_folder 分隔复制到 out_base_folder 下对应的子文件夹中。

    参数:
        in_folder (str): 输入文件夹路径。
        out_base_folder (str): 输出文件夹根目录，子文件夹名称为对应的 dose，例如 dose10000。
        doses (list): dose 关键字列表。
    """
    # 遍历输入文件夹中所有文件（可以根据实际情况添加扩展名筛选）
    for fname in os.listdir(in_folder):
        # 这里只处理文件（忽略子目录）
        src_path = os.path.join(in_folder, fname)
        if not os.path.isfile(src_path):
            continue
        # 遍历每个 dose 关键字
        for dose in doses:
            if dose in fname:
                target_folder = os.path.join(out_base_folder, dose)
                os.makedirs(target_folder, exist_ok=True)
                dst_path = os.path.join(target_folder, fname)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {fname} to {target_folder}")
                break  # 如果匹配到一个 dose 后就跳出 dose 检查

def main():
    parser = argparse.ArgumentParser(description="根据文件名中的 dose 关键字分离图像和标签数据")
    parser.add_argument("--img_folder", type=str, default='./data_generation/simulate_data/aug_img', help="输入图像文件夹路径")
    parser.add_argument("--label_folder", type=str, default='./data_generation/simulate_data/aug_label', help="输入标签文件夹路径")
    parser.add_argument("--out_img_folder", type=str, default='./data_generation/split_data/img', help="输出图像的根文件夹路径")
    parser.add_argument("--out_label_folder", type=str, default='./data_generation/split_data/label', help="输出标签的根文件夹路径")
    args = parser.parse_args()

    # 定义 dose 关键字列表
    doses = ["dose10000", "dose20000", "dose30000", "dose40000", "dose50000", "dose60000"]

    print("开始分割图像文件...")
    separate_by_dose(args.img_folder, args.out_img_folder, doses)
    print("图像文件分割完成。")

    print("开始分割标签文件...")
    separate_by_dose(args.label_folder, args.out_label_folder, doses)
    print("标签文件分割完成。")

if __name__ == "__main__":
    main()
