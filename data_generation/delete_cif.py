#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本脚本读取一个 JSON 文件，其中包含一个名为“其他”的键，该键对应的值是一个文件名列表。
脚本会遍历列表，对每个文件判断是否存在，如果存在则将其删除，并打印删除状态。
"""

import os
import sys
import json

def delete_files_from_json(json_file, cif_folder):
    """
    读取 JSON 文件，删除“其他”键中列出的所有文件。

    参数：
        json_file: JSON 文件路径

    返回：无
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取 JSON 文件时出错：{e}")
        return

    # 检查 JSON 中是否包含“其他”键
    if "其他" not in data:
        print("JSON 文件中不包含 '其他' 关键字")
        return

    files_to_delete = data["其他"]

    # 遍历文件列表进行删除
    for file in files_to_delete:
        file_path = os.path.join(cif_folder, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"已删除文件：{file_path}")
            except Exception as e:
                print(f"删除文件 {file_path} 时出错：{e}")
        else:
            print(f"文件不存在：{file_path}")

if __name__ == "__main__":
    json_file = './data_generation/classification_result.json'
    cif_folder = './data_generation/selected_cif_files'
    delete_files_from_json(json_file, cif_folder)
