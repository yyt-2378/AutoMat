#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本用于加载训练好的分类模型（支持灰度图输入），对指定文件夹中的图片进行单张推理，
并将每张图片的预测结果保存到 CSV 文件中。

请根据实际情况修改以下参数：
  - image_folder: 测试图片所在的文件夹路径
  - checkpoint_path: 模型检查点文件路径（保存训练得到的权重）
  - output_csv: 存放推理结果的 CSV 文件路径
  - model_name: 使用的模型名称，可选 'resnet18'、'resnet50'、'resnet101'
  - num_classes: 分类任务类别数（例如6）
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd


# -------------------------------
# 1. 模型构建辅助函数
# -------------------------------
def get_model(model_name, num_classes=6):
    """
    根据模型名称加载预训练模型，并修改：
      1. 修改第一层卷积 conv1 为支持 1 通道输入（灰度图），
         利用原 conv1 权重在通道维度均值初始化新 conv1；
      2. 修改最后全连接层输出为 num_classes。
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=False)
    else:
        raise ValueError("不支持的模型名称: {}".format(model_name))
    
    # 修改第一层卷积，原模型默认输入是3通道；这里调整为1通道
    old_conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=1,  # 灰度图为1通道
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False
    )
    # 使用原 conv1 权重各通道均值对新卷积层进行初始化
    if old_conv1.weight is not None:
        new_conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)
    model.conv1 = new_conv1

    # 修改最后全连接层，使得输出类别数为 num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

# -------------------------------
# 2. 主函数：单张图片推理并保存结果
# -------------------------------
def main():
    # 参数设置，根据自己的实际路径修改
    image_folder = "./data_generation/simulate_data/img"  # 测试图片存放的文件夹
    checkpoint_path = "./structure_recongnition/saved_models/resnet18_best.pth"
    output_csv = "./structure_recongnition/model_test_results.csv"
    model_name = 'resnet18'      # 如有需要，也可选择 resnet50 或 resnet101
    num_classes = 6              # 分类任务类别数

    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 构建模型并加载训练检查点
    model = get_model(model_name, num_classes=num_classes)
    if os.path.exists(checkpoint_path):
        print("加载模型检查点：{}".format(checkpoint_path))
        # 如果保存时保存的是 state_dict() 或者是包含多个key的字典，请根据实际情况调整加载方式
        state_dict = torch.load(checkpoint_path, map_location=device)
        # 如训练时保存的是一个字典并且模型参数存储在 'model' 字段中，则需要使用 state_dict['model']
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError("检查点文件未找到：{}".format(checkpoint_path))
    model = model.to(device)
    model.eval()

    # 定义图片预处理（与训练/验证时保持一致）
    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # 遍历指定文件夹中所有图片，进行推理
    results = []  # 用于存储结果
    for file in os.listdir(image_folder):
        # 根据文件扩展名判断是否为图片文件
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(image_folder, file)
            print("处理图片：", img_path)
            try:
                # 打开图片，并转换为灰度图
                image = Image.open(img_path).convert("L")
            except Exception as e:
                print("读取图片 {} 出现错误: {}".format(img_path, e))
                continue

            # 预处理并加入批次维度
            image_tensor = test_transforms(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                outputs = model(image_tensor)
                # torch.max 返回 (values, indices)，这里 indices 为预测类别
                _, pred = torch.max(outputs, 1)
                predicted_label = int(pred.item())

            print("预测类别：", predicted_label)
            results.append({
                "image_path": img_path,
                "predicted_label": predicted_label
            })
    
    # 保存推理结果到 CSV 文件
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print("推理结果已保存到：", output_csv)

if __name__ == '__main__':
    main()
