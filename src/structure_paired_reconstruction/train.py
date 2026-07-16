#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本用于对给定数据集进行训练，比较不同预训练模型（resnet18, resnet50, resnet101）在多任务分类上的表现。
每个样本包含两个标签：6个子类别和2个父类别。
数据集 CSV 文件要求包含三列：image_path, child_label, parent_label。
同时，为了适应灰度图输入，预训练模型的第一层卷积被修改为接受 1 通道输入。
最终训练过程会保存 loss/accuracy 曲线，并保存验证集上表现最优的模型权重。
"""

import os
import copy
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image


# -------------------------------
# 1. 数据集定义（CSV文件中要求包含 image_path, child_label, parent_label）
# -------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): CSV 文件路径，文件中应包含 image_path, child_label, parent_label 三列
            transform (callable, optional): 对图像进行预处理的函数
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 获取图像路径和标签
        img_path = self.data_frame.iloc[idx]['image_path']
        # 子类别和父类别标签需为整数类型
        child_label = int(self.data_frame.iloc[idx]['child_label'])
        parent_label = int(self.data_frame.iloc[idx]['parent_label'])
        # 打开图像并转换为灰度图（L模式）
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, (child_label, parent_label)


# -------------------------------
# 2. 多任务模型定义
# -------------------------------
class MultiTaskResNet(nn.Module):
    def __init__(self, base_model, num_child=6, num_parent=2):
        """
        参数：
          base_model: 预训练 ResNet 模型（已修改 conv1 以适应灰度图输入）
          num_child: 子类别数（例如6）
          num_parent: 父类别数（例如2）
        """
        super(MultiTaskResNet, self).__init__()
        # 使用除最后全连接层之外的所有层作为特征提取器
        self.base = nn.Sequential(*list(base_model.children())[:-1])
        in_features = base_model.fc.in_features
        self.child_fc = nn.Linear(in_features, num_child)
        self.parent_fc = nn.Linear(in_features, num_parent)

    def forward(self, x):
        features = self.base(x)  # 输出形状 (batch, in_features, 1, 1)
        features = torch.flatten(features, 1)
        out_child = self.child_fc(features)
        out_parent = self.parent_fc(features)
        return out_child, out_parent


# -------------------------------
# 3. 模型构建辅助函数（加载预训练模型并转换为多任务模型）
# -------------------------------
def get_model(model_name, input_layer, num_child=6, num_parent=2, pretrained_local_folder=None):
    """
    根据模型名称加载预训练模型，并做如下修改：
      1. 如果指定本地预训练权重目录，则从该目录加载对应的权重，否则使用 torchvision 在线预训练权重。
      2. 修改第一层卷积 conv1 使其支持 1 通道输入（input_layer 控制输入通道数）。
      3. 将模型转为多任务模型，输出两个分支：子类别和父类别。
    """
    # 加载原模型
    if pretrained_local_folder is not None:
        print("使用本地预训练权重目录: {}".format(pretrained_local_folder))
        if model_name == 'resnet18':
            local_path = os.path.join(pretrained_local_folder, 'resnet18.pth')
            base_model = models.resnet18(pretrained=False)
        elif model_name == 'resnet50':
            local_path = os.path.join(pretrained_local_folder, 'resnet50.pth')
            base_model = models.resnet50(pretrained=False)
        elif model_name == 'resnet101':
            local_path = os.path.join(pretrained_local_folder, 'resnet101.pth')
            base_model = models.resnet101(pretrained=False)
        else:
            raise ValueError("不支持的模型名称: {}".format(model_name))
        if os.path.exists(local_path):
            state_dict = torch.load(local_path)
            if 'conv1.weight' in state_dict:
                del state_dict['conv1.weight']
            base_model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError("本地预训练权重未找到: {}".format(local_path))
    else:
        print("未指定本地预训练权重，使用 torchvision 在线预训练权重")
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            base_model = models.resnet101(pretrained=True)
        else:
            raise ValueError("不支持的模型名称: {}".format(model_name))

    # 修改第一层卷积，将输入通道设为 input_layer（例如1表示灰度图）
    old_conv1 = base_model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=input_layer,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False
    )
    if old_conv1.weight is not None:
        new_conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)
    base_model.conv1 = new_conv1

    # 用 MultiTaskResNet 封装为多任务模型
    model = MultiTaskResNet(base_model, num_child, num_parent)
    return model


# -------------------------------
# 4. 训练与验证函数（多任务训练）
# -------------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, model_name, save_dir):
    """
    模型训练函数
    参数：
        model: 待训练模型
        dataloaders: 包含 'train' 和 'val' 的 DataLoader 字典
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮次
        device: 计算设备
        model_name: 模型名称（用于保存）
        save_dir: 模型保存的文件夹
    返回：
      best_model_wts: 在验证集上综合准确率最高的模型权重
      history: 包含训练/验证各项指标的历史记录字典
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # 记录综合准确率（子+父准确率平均）

    # 指标记录字典
    history = {
        'train_loss': [], 'val_loss': [],
        'train_child_acc': [], 'val_child_acc': [],
        'train_parent_acc': [], 'val_parent_acc': [],
        'train_overall_acc': [], 'val_overall_acc': []
    }

    for epoch in range(num_epochs):
        print("模型 {} Epoch {}/{}".format(model_name, epoch + 1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_correct_child = 0
            running_correct_parent = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                # 手动拆分标签，由于 __getitem__ 返回 (child_label, parent_label)
                child_labels, parent_labels = labels
                child_labels = torch.tensor(child_labels, dtype=torch.long, device=device)
                parent_labels = torch.tensor(parent_labels, dtype=torch.long, device=device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_child, outputs_parent = model(inputs)
                    loss_child = criterion(outputs_child, child_labels)
                    loss_parent = criterion(outputs_parent, parent_labels)
                    loss = loss_child + loss_parent

                    _, preds_child = torch.max(outputs_child, 1)
                    _, preds_parent = torch.max(outputs_parent, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                batch_size_actual = inputs.size(0)
                running_loss += loss.item() * batch_size_actual
                running_correct_child += torch.sum(preds_child == child_labels.data).item()
                running_correct_parent += torch.sum(preds_parent == parent_labels.data).item()
                total_samples += batch_size_actual

            epoch_loss = running_loss / total_samples
            epoch_child_acc = running_correct_child / total_samples
            epoch_parent_acc = running_correct_parent / total_samples
            epoch_overall_acc = (epoch_child_acc + epoch_parent_acc) / 2.0

            history[phase + '_loss'].append(epoch_loss)
            history[phase + '_child_acc'].append(epoch_child_acc)
            history[phase + '_parent_acc'].append(epoch_parent_acc)
            history[phase + '_overall_acc'].append(epoch_overall_acc)

            print('{} Loss: {:.4f} Child Acc: {:.4f} Parent Acc: {:.4f} Overall Acc: {:.4f}'
                  .format(phase, epoch_loss, epoch_child_acc, epoch_parent_acc, epoch_overall_acc))
            if phase == 'val' and epoch_overall_acc > best_acc:
                best_acc = epoch_overall_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_save_path = os.path.join(save_dir, "{}_best.pth".format(model_name))
                torch.save(model.state_dict(), model_save_path)
                print("保存更优模型到: {}".format(model_save_path))
        print('-' * 30)

    time_elapsed = time.time() - since
    print("训练完成，用时 {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("验证集最高 Overall Acc: {:.4f}".format(best_acc))
    return best_model_wts, history


# -------------------------------
# 5. 主函数入口
# -------------------------------
def main():
    # 超参数设置
    csv_file = './data_generation/classification_parent/augmented_dataset.csv'  # CSV 文件包含 image_path, child_label, parent_label
    input_layer = 1
    num_child = 6
    num_parent = 2
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.001
    train_val_split = 0.8

    # 本地预训练权重存放文件夹（该文件夹中应存放 resnet18.pth 等，如果不需要可设为 None）
    local_pretrain_path = './structure_recongnition/pretrain_local_path'
    # 模型和结果保存目录
    save_dir = './structure_recongnition/saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plots_dir = './structure_recongnition/plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # 数据预处理（注意：灰度图经过 ToTensor() 后通道数为 1）
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]),
    }

    # 构建数据集
    full_dataset = CustomImageDataset(csv_file, transform=data_transforms['train'])
    dataset_size = len(full_dataset)
    train_size = int(train_val_split * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))
    # 验证集使用验证的 transform
    val_dataset.dataset.transform = data_transforms['val']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # 使用 GPU（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 模型列表（可以扩展多个模型比较）
    model_names = ['resnet18']

    all_histories = {}
    best_acc_dict = {}

    for model_name in model_names:
        print("\n开始训练模型: {}\n".format(model_name))
        # 构建多任务模型（子类别和父类别同时预测）
        model = get_model(model_name, input_layer, num_child, num_parent, local_pretrain_path)
        model = model.to(device)

        # 定义损失函数和优化器（均采用 CrossEntropyLoss）
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        best_model_wts, history = train_model(model, dataloaders, criterion, optimizer,
                                              num_epochs, device, model_name, save_dir)
        all_histories[model_name] = history
        best_acc_dict[model_name] = max(history['val_overall_acc'])

        # 绘制并保存损失曲线
        epochs_range = range(1, num_epochs + 1)
        plt.figure()
        plt.plot(epochs_range, history['train_loss'], label='Train Loss')
        plt.plot(epochs_range, history['val_loss'], label='Val Loss')
        plt.title('{} Loss Curve'.format(model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        loss_plot_path = os.path.join(plots_dir, "{}_loss.png".format(model_name))
        plt.savefig(loss_plot_path)
        plt.close()
        print("保存 {} 损失曲线到: {}".format(model_name, loss_plot_path))

        # 绘制并保存子类别准确率曲线
        plt.figure()
        plt.plot(epochs_range, history['train_child_acc'], label='Train Child Acc')
        plt.plot(epochs_range, history['val_child_acc'], label='Val Child Acc')
        plt.title('{} Child Accuracy Curve'.format(model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        child_acc_plot_path = os.path.join(plots_dir, "{}_child_acc.png".format(model_name))
        plt.savefig(child_acc_plot_path)
        plt.close()
        print("保存 {} 子类别准确率曲线到: {}".format(model_name, child_acc_plot_path))

        # 绘制并保存父类别准确率曲线
        plt.figure()
        plt.plot(epochs_range, history['train_parent_acc'], label='Train Parent Acc')
        plt.plot(epochs_range, history['val_parent_acc'], label='Val Parent Acc')
        plt.title('{} Parent Accuracy Curve'.format(model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        parent_acc_plot_path = os.path.join(plots_dir, "{}_parent_acc.png".format(model_name))
        plt.savefig(parent_acc_plot_path)
        plt.close()
        print("保存 {} 父类别准确率曲线到: {}".format(model_name, parent_acc_plot_path))

    # 比较不同模型的最佳验证综合准确率（子和父平均）
    plt.figure()
    model_list = list(best_acc_dict.keys())
    acc_values = [best_acc_dict[m] for m in model_list]
    plt.bar(model_list, acc_values)
    plt.title("Different ResNet Models Best Overall Val Accuracy")
    plt.xlabel("Model")
    plt.ylabel("Overall Accuracy")
    for i, v in enumerate(acc_values):
        plt.text(i, v + 0.01, "{:.2f}".format(v), ha='center')
    summary_plot_path = os.path.join(plots_dir, "models_comparison.png")
    plt.savefig(summary_plot_path)
    plt.close()
    print("\n各模型最佳验证综合准确率:")
    for m in best_acc_dict:
        print("{}: {:.4f}".format(m, best_acc_dict[m]))
    print("保存模型比较图到: {}".format(summary_plot_path))


if __name__ == '__main__':
    main()
