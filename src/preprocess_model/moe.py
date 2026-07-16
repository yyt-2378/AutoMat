import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from .image_preprocess_model import DIVAESR


# -------------------------------
# 多任务模型定义
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
# 模型构建辅助函数（加载预训练模型并转换为多任务模型）
# -------------------------------
def get_model(input_layer, num_child=6, num_parent=2, pretrained_local_folder=None):
    """
    根据模型名称加载预训练模型，并做如下修改：
      1. 如果指定本地预训练权重目录，则从该目录加载对应的权重，否则使用 torchvision 在线预训练权重。
      2. 修改第一层卷积 conv1 使其支持 1 通道输入（input_layer 控制输入通道数）。
      3. 将模型转为多任务模型，输出两个分支：子类别和父类别。
    """
    # 加载原模型
    base_model = resnet18(pretrained=False)

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

    # 如果指定了预训练权重目录，尝试加载训练好的权重
    if pretrained_local_folder is not None:
        print("使用本地gating预训练权重目录: {}".format(pretrained_local_folder))
        checkpoint_file = os.path.join(pretrained_local_folder, 'resnet18_best.pth')
        if os.path.exists(checkpoint_file):
            state_dict = torch.load(checkpoint_file, weights_only=True)
            model.load_state_dict(state_dict)
            print("加载训练好gating network的权重：{}".format(checkpoint_file))
        else:
            print("训练好的权重文件未找到：{}".format(checkpoint_file))

    return model


# -------------------------------
# GatingNetwork 定义
# -------------------------------
class GatingNetwork(nn.Module):
    def __init__(self, num_child=6, num_parent=2, temperature=1.0, noise_std=0.1, w_parent=0.3, w_child=0.7, pretrained_local_folder=None):
        """
        参数：
          temperature: 温度参数，用于 logits 缩放
          noise_std: 训练时添加噪声的标准差
          w_parent: 父类别分支权重（例如 0.3）
          w_child: 子类别分支权重（例如 0.7）
          pretrained_local_folder: 如有本地预训练权重目录，由 get_model 加载
        """
        super(GatingNetwork, self).__init__()
        # 加载多任务 resnet18 模型，输入通道为1（灰度图），输出子类别6类、父类别2类
        self.multi_task_model = get_model(input_layer=1, num_child=6, num_parent=2,
                                          pretrained_local_folder=pretrained_local_folder)
        self.temperature = temperature
        self.noise_std = noise_std
        self.w_parent = w_parent
        self.w_child = w_child

        # 冻结 gating network 的所有参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 得到多任务模型输出：child_logits (B,6), parent_logits (B,2)
        child_logits, parent_logits = self.multi_task_model(x)

        # 温度缩放
        child_logits = child_logits / self.temperature
        parent_logits = parent_logits / self.temperature

        # 训练时加入噪声增加鲁棒性
        if self.training:
            child_logits = child_logits + torch.randn_like(child_logits) * self.noise_std
            parent_logits = parent_logits + torch.randn_like(parent_logits) * self.noise_std

        # 分别归一化得到概率
        child_probs = F.softmax(child_logits, dim=1)  # shape: (B, 6)
        parent_probs = F.softmax(parent_logits, dim=1)  # shape: (B, 2)

        # 分别从父/子分支中选取概率最大的 expert（对于父分支，输出索引范围是[0,1]；对于子分支，输出索引范围是[0,5]）
        parent_val, parent_idx = torch.max(parent_probs, dim=1)  # (B,)
        child_val, child_idx = torch.max(child_probs, dim=1)  # (B,)

        # 将子分支 expert 的索引加偏移，将全局专家索引设置为：
        # 父类别专家为 0 和 1，子类别专家为 2～7
        child_global_idx = child_idx + 2

        # 对两个分支概率加权
        weighted_parent = self.w_parent * parent_val  # 父分支调小一些
        weighted_child = self.w_child * child_val  # 子分支调大一些
        total_weighted = weighted_parent + weighted_child + 1e-8
        balanced_parent = weighted_parent / total_weighted
        balanced_child = weighted_child / total_weighted

        # 组合成 topk_indices 和对应概率（按照要求返回类似于 return topk_indices, probs）
        # topk_indices 第一列为父类别 expert index（0或1），第二列为子类别 expert（加偏移后在 [2,7]）
        topk_indices = torch.stack([parent_idx, child_global_idx], dim=1)
        probs = torch.stack([balanced_parent, balanced_child], dim=1)

        return topk_indices, probs


def tensor_mean_shift(img: torch.Tensor):
    img_max, img_min = img.max(), img.min()
    img_output = (img - img_min) / (img_max - img_min)
    return img_output


class MOEDIVAESR(nn.Module):
    def __init__(self, sr_args, vae_args, gating_weights, num_expert=8, top_k=2, w_parent=0.3, w_child=0.7, temperature=1.0, noise_std=0.01):
        super(MOEDIVAESR, self).__init__()
        self.num_expert = num_expert
        self.top_k = top_k
        self.gating_network = GatingNetwork(num_child=6, num_parent=2, temperature=temperature, noise_std=noise_std, w_parent=w_parent, w_child=w_child, pretrained_local_folder=gating_weights)

        # Create expert models
        self.experts = nn.ModuleList([DIVAESR(sr_args, vae_args) for _ in range(num_expert)])

    def distribute_data_to_experts(self, img_data_tensor, label_data_tensor, hr_label_data_tensor, topk_indices):
        """
        Distributes data points in data_tensor to corresponding experts based on topk_indices.
        Returns:
        tuple: A tuple containing:
            - List of Batch objects: Each Batch object contains data batches for a specific expert.
            - List of lists: Each sub-list contains indices of the original data points assigned to the corresponding expert.
        """
        # Number of experts can be inferred from the unique elements in topk_indices
        num_experts = torch.unique(topk_indices).max().item() + 1
        # Initialize a list of lists to store data for each expert
        expert_img_data_lists = [[] for _ in range(num_experts)]
        expert_label_data_lists = [[] for _ in range(num_experts)]
        expert_hr_label_data_lists = [[] for _ in range(num_experts)]
        # Initialize a list of lists to store indices for each expert
        expert_indices_lists = [[] for _ in range(num_experts)]

        # Iterate over each data point and its corresponding topk indices
        for idx, indices in enumerate(topk_indices):  # idx represents the data point
            # Add the data point to each of the top k expert lists specified by indices
            for index in indices:  # indices can be multiple experts, so distribute data
                expert_img_data_lists[index.item()].append(img_data_tensor[idx])
                expert_label_data_lists[index.item()].append(label_data_tensor[idx])
                expert_hr_label_data_lists[index.item()].append(hr_label_data_tensor[idx])
                expert_indices_lists[index.item()].append(idx)

        # Return the expert data lists and indices lists
        return expert_img_data_lists, expert_label_data_lists, expert_hr_label_data_lists, expert_indices_lists

    def combine_and_concatenate_outputs(self, expert_outputs, expert_indices_lists, expert_probs, original_img, topk_indices):
        """
        根据原始 batch 顺序将各专家输出结合为最终输出。
        对于每个样本 i：
          - gating 网络返回 topk_indices[i] 为一个 2 维向量，其中第一项为父类别 expert 全局索引（0或1），
            第二项为子类别 expert 全局索引（原子分支输出加偏移后的编号，范围 [2,7]）。
          - 同时 gating 网络输出的 expert_probs[i] 为 (p_prob, c_prob)。
        如果专家输出为 tensor，则：
            final_output[i] = p_prob * (父专家输出) + c_prob * (子专家输出)
        如果专家输出为 list，则对每个组件 j（例如 j 从 0 到 L-1）进行：
            final_output_component[j][i] = p_prob * (父专家输出[j]) + c_prob * (子专家输出[j])
        若样本 i 在对应专家分支的索引列表中找不到，则直接 raise error。
        
        参数：
          expert_outputs: 长度为 num_expert 的列表，每个元素可能是 tensor（形状 (N, C, H, W)）
                          或者是 list，其中每个元素为 tensor（形状 (N, C, H, W)）。
          expert_indices_lists: 长度为 num_expert 的列表，每个元素是列表，记录该 expert 接收到的原始 batch 索引。
          expert_probs: gating 网络平衡后的概率，形状 (B, 2)，第一列对应父分支，第二列对应子分支。
          original_img: 原始输入图，用于获取 batch 大小和设备信息。
          topk_indices: gating 网络返回的专家索引，形状 (B, 2)，第一列为父 expert（全局索引 0或1），
                        第二列为子 expert（全局索引，已加偏移，应在 [2,7]）。
        返回：
          如果专家输出为 tensor，则返回 tensor，形状为 (B, C, H, W)；
          如果专家输出为 list，则返回 list，其中每个元素的形状均为 (B, C, H, W)。
        """
        B = original_img.size(0)
        # 选择第一个非空的专家输出以确定输出类型
        out_sample = None
        for out in expert_outputs:
            if out is not None:
                out_sample = out
                break
        if out_sample is None:
            raise ValueError("没有任何专家产生输出。")

        if isinstance(out_sample, torch.Tensor):
            out_shape = out_sample.shape[1:]  # (C, H, W)
            combined_outputs = torch.zeros(B, *out_shape, device=original_img.device)
            for i in range(B):
                p_expert = topk_indices[i, 0].item()  # 父类别 expert（0或1）
                c_expert = topk_indices[i, 1].item()  # 子类别 expert（范围 [2,7]）
                p_prob = expert_probs[i, 0].item()
                c_prob = expert_probs[i, 1].item()
                if i not in expert_indices_lists[p_expert]:
                    raise ValueError(f"Sample {i}未在父专家 {p_expert} 的索引列表中。")
                pos_p = expert_indices_lists[p_expert].index(i)
                parent_output = expert_outputs[p_expert][pos_p]
                if i not in expert_indices_lists[c_expert]:
                    raise ValueError(f"Sample {i}未在子专家 {c_expert} 的索引列表中。")
                pos_c = expert_indices_lists[c_expert].index(i)
                child_output = expert_outputs[c_expert][pos_c]
                combined_outputs[i] = p_prob * parent_output + c_prob * child_output
            return combined_outputs
        else:
            L = len(out_sample)  # 例如有 L 个组件
            out_shapes = [None] * L
            for expert_out in expert_outputs:
                if expert_out is not None:
                    for j in range(L):
                        if out_shapes[j] is None and len(expert_out) > j and expert_out[j] is not None and expert_out[j].nelement() > 0:
                            out_shapes[j] = expert_out[j].shape[1:]
                    if all(shape is not None for shape in out_shapes):
                        break
            if any(shape is None for shape in out_shapes):
                raise ValueError("无法确定所有输出组件的形状。")
            combined_components = [torch.zeros(B, *out_shapes[j], device=original_img.device) for j in range(L)]
            for i in range(B):
                p_expert = topk_indices[i, 0].item()
                c_expert = topk_indices[i, 1].item()
                p_prob = expert_probs[i, 0].item()
                c_prob = expert_probs[i, 1].item()
                if i not in expert_indices_lists[p_expert]:
                    raise ValueError(f"Sample {i}未在父专家 {p_expert} 的索引列表中。")
                pos_p = expert_indices_lists[p_expert].index(i)
                parent_components = [expert_outputs[p_expert][j][pos_p] for j in range(L)]
                if i not in expert_indices_lists[c_expert]:
                    raise ValueError(f"Sample {i}未在子专家 {c_expert} 的索引列表中。")
                pos_c = expert_indices_lists[c_expert].index(i)
                child_components = [expert_outputs[c_expert][j][pos_c] for j in range(L)]
                for j in range(L):
                    combined_components[j][i] = p_prob * parent_components[j] + c_prob * child_components[j]
            return combined_components

    def forward(self, img, label, hr_label):
        # Step 1: Use the gating network to select top_k experts and get their probabilities
        topk_indices, expert_probs = self.gating_network(img)  # Select top_k experts and all expert probs
        # log gating info
        # if not hasattr(self, 'gating_log'):
        #     self.gating_log = []          # 每个元素: {"indices":[...], "probs":[...]}
        # self.gating_log.append({
        #      "indices": topk_indices.detach().cpu().tolist(),
        #      "probs":   expert_probs.detach().cpu().tolist()})
        
        # # saved json files
        # log_file = getattr(self, "gating_log_path", "gating_info.jsonl")
        # os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        # with open(log_file, "a") as f:
        #     f.write(json.dumps(self.gating_log[-1]) + "\n")

        # Step 2: Distribute the input data to the selected experts
        expert_img_data_lists, expert_label_data_lists, expert_hr_label_data_lists, expert_indices_lists = self.distribute_data_to_experts(img, label, hr_label, topk_indices)

        # 确保每个批次都被发送到正确的设备，同时跳过空批次
        processed_expert_batches_imgs = [
            torch.stack(expert_batch, dim=0).to(img.device) if len(expert_batch) > 0 else 0
            for expert_batch in expert_img_data_lists
        ]
        processed_expert_batches_labels = [
            torch.stack(expert_batch, dim=0).to(img.device) if len(expert_batch) > 0 else 0
            for expert_batch in expert_label_data_lists
        ]
        processed_expert_batches_hr_labels = [
            torch.stack(expert_batch, dim=0).to(img.device) if len(expert_batch) > 0 else 0
            for expert_batch in expert_hr_label_data_lists
        ]

        # Step 3: Compute outputs for each expert
        expert_outputs_sr = []
        expert_outputs_noise = []
        expert_outputs_vae = []

        for idx, expert_batch in enumerate(processed_expert_batches_imgs):
            if expert_batch is 0:
                expert_outputs_sr.append(None)
                expert_outputs_vae.append(None)
                expert_outputs_noise.append(None)
            else:
                expert_batch_labels = processed_expert_batches_labels[idx]
                expert_batch_hr_label = processed_expert_batches_hr_labels[idx]
                vae_output_noise, vae_output, sr_output, _ = self.experts[idx](
                    expert_batch, expert_batch_labels, expert_batch_hr_label
                )
                expert_outputs_sr.append(sr_output)
                expert_outputs_vae.append(vae_output)
                expert_outputs_noise.append(vae_output_noise)

        # Step 4: Combine expert outputs according to original batch indices, applying weights
        combined_sr_output = self.combine_and_concatenate_outputs(
            expert_outputs_sr, expert_indices_lists, expert_probs, img, topk_indices
        )
        combined_vae_output = self.combine_and_concatenate_outputs(
            expert_outputs_vae, expert_indices_lists, expert_probs, img, topk_indices
        )
        combined_noise_output = self.combine_and_concatenate_outputs(
            expert_outputs_noise, expert_indices_lists, expert_probs, img, topk_indices
        )

        # Step 5: Return the output as [vae_output_noise, vae_output, final_sr_output, hr_label]
        return [combined_noise_output, combined_vae_output, combined_sr_output, hr_label]
