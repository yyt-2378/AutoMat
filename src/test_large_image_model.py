#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_best_moe_model_mix.py

流程  
1. 用 **cv2** 读取整幅灰度大图  
2. 按 128×128、固定 stride（默认=128）自左→右、自上→下裁剪并 *保存* 小图  
3. 再用 **PIL** 逐块读取小图，送入模型做 SR / 去噪  
4. 根据滑窗位置拼接回原尺寸  
5. 对所有模型计算与标签的 MSE，选出最佳
"""
import os
from pathlib import Path
import yaml
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms

from collections import defaultdict
# 自适应直方图均衡（CLAHE）
from functools import lru_cache

# -------------- model ----------------
from preprocess_model.moe import MOEDIVAESR
from preprocess_model.configs.option import args
# ---------------------------------------


@lru_cache(maxsize=1)
def _get_clahe(clip=3.0, grid=8):
    """返回一个共享的 CLAHE 实例；clipLimit 可按噪声大小微调"""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))


# ---------- 滑窗裁剪（左→右，上→下） ----------
def sliding_window_crop(img: np.ndarray, size, stride):
    """
    img  : H×W uint8
    size : (h, w)
    返回: patches(list[np.ndarray]), positions[(top,left), ...]
    """
    H, W = img.shape[:2]
    th, tw = size
    tops  = list(range(0, H - th + 1, stride))
    if (H - th) not in tops:
        tops.append(H - th)
    lefts = list(range(0, W - tw + 1, stride))
    if (W - tw) not in lefts:
        lefts.append(W - tw)

    patches, positions = [], []
    for top in tops:                        # 先纵向扫描
        for left in lefts:                  # 再横向
            patch = img[top:top+th, left:left+tw].copy()
            patches.append(patch)
            positions.append((top, left))
    return patches, positions


# ---------- large image reconstruction ----------
def _hann_window_2d(size):
    """生成 size×size 的 2-D Hann (余弦) 窗，取值 0~1，中心 1，边缘 0。"""
    one_d = 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))
    w2d = np.outer(one_d, one_d)          # 外积得到 2-D
    return w2d.astype(np.float32)         # float32 以便后续乘


def reconstruct_from_crops(
    crops,                # List[np.ndarray], 每块 uint8 / float
    positions,            # List[(top,left)]
    original_size,        # (W,H)
    crop_size,            # int or (h,w)
    window: str = "hann"  # "hann" | "cosine" | "none"
):
    """
    将裁剪块按 positions 拼回整图，并用平滑窗做重叠融合。

    返回: uint8 H×W
    """
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    W, H = original_size
    out  = np.zeros((H, W), np.float32)
    accw = np.zeros((H, W), np.float32)   # 权重累加

    # 1) 准备窗权重
    if window == "none":
        win = np.ones((crop_h, crop_w), np.float32)
    else:  # "hann" / "cosine" 同义
        win = _hann_window_2d(crop_h)

    # 2) 累加
    for crop, (top, left) in zip(crops, positions):
        crop_f = crop.astype(np.float32)
        weighted = crop_f * win

        out[top:top+crop_h, left:left+crop_w] += weighted
        accw[top:top+crop_h, left:left+crop_w] += win

    # 3) 归一化
    eps = 1e-6
    out /= (accw + eps)

    return out.clip(0, 255).astype(np.uint8)



def mse_np(a, b):
    return mean_squared_error(a.astype(np.float32).ravel(),
                              b.astype(np.float32).ravel())


def mae_norm_ssim_score(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    """
    综合指标：mae_norm + (1 - ssim)  → 越小越好
      mae_norm = mae / 255          (0~1)
      ssim     = 0~1                越高越好
    """
    # --- L1 / MAE ---
    mae = mean_absolute_error(pred_u8.ravel().astype(np.float32),
                              gt_u8.ravel().astype(np.float32))
    mae_norm = mae / 255.0          # 归一化到 0~1

    # --- SSIM ---
    ssim = compare_ssim(pred_u8, gt_u8, data_range=255)

    return mae_norm + (1.0 - ssim)


# ---------- 单图推理 ----------
def inference_large_image_cv2_pil(
        img_path: str,
        weight_path: str,
        out_dir: str,
        crop_size=128,
        stride=128,
        device="cpu"):
    # ---- cfg & model ----
    with open(args.filename, "r") as f:
        cfg = yaml.safe_load(f)
    vae_args = cfg["model_params"]

    model = MOEDIVAESR(args, vae_args, gating_weights=None)
    ckpt  = torch.load(weight_path, map_location=device, weights_only=True)
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        if key.startswith("model."):  # Only modify keys that start with "model."
            new_key = key[len("model."):]  # Remove the "model." prefix
        
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    # ---- 读大图并裁剪 ----
    big = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    patches_np, positions = sliding_window_crop(big, (crop_size, crop_size), stride)

    # 保存小图（避免不同库差异）
    patch_dir = out_dir / "patches"
    if patch_dir.exists():
        shutil.rmtree(patch_dir)
    patch_dir.mkdir(parents=True)

    for i, p in enumerate(patches_np, 1):
        cv2.imwrite(str(patch_dir / f"{i:05}.png"), p)

    # ---- PIL 读取小图送模型 ----
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    recon_patches = []
    with torch.no_grad():
        for i in range(1, len(patches_np)+1):
            img_pil = Image.open(patch_dir / f"{i:05}.png")
            lr_image_tensor = tfm(img_pil).unsqueeze(0).to(device)
            _ = torch.zeros_like(lr_image_tensor).to(device)
            hr_label_tensor = torch.zeros([1, 1, 128, 128]).to(device)
            results = model.forward(lr_image_tensor, _, hr_label_tensor)
            vae_output, sr_output, output_label = results[1], results[2], results[3]
            # 取实部，NaN→0.05，+∞→1，-∞→0.05，再把所有 <0.05 的值截到 0.05
            sr_output = torch.nan_to_num(sr_output, nan=1, posinf=1.0, neginf=0.01).clamp_min_(0.01)
            
            # Save SR reconstructed image
            SR_reconstructed_image = sr_output.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze()
            SR_reconstructed_image = (SR_reconstructed_image - SR_reconstructed_image.min()) / (
                    SR_reconstructed_image.max() - SR_reconstructed_image.min())
            SR_reconstructed_image = (SR_reconstructed_image * 255).astype("uint8")
            SR_save_path = os.path.join(out_dir, f"SR_reconstructed_img_{i}.png")
            Image.fromarray(SR_reconstructed_image, mode='L').save(SR_save_path)
            recon_patches.append(SR_reconstructed_image)
            # # >>> 新增：对每个 SR patch 做 CLAHE 以统一亮度
            # clahe = _get_clahe()                 # 见下方缓存函数
            # SR_eq = clahe.apply(SR_reconstructed_image)
            # recon_patches[-1] = SR_eq            # 用均衡后的 patch 替换原 patch

    recon_full = reconstruct_from_crops(recon_patches, positions, big.shape[::-1], crop_size)
    
    # # 新增：再做一次整体 CLAHE，保证全图一致
    # recon_full_eq = _get_clahe().apply(recon_full)
    
    # === 新增：流程结束后删除临时 crop 目录 =========================
    shutil.rmtree(patch_dir, ignore_errors=True)
    
    return recon_full


# ---------- evaluate different model ----------
def find_best_model(img_paths, weight_paths, label_dir, device):
    best_score, best_w, best_recon = float("inf"), None, None
    base_out = Path(
        "./baseline/define_best_model"
    )
    base_out.mkdir(parents=True, exist_ok=True)

    for w in weight_paths:
        w_name = Path(w).stem
        for img_p in img_paths:
            stem = Path(img_p).stem
            lbl_p = label_dir / f"{stem}.png"
            if not lbl_p.exists():
                print("缺少标签:", lbl_p); continue

            recon = inference_large_image_cv2_pil(
                img_path=img_p,
                weight_path=w,
                out_dir=base_out / w_name / stem,
                crop_size=128, stride=64,
                device=device
            )
            save_fp = base_out / w_name / stem / "reconstructed_large.png"
            cv2.imwrite(str(save_fp), recon)          # recon 是 uint8 numpy(H×W)
            label_np = cv2.imread(str(lbl_p), cv2.IMREAD_GRAYSCALE)
            score = mae_norm_ssim_score(recon, label_np)
            if score < best_score:
                best_score, best_w, best_recon = score, w, recon
        print(f"{Path(w).name} 评估完毕，当前最佳 score={best_score:.4f}")

    print("\n最终最佳模型:", best_w, "MSE=", best_score)
    return best_w, best_recon


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_paths = sorted(Path("./MOE_model_weights").glob("*.ckpt"))

    img_paths = sorted(Path("./data_generation/large_data_test/img").glob("*.[pj][pn][gf]*"))  # jpg/png/tif/tiff

    label_dir = Path("./data_generation/large_data_test/label")

    best_w, best_np = find_best_model(img_paths, weight_paths, label_dir, device)

    if best_np is not None:
        save_path = Path(
            "./baseline/define_best_model/best_reconstructed_image.png"
        )
        cv2.imwrite(str(save_path), best_np)
        print("最佳重建图保存到", save_path)
