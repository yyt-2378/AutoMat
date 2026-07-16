#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_best_moe_model_mix.py   – batch 版本

流程
1. 用 **cv2** 读取整幅灰度大图
2. 按 128×128、固定 stride（默认=128）自左→右、自上→下裁剪并 *保存* 小图
3. 再用 **PIL** 逐块读取小图，送入模型做 SR / 去噪
4. 根据滑窗位置拼接回原尺寸
5. 对所有模型计算与标签的 MSE，选出最佳
"""
import os
import shutil
import yaml
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms

# -------------- model ----------------
from preprocess_model.moe import MOEDIVAESR
from preprocess_model.configs.option import args
# --------------------------------------

# ---------- 全局参数 ----------
BATCH_SIZE = 64       # <<< 批量大小，可按显存调
CROP_SIZE = 128
STRIDE = 64
# -----------------------------

# ---------------------------------------


# ---------- 滑窗裁剪（左→右，上→下） ----------
def sliding_window_crop(img: np.ndarray, size, stride):
    """
    img  : H×W uint8
    size : (h, w)
    返回: patches(list[np.ndarray]), positions[(top,left), ...]
    """
    H, W = img.shape[:2]
    th, tw = size
    tops = list(range(0, H - th + 1, stride))
    if (H - th) not in tops:
        tops.append(H - th)
    lefts = list(range(0, W - tw + 1, stride))
    if (W - tw) not in lefts:
        lefts.append(W - tw)

    patches, positions = [], []
    for top in tops:  # 先纵向扫描
        for left in lefts:  # 再横向
            patch = img[top:top + th, left:left + tw].copy()
            patches.append(patch)
            positions.append((top, left))
    return patches, positions


# ---------- large image reconstruction ----------
def _hann_window_2d(size):
    """生成 size×size 的 2-D Hann (余弦) 窗，取值 0~1，中心 1，边缘 0。"""
    one_d = 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))
    w2d = np.outer(one_d, one_d)  # 外积得到 2-D
    return w2d.astype(np.float32)  # float32 以便后续乘


def reconstruct_from_crops(
        crops,  # List[np.ndarray], 每块 uint8 / float
        positions,  # List[(top,left)]
        original_size,  # (W,H)
        crop_size,  # int or (h,w)
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
    out = np.zeros((H, W), np.float32)
    accw = np.zeros((H, W), np.float32)  # 权重累加

    # 1) 准备窗权重
    if window == "none":
        win = np.ones((crop_h, crop_w), np.float32)
    else:  # "hann" / "cosine" 同义
        win = _hann_window_2d(crop_h)

    # 2) 累加
    for crop, (top, left) in zip(crops, positions):
        crop_f = crop.astype(np.float32)
        weighted = crop_f * win

        out[top:top + crop_h, left:left + crop_w] += weighted
        accw[top:top + crop_h, left:left + crop_w] += win

    # 3) 归一化
    eps = 1e-6
    out /= (accw + eps)

    return out.clip(0, 255).astype(np.uint8)


def mae_norm_ssim_score(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    """
    综合指标：mae_norm + (1 - ssim)  → 越小越好
      mae_norm = mae / 255          (0~1)
      ssim     = 0~1                越高越好
    """
    # --- L1 / MAE ---
    mae = mean_absolute_error(pred_u8.ravel().astype(np.float32),
                              gt_u8.ravel().astype(np.float32))
    mae_norm = mae / 255.0  # 归一化到 0~1

    # --- SSIM ---
    ssim = compare_ssim(pred_u8, gt_u8, data_range=255)

    return mae_norm + (1.0 - ssim)


# ---------- 单幅大图推理（批量 patch） ----------
def inference_large_image_cv2_pil(img_path: str,
                                  weight_path: str,
                                #   out_dir: Path,
                                  crop_size=CROP_SIZE,
                                  stride=STRIDE,
                                  batch_size=BATCH_SIZE,
                                  device="cpu"):

    # ---- 加载模型 ----
    with open(args.filename, "r") as f:
        cfg = yaml.safe_load(f)
    vae_args = cfg["model_params"]

    model = MOEDIVAESR(args, vae_args, gating_weights=None)
    ckpt = torch.load(weight_path, map_location=device, weights_only=True)
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

    # 变换
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # >>>> 批量推理 <<<<
    tensors = [tfm(Image.fromarray(p)).unsqueeze(0) for p in patches_np]
    tensors = torch.cat(tensors)                 # (N,1,64,64)

    recon_patches = []
    # os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            end   = min(start+batch_size, len(tensors))
            lr_b  = tensors[start:end].to(device)
            zero  = torch.zeros_like(lr_b).to(device)
            hr_lbl = torch.zeros((lr_b.size(0), 1, crop_size, crop_size),
                                 device=device)
            # forward
            _, sr, _ = model.forward(lr_b, zero, hr_lbl)[1:]
            sr = torch.nan_to_num(sr, nan=1, posinf=1, neginf=0.01).clamp_min_(0.01)

            # loop over batch outputs保持顺序
            for k in range(sr.size(0)):
                arr = sr[k].permute(1,2,0).cpu().numpy().squeeze()
                arr = (arr-arr.min())/(arr.max()-arr.min()+1e-6)
                arr = (arr*255).astype(np.uint8)
                recon_patches.append(arr)

                # 可选：保存 patch 结果
                idx = start + k + 1
                # Image.fromarray(arr).save(out_dir / f"SR_recon_{idx:05}.png")

    # ---- 删除临时目录（无小图输出） ----
    # 若想保留，请注释掉
    # shutil.rmtree(patch_dir, ignore_errors=True)

    return reconstruct_from_crops(recon_patches,
                                  positions,
                                  big.shape[::-1],
                                  crop_size)


# ---------- 多模型评估 ----------
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
                print("缺少标签:", lbl_p);
                continue

            out_dir = base_out / w_name / stem
            recon = inference_large_image_cv2_pil(
                img_path=img_p,
                weight_path=w,
                out_dir=out_dir,
                device=device
            )
            save_fp = base_out / w_name / stem / "reconstructed_large.png"
            cv2.imwrite(str(save_fp), recon)  # recon 是 uint8 numpy(H×W)
            label_np = cv2.imread(str(lbl_p), cv2.IMREAD_GRAYSCALE)
            score = mae_norm_ssim_score(recon, label_np)
            if score < best_score:
                best_score, best_w, best_recon = score, w, recon
        print(f"{Path(w).name} 评估完毕，当前最佳 score={best_score:.4f}")

    print("\n最终最佳模型:", best_w, "MSE=", best_score)
    return best_w, best_recon


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_dir = Path("./MOE_model_weights")
    img_dir = Path("./data_generation/large_data_test/img")
    label_dir = Path("./data_generation/large_data_test/label")

    weight_paths = sorted(weight_dir.glob("*.ckpt"))
    img_paths = sorted(img_dir.glob("*.[pj][pn][gf]*"))

    best_w, best_img = find_best_model(img_paths, weight_paths, label_dir, device)
    if best_img is not None:
        fp = Path(
            "./"
            "baseline/define_best_model/best_reconstructed_image.png"
        )
        cv2.imwrite(str(fp), best_img)
        print("最佳重建图保存到", fp)
