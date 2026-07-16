#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_labels.py

批量将去噪后的 STEM 图像与一系列 label 图像进行匹配，
并将每张输入图像的 top-1 最相似且元素匹配的 label 复制到指定目录。

- 对每个去噪输入图像：
  1) 提取原子点质心
  2) 计算径向分布函数（RDF）直方图作为全局描述符
  3) 与所有 label 图像做同样处理（但跳过形态学去噪）
  4) 在特征空间中找出 top-k 最近的 label
  5) 从 top-k 中筛选出元素集与输入图像相同的那些，若有多个则取距离最小者
  6) 将筛出的 top-1 复制到输出目录
"""
import json
import argparse
import shutil
import csv
import re
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from sklearn.metrics import pairwise_distances


def detect_atoms(img_uint8: np.ndarray,
                 min_area: int = 5,
                 do_morph: bool = True):
    """
    检测原子斑点并返回质心坐标列表 (x, y)。
    - Otsu 阈值分割
    - 可选：开运算去噪
    - 连通域分析并过滤小面积
    """
    _, binary = cv2.threshold(
        img_uint8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if do_morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    pts = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            pts.append(tuple(centroids[i]))
    return np.array(pts, dtype=float)


def compute_rdf_descriptor(pts: np.ndarray,
                           max_dist: float = None,
                           bin_width: float = 5.0):
    """
    计算点阵的径向分布函数(RDF)直方图并归一化
    """
    if pts.shape[0] < 2:
        nbins = int(np.ceil((max_dist or 0) / bin_width))
        return np.zeros(nbins, dtype=float)
    dists = pairwise_distances(pts, pts, metric="euclidean")
    iu = np.triu_indices_from(dists, k=1)
    d = dists[iu]
    r_max = max_dist or d.max()
    nbins = int(np.ceil(r_max / bin_width))
    hist, _ = np.histogram(d, bins=nbins, range=(0, nbins * bin_width))
    return hist.astype(float) / hist.sum() if hist.sum() > 0 else hist


def make_descriptor(path: Path,
                    min_area: int,
                    max_dist: float,
                    bin_width: float,
                    do_morph: bool):
    """
    从图像文件生成 RDF 描述符
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image {path}")
    pts = detect_atoms(img, min_area=min_area, do_morph=do_morph)
    return compute_rdf_descriptor(pts, max_dist=max_dist, bin_width=bin_width)


def match_one(input_path: Path,
              labels_dir: Path,
              topk: int,
              min_area: int,
              max_dist: float,
              bin_width: float):
    """
    对单张输入图像做匹配，返回 topk (label_name, distance) 列表
    """
    desc_inp = make_descriptor(input_path, min_area, max_dist, bin_width, do_morph=True)
    results = []
    for p in labels_dir.iterdir():
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            continue
        desc_lbl = make_descriptor(p, min_area, max_dist, bin_width, do_morph=False)
        L = max(len(desc_inp), len(desc_lbl))
        d1 = np.pad(desc_inp, (0, L - len(desc_inp)), 'constant')
        d2 = np.pad(desc_lbl, (0, L - len(desc_lbl)), 'constant')
        dist = np.linalg.norm(d1 - d2)
        results.append((p.name, float(dist)))
    results.sort(key=lambda x: x[1])
    return results[:topk]


def extract_material_id(filename: str):
    """
    从文件名中提取 material_id (如 2dm-1176)
    """
    m = re.search(r"(2dm-\d+)", filename)
    return m.group(1) if m else None


def load_metadata(csv_path: Path):
    """
    从 CSV 加载 material_id -> set(elements) 映射
    """
    meta = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["material_id"]
            elems = row["elements"].strip('"')
            meta[mid] = set(elems.split(','))
    return meta


def refine_top1(top_matches, recon_id, metadata):
    """
    在 top_matches 中筛选与 recon_id 元素集相同的那些，
    若有多个则返回距离最小的那一个，否则返回原 top1。
    """
    recon_elems = metadata.get(recon_id)
    if recon_elems is None:
        return top_matches[0][0]
    filtered = []
    for name, dist in top_matches:
        label_id = extract_material_id(name)
        lbl_elems = metadata.get(label_id)
        if lbl_elems == recon_elems:
            filtered.append((name, dist))
    return (filtered or top_matches)[0][0]


def process_image(inp: Path,
                  labels_dir: Path,
                  json_dir: Path,
                  copy_root: Path,
                  metadata: dict,
                  topk: int,
                  min_area: int,
                  max_dist: float,
                  bin_width: float):
    """
    单张图像的处理逻辑，可并行调用
    """
    recon_id = extract_material_id(inp.stem)
    top_matches = match_one(inp, labels_dir, topk, min_area, max_dist, bin_width)

    # 写 JSON
    out_json = json_dir / f"{inp.stem}_matches.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([{"label": name, "distance": dist} for name, dist in top_matches],
                  f, indent=2, ensure_ascii=False)
    print(f"✔ {inp.name} → wrote {out_json.name}")

    # 筛选并复制
    if top_matches:
        selected = refine_top1(top_matches, recon_id, metadata)
        src = labels_dir / selected
        dest_sub = copy_root / inp.stem
        dest_sub.mkdir(parents=True, exist_ok=True)
        dst = dest_sub / selected
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied selected {selected} → {dest_sub}")
        else:
            print(f"  ⚠️  Selected not found: {src}")


def main():
    parser = argparse.ArgumentParser(
        description="批量匹配去噪图像与 label，元素过滤后复制 top-1"
    )
    parser.add_argument("--input-dir",
                        required=True,
                        help="去噪后图像的文件夹路径")
    parser.add_argument("--labels-dir",
                        required=True,
                        help="label 图像文件夹路径")
    parser.add_argument("--topk", type=int, default=3,
                        help="返回前 K 名")
    parser.add_argument("--min-area", type=int, default=5,
                        help="连通区域最小像素面积，过滤噪声（仅对去噪图像）")
    parser.add_argument("--max-dist", type=float, default=None,
                        help="RDF 最大距离，默认自动取点对最大距离")
    parser.add_argument("--bin-width", type=float, default=5.0,
                        help="RDF 直方图 bin 宽度")
    parser.add_argument("--output-json-dir", required=True,
                        help="保存每张匹配结果 JSON 的文件夹")
    parser.add_argument("--copy-dir", required=True,
                        help="复制 top-1 label 的输出根目录")
    parser.add_argument("--meta-csv", required=True,
                        help="包含 material_id 与 elements 映射的 CSV 文件")
    parser.add_argument("--workers", type=int, default=96,
                        help="并行 worker 数量")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    labels_dir = Path(args.labels_dir)
    json_dir = Path(args.output_json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    copy_root = Path(args.copy_dir)
    metadata = load_metadata(Path(args.meta_csv))

    # 收集所有待处理图像
    images = [p for p in input_dir.iterdir()
              if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]

    # 部分化
    worker = partial(
        process_image,
        labels_dir=labels_dir,
        json_dir=json_dir,
        copy_root=copy_root,
        metadata=metadata,
        topk=args.topk,
        min_area=args.min_area,
        max_dist=args.max_dist,
        bin_width=args.bin_width
    )

    # 并行执行
    with Pool(args.workers) as pool:
        pool.map(worker, images)

    print("\nAll done.")


if __name__ == "__main__":
    main()
