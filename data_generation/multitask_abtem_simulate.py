#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from math import pi
from typing import Tuple
import concurrent.futures

# 导入 abTEM 及 ASE 库（确保已安装）
from abtem import __version__, noise
from abtem.transfer import CTF, scherzer_defocus, point_resolution, energy2wavelength, cartesian2polar
from abtem.measure import center_of_mass, probe_profile
from abtem.detect import SegmentedDetector
from abtem.structures import orthogonalize_cell
from abtem import *
from skimage.filters import butterworth

from ase.io import read, write
from ase.build import surface

# 设置使用的GPU设备（此处默认设置，在任务启动时会修改）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def intgrad2d(gradient: np.ndarray, sampling: Tuple[float, float] = (1.0, 1.0)):
    """
    通过傅里叶空间积分计算梯度积分图像。
    
    参数：
        gradient : 包含 x 和 y 分量的梯度 (gx, gy)
        sampling : 横向采样间隔 (默认 (1.0, 1.0))
    
    返回：
        积分后的图像（去除最小值，使图像非负）
    """
    gx, gy = gradient
    (nx, ny) = gx.shape
    ikx = np.fft.fftfreq(nx, d=sampling[0])
    iky = np.fft.fftfreq(ny, d=sampling[1])
    grid_ikx, grid_iky = np.meshgrid(ikx, iky, indexing='ij')
    k = grid_ikx ** 2 + grid_iky ** 2
    k[k == 0] = 1e-12
    That = (np.fft.fft2(gx) * grid_ikx + np.fft.fft2(gy) * grid_iky) / (2j * np.pi * k)
    T = np.real(np.fft.ifft2(That))
    T -= T.min()
    return T


def gauss(kernel_size: int, sigma: float):
    """
    生成高斯卷积核。
    
    参数：
        kernel_size: 卷积核尺寸（正方形）
        sigma: 标准差，如果 <=0 则自动计算默认值
    
    返回：
        高斯核（归一化后）
    """
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * s))
            sum_val += kernel[i, j]
    kernel /= sum_val
    return kernel


def conv_2d(kernel: np.ndarray, img: np.ndarray, mode: str = 'fill'):
    """
    对图像进行二维卷积操作。
    
    参数：
        kernel: 卷积核
        img: 输入图像
        mode: 填充模式（默认为 'fill'，使用反射填充）
    
    返回：
        卷积后的图像
    """
    if mode == 'fill':
        h = kernel.shape[0] // 2
        w = kernel.shape[1] // 2
        img = np.pad(img, ((h, h), (w, w)), 'reflect')

    res_h = img.shape[0] - kernel.shape[0] + 1
    res_w = img.shape[1] - kernel.shape[1] + 1
    res = np.zeros((res_h, res_w))
    dh, dw = kernel.shape
    for i in range(res_h):
        for j in range(res_w):
            res[i, j] = np.sum(img[i:i + dh, j:j + dw] * kernel)
    return res


def simulate_stem(args):
    """
    模拟 STEM 图像生成的主要流程。
    
    参数：
        args: 从 argparse 传入的参数对象，包含 cif_file、device、orientation、cell_size、FWHM、
              sampling_values、doses、kernel_size 以及 output_folder 等参数。
    """
    print("abTEM version:", __version__)
    print("Reading CIF file:", args.cif_file)

    # 确保输出文件夹存在
    os.makedirs(args.output_folder, exist_ok=True)

    # 读取 CIF 文件，生成原始 Atoms 对象
    atoms = read(args.cif_file)
    orientation = tuple(int(x) for x in args.orientation.split(','))
    atoms = surface(atoms, indices=orientation, layers=1, vacuum=2, periodic=True)

    # 重复晶胞（cell_size 参数控制各方向重复次数）
    cell_size = tuple(int(x) for x in args.cell_size.split(','))
    atoms *= cell_size
    print("Initial cell:", atoms.cell)

    # 转换为正交晶胞（根据需要，可取消注释下面代码）
    # orthogonal_atoms, transform = orthogonalize_cell(atoms, max_repetitions=50, return_transform=True)
    orthogonal_atoms = atoms
    print("Orthogonal cell:")
    print(orthogonal_atoms.cell)

    # 保存正交晶胞到指定的文件夹（输出文件名基于 cif 文件名）
    base_name = os.path.splitext(os.path.basename(args.cif_file))[0]
    output_cif_path = os.path.join(args.output_folder, f"{base_name}_orthogonal.cif")
    write(output_cif_path, orthogonal_atoms, format="cif")
    print(f"Orthogonal cell saved to: {output_cif_path}")

    # 创建 Potential 对象，参数中 gpts、projection、slice_thickness、parametrization 固定
    potential = Potential(orthogonal_atoms,
                          gpts=2048,
                          projection='infinite',
                          slice_thickness=1,
                          parametrization='kirkland',
                          device=args.device)
    print("Real space sampling:", potential.sampling)

    # 创建 Probe 对象，能量、半角截止、rolloff、defocus 等参数可以按需修改
    probe = Probe(energy=300e3, semiangle_cutoff=21.3, rolloff=0.1, defocus=0, device=args.device)
    probe.grid.match(potential)

    # 计算 FWHM
    profile = probe.profile()
    array = profile.array
    peak_idx = np.argmax(array)
    peak_value = array[peak_idx]
    left = np.argmin(np.abs(array[:peak_idx] - peak_value / 2))
    right = peak_idx + np.argmin(np.abs(array[peak_idx:] - peak_value / 2))
    fwhm = (right - left) * profile.calibrations[0].sampling
    print('fwhm: ', fwhm)

    # 创建探测器：HAADF、Bright field 和 DPC 探测器
    haadf = AnnularDetector(inner=50, outer=200)
    bright = AnnularDetector(inner=0, outer=11)
    dpc_detector = SegmentedDetector(inner=4, outer=22, nbins_radial=1, nbins_angular=4, rotation=-np.pi / 4)

    # 定义扫描区域：例如 (0,0) 到 (extent_x, extent_y)
    scan_end = (potential.extent[0] / cell_size[0], potential.extent[1] / cell_size[1])
    gridscan = GridScan(start=(0, 0), end=scan_end, sampling=probe.ctf.nyquist_sampling * 0.9)
    measurement = probe.scan(gridscan, [dpc_detector, haadf, bright], potential)

    # 解析采样和剂量参数（字符串逗号分隔转换为数值列表）
    sampling_values = [float(s.strip()) for s in args.sampling_values.split(',')]
    dose_values = [int(d.strip()) for d in args.doses.split(',')]

    # 针对不同采样和剂量，进行 DPC 信号模拟、图像重建及保存
    for sampling in sampling_values:
        # 计算高斯核参数
        FWHM = args.FWHM  # 单位：埃
        length = sampling
        kernel_size = args.kernel_size
        best_sigma = FWHM / (length * 2.355)
        gauss_kernel = gauss(kernel_size, best_sigma)

        for dose in dose_values:
            print(f"Processing dose: {dose} at sampling: {sampling}")

            # 对 DPC 信号进行插值（假设 measurement[0] 中的各通道数据具备 tile 和 interpolate 方法）
            dpc_1 = measurement[0][:, :, 1].tile((1, 1)).interpolate(sampling)
            dpc_2 = measurement[0][:, :, 2].tile((1, 1)).interpolate(sampling)
            dpc_3 = measurement[0][:, :, 3].tile((1, 1)).interpolate(sampling)
            dpc_4 = measurement[0][:, :, 0].tile((1, 1)).interpolate(sampling)

            # 添加泊松噪声以模拟不同剂量下的噪声水平
            dpc_1 = noise.poisson_noise(dpc_1, dose)
            dpc_2 = noise.poisson_noise(dpc_2, dose)
            dpc_3 = noise.poisson_noise(dpc_3, dose)
            dpc_4 = noise.poisson_noise(dpc_4, dose)

            # 提取数组数据
            dpc_1_array = dpc_1.array
            dpc_2_array = dpc_2.array
            dpc_3_array = dpc_3.array
            dpc_4_array = dpc_4.array

            # 绘制 DPC 信号图像（四个通道）
            fig, axes = plt.subplots(1, 4, figsize=(20, 10))
            dpc_1.tile((1, 1)).show(ax=axes[0], title='DPC_A')
            dpc_2.tile((1, 1)).show(ax=axes[1], title='DPC_B')
            dpc_3.tile((1, 1)).show(ax=axes[2], title='DPC_C')
            dpc_4.tile((1, 1)).show(ax=axes[3], title='DPC_D')
            plt.suptitle(f"DPC Signals at dose {dose}")
            plt.close(fig)

            # 计算水平和垂直方向的 DPC 差异，作为梯度分量
            dpcx = -(dpc_1_array - dpc_3_array)
            dpcy = -(dpc_2_array - dpc_4_array)

            # 通过傅里叶积分得到 iDPC 图像
            idpc_temp = intgrad2d((dpcy, dpcx), sampling=(1.0, 1.0))
            idpc_highpass_temp = butterworth(idpc_temp, 0.02, True, channel_axis=0)

            # 生成三种不同处理的 iDPC 图像（原始积分、高通滤波和高斯滤波）
            iDPC_temp = dpc_1.copy()
            iDPC_temp.array = idpc_temp

            iDPC_highpass = dpc_1.copy()
            iDPC_highpass.array = idpc_highpass_temp

            iDPC_gauss = dpc_1.copy()
            iDPC_gauss.array = conv_2d(gauss_kernel, idpc_temp)

            # 绘制 iDPC 图像对比
            fig, axes = plt.subplots(1, 3, figsize=(20, 12))
            iDPC_temp.tile((1, 1)).show(ax=axes[0], title='iDPC')
            iDPC_highpass.tile((1, 1)).show(ax=axes[1], title='iDPC_highpass')
            iDPC_gauss.tile((1, 1)).show(ax=axes[2], title='iDPC_gauss')
            plt.suptitle(f"iDPC Images at dose {dose} and sampling {sampling}")
            plt.close(fig)

            # 构造输出图像文件名，并保存到指定文件夹
            out_filename = os.path.join(args.output_folder,
                                        f"{base_name}_dose{dose}_sampling{sampling}_iDPC_V3.tif")
            iDPC_temp.tile((1, 1)).save_as_image(out_filename)
            print(f"Saved iDPC image to: {out_filename}")


def worker(cif_file, base_args, gpu_id):
    """
    单个任务的工作函数：
      - 设置该任务使用的 GPU（通过修改环境变量）
      - 使用 base_args 生成新的参数对象，并将 cif_file 参数更新成当前任务的文件
      - 调用 simulate_stem 执行 STEM 模拟
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    new_args = argparse.Namespace(**vars(base_args))
    new_args.cif_file = cif_file
    # 自动生成输出文件名（CIF输出文件也保存到指定文件夹中）
    base_name = os.path.splitext(os.path.basename(cif_file))[0]
    new_args.output_cif = os.path.join(new_args.output_folder, f"{base_name}_orthogonal.cif")
    print(f"[GPU {gpu_id}] Processing file: {cif_file}")
    try:
        simulate_stem(new_args)
    except Exception as e:
        print(f"Error processing {cif_file} on GPU {gpu_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Simulate STEM images using abTEM (Multi-GPU Batch Mode)")
    # 参数设置
    parser.add_argument('--cif_file', type=str,
                        default="./data_generation/modified_supercell_orthonalize_selected_cifs/orthogonal_0_supercell_12x12x1.cif",
                        help='Path to the CIF file (单文件模式)')
    parser.add_argument('--cif_folder', type=str, default="./data_generation/train_selected_samples",
                        help='Path to a folder containing CIF files for batch simulation')
    parser.add_argument('--orientation', type=str, default='0,0,1',
                        help='Surface orientation as comma-separated values (e.g., "1,0,0")')
    parser.add_argument('--cell_size', type=str, default='1,1,1',
                        help='Cell repetition as comma-separated values (e.g., "1,1,1")')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device to use ("gpu" or "cpu")')
    parser.add_argument('--FWHM', type=float, default=0.7,
                        help='Full width at half maximum for Gaussian filter (in Angstroms)')
    parser.add_argument('--sampling_values', type=str, default='0.10',
                        help='Comma-separated list of pixel sizes for interpolation')
    parser.add_argument('--doses', type=str, default='10000, 20000, 30000, 40000, 50000, 60000',
                        help='Comma-separated list of electron doses')
    parser.add_argument('--kernel_size', type=int, default=101,
                        help='Kernel size for Gaussian filter')
    # 输出文件夹参数，用于存放所有生成的文件
    parser.add_argument('--output_folder', type=str,
                        default="./data_generation/output_simulate_stem_train",
                        help='Folder to save all output files')
    # 新增多GPU相关的参数
    parser.add_argument('--gpu_ids', type=str, default="4,5,6,7",
                        help='Comma-separated list of GPU ids to use (默认使用四张 GPU)')
    parser.add_argument('--max_tasks_per_gpu', type=int, default=4,
                        help='Maximum concurrent tasks per GPU')
    
    args = parser.parse_args()

    if args.cif_folder:
        # 批量模式：处理文件夹中所有 .cif 文件
        cif_files = sorted(glob.glob(os.path.join(args.cif_folder, '*.cif')))
        if not cif_files:
            print("Error: No CIF files found in folder:", args.cif_folder)
            sys.exit(1)
        # 解析 GPU id 列表
        gpu_ids = [g.strip() for g in args.gpu_ids.split(',')]
        num_gpus = len(gpu_ids)
        # 计算总的并发任务数（每张 GPU 同时处理 max_tasks_per_gpu 个任务）
        max_workers = num_gpus * args.max_tasks_per_gpu
        print(f"Found {len(cif_files)} CIF files. Using GPUs: {gpu_ids} with maximum {max_workers} total parallel tasks.")

        # 使用 ProcessPoolExecutor 分发任务
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, cif_file in enumerate(cif_files):
                assigned_gpu = gpu_ids[idx % num_gpus]
                futures.append(executor.submit(worker, cif_file, args, assigned_gpu))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
    else:
        simulate_stem(args)


if __name__ == "__main__":
    main()
