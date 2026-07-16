#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from math import pi
from typing import Tuple

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

# 设置使用的GPU设备（修改为你的实际设备编号）
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
        args: 从 argparse 传入的参数对象
    """
    print("abTEM version:", __version__)
    print("Reading CIF file:", args.cif_file)

    # 读取 CIF 文件，生成原始 Atoms 对象
    atoms = read(args.cif_file)
    orientation = tuple(int(x) for x in args.orientation.split(','))
    atoms = surface(atoms, indices=orientation, layers=1, vacuum=2, periodic=True)

    # 重复晶胞（cell_size 参数控制各方向重复次数）
    cell_size = tuple(int(x) for x in args.cell_size.split(','))
    repetitions = cell_size
    atoms *= repetitions
    print("Initial cell:", atoms.cell)

    # 转换为正交晶胞
    # orthogonal_atoms, transform = orthogonalize_cell(atoms, max_repetitions=50, return_transform=True)
    orthogonal_atoms = atoms
    print("Orthogonal cell:")
    print(orthogonal_atoms.cell)

    # 如设置了输出文件，则保存正交晶胞
    if args.output_cif:
        write(args.output_cif, orthogonal_atoms, format="cif")
        print(f"Orthogonal cell saved to: {args.output_cif}")

    # 创建 Potential 对象（用于模拟物理势场），参数中 gpts、projection、slice_thickness、parametrization 固定
    potential = Potential(orthogonal_atoms,
                          gpts=2048,
                          projection='infinite',
                          slice_thickness=1,
                          parametrization='kirkland',
                          device=args.device)
    print("Real space sampling:", potential.sampling)

    # 创建 Probe 对象（电子探针），能量、半角截止、rolloff、defocus 等可按需要修改
    probe = Probe(energy=300e3, semiangle_cutoff=21.3, rolloff=0.1, defocus=0, device=args.device)
    probe.grid.match(potential)

    # calculate FWHM
    array = probe.profile().array
    peak_idx = np.argmax(array)
    peak_value = array[peak_idx]
    left = np.argmin(np.abs(array[:peak_idx] - peak_value / 2))
    right = peak_idx + np.argmin(np.abs(array[peak_idx:] - peak_value / 2))
    fwhm = right - left
    fwhm = fwhm * probe.profile().calibrations[0].sampling
    print('fwhm: ', fwhm)

    # 创建探测器：HAADF、Bright field 和 DPC 探测器
    haadf = AnnularDetector(inner=50, outer=200)
    bright = AnnularDetector(inner=0, outer=11)
    dpc_detector = SegmentedDetector(inner=4, outer=22, nbins_radial=1, nbins_angular=4, rotation=-np.pi / 4)

    # 显示探针、CTF等图像（可选）
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # probe.profile().show(ax=axes[0]); probe.ctf.show(50, ax=axes[1]); probe.show(ax=axes[2])

    # 定义扫描区域：这里使用 (0,0) 到 (extent_x, extent_y)，extent 根据 Potential 的属性和 cell_size 计算
    start = (potential.extent[0] / cell_size[0], potential.extent[1] / cell_size[1])
    scan_end = (potential.extent[0] / cell_size[0], potential.extent[1] / cell_size[1])
    gridscan = GridScan(start=(0, 0), end=scan_end, sampling=probe.ctf.nyquist_sampling * 0.9)
    measurement = probe.scan(gridscan, [dpc_detector, haadf, bright], potential)

    # 解析采样与剂量参数（字符串逗号分隔转换为数值列表）
    sampling_values = [float(s) for s in args.sampling_values.split(',')]
    dose_values = [int(d) for d in args.doses.split(',')]

    # 针对不同采样值和不同剂量，进行 DPC 信号模拟、图像重建及保存
    for sampling in sampling_values:
        # 计算高斯核参数
        FWHM = args.FWHM  # 单位：埃
        length = sampling
        kernel_size = args.kernel_size
        best_sigma = FWHM / (length * 2.355)
        gauss_kernel = gauss(kernel_size, best_sigma)

        for dose in dose_values:
            print(f"Processing dose: {dose} at sampling: {sampling}")

            # 对 DPC 信号进行插值（measurement[0] 中各通道数据），tile 保持数据格式
            dpc_1 = measurement[0][:, :, 1].tile((1, 1)).interpolate(sampling)
            dpc_2 = measurement[0][:, :, 2].tile((1, 1)).interpolate(sampling)
            dpc_3 = measurement[0][:, :, 3].tile((1, 1)).interpolate(sampling)
            dpc_4 = measurement[0][:, :, 0].tile((1, 1)).interpolate(sampling)

            # 添加泊松噪声模拟实验中噪声水平
            dpc_1 = noise.poisson_noise(dpc_1, dose)
            dpc_2 = noise.poisson_noise(dpc_2, dose)
            dpc_3 = noise.poisson_noise(dpc_3, dose)
            dpc_4 = noise.poisson_noise(dpc_4, dose)

            # 提取数组进行数值计算
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

            # 生成三种不同处理的 iDPC 图像（原始积分、高通滤波、和高斯滤波）
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

            # 保存 iDPC 图像（这里保存原始积分的图像，可根据需要修改保存哪种结果）
            base_name = os.path.splitext(args.cif_file)[0]
            out_filename = f"{base_name}_dose{dose}_sampling{sampling}_iDPC_V3.tif"
            iDPC_temp.tile((1, 1)).save_as_image(out_filename)
            print(f"Saved iDPC image to: {out_filename}")


def main():
    parser = argparse.ArgumentParser(description="Simulate STEM images using abTEM")
    # 可修改参数
    parser.add_argument('--cif_file', type=str,
                        default="./data_generation/modified_supercell_orthonalize_selected_cifs/orthogonal_2dm-9_supercell_16x16x1.cif",
                        help='Path to the CIF file')
    parser.add_argument('--orientation', type=str, default='0,0,1',
                        help='Surface orientation as comma-separated values (e.g., "1,0,0")')
    parser.add_argument('--cell_size', type=str, default='1,1,1',
                        help='Cell repetition as comma-separated values (e.g., "1,1,1")')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device to use ("gpu" or "cpu")')
    parser.add_argument('--FWHM', type=float, default=0.7,
                        help='Full width at half maximum for Gaussian filter (in Angstroms)')
    parser.add_argument('--sampling_values', type=str, default='0.05, 0.075, 0.10, 0.125, 0.15',
                        help='Comma-separated list of pixel sizes for interpolation')
    parser.add_argument('--doses', type=str, default='1000000',
                        help='Comma-separated list of electron doses')
    parser.add_argument('--kernel_size', type=int, default=101,
                        help='Kernel size for Gaussian filter')
    # 不常修改的参数
    parser.add_argument('--output_cif', type=str, default='./data_generation/example_cif/orthogonal_cell_v2.cif',
                        help='Output filename for the orthogonal CIF file')

    args = parser.parse_args()

    simulate_stem(args)


if __name__ == "__main__":
    main()
