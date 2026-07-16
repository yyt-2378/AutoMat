import numpy as np


def add_poisson_noise_and_merge(original_image):
    """
    向图像添加泊松噪声，并与原图合并以保留更多的原始信息。

    Parameters:
    - original_image: 原始图像，一个numpy数组。

    Returns:
    - merged_image: 合并后的图像。
    """
    # 确保图像为浮点类型以避免数据类型问题
    if original_image.dtype != np.float32 and original_image.dtype != np.float64:
        lam = original_image.astype(np.float32)
    else:
        lam = original_image

    # 生成泊松噪声图
    noisy_image = np.random.poisson(lam)

    # 初始化合并后的图像
    merged_image = np.zeros_like(lam)

    # 查找原图中的非零区域
    non_zero_mask = original_image > 0

    # 计算重叠区域（原图和噪声图都有非零值的区域）
    overlap_mask = non_zero_mask & (noisy_image > 0)

    # 重叠区域像素值取原图和噪声图像素值的平均
    merged_image[overlap_mask] = (lam[overlap_mask] + noisy_image[overlap_mask]) / 2.0

    # 非重叠区域，直接取噪声图的像素值
    merged_image[~overlap_mask] = noisy_image[~overlap_mask]

    return merged_image