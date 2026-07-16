import os
import cv2
import numpy as np
import random


def random_sample_from_list(input_list, num_samples, seed=3):
    """
    随机从列表中选择指定数量的元素。

    参数:
        input_list (list): 要采样的列表。
        num_samples (int): 要随机选择的元素数量。
        seed (int): 随机种子。

    返回:
        list: 随机选择的元素列表。
    """
    random.seed(seed)  # 设置随机种子
    if num_samples > len(input_list):
        raise ValueError("请求的样本数量超过列表大小。")
    return random.sample(input_list, num_samples)


# 旋转
def rotate(images, angle, center=None, scale=1.0):
    """
    旋转图像。

    参数:
        images (list of numpy.ndarray): 要旋转的图像列表。
        angle (float): 旋转角度。
        center (tuple): 旋转中心。
        scale (float): 缩放比例。

    返回:
        list of numpy.ndarray: 旋转后的图像列表。
    """
    rotated_images = []
    for img in images:
        (h, w) = img.shape[:2]
        if center is None:
            center_pt = (w / 2, h / 2)
        else:
            center_pt = center
        M = cv2.getRotationMatrix2D(center_pt, angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        rotated_images.append(rotated)
    return rotated_images


# 翻转
def flip(images):
    """
    水平翻转图像。

    参数:
        images (list of numpy.ndarray): 要翻转的图像列表。

    返回:
        list of numpy.ndarray: 翻转后的图像列表。
    """
    return [np.fliplr(img) for img in images]


def center_crop(images, size):
    """
    中心裁剪图像。

    参数:
        images (list of numpy.ndarray): 要裁剪的图像列表。
        size (tuple): 目标裁剪大小 (高度, 宽度)。

    返回:
        list of numpy.ndarray: 裁剪后的图像列表。
    """
    cropped_images = []
    for img in images:
        h, w = img.shape[:2]
        target_h, target_w = size
        if h < target_h or w < target_w:
            raise ValueError("目标大小大于输入图像大小。")
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        cropped = img[top:top + target_h, left:left + target_w]
        cropped_images.append(cropped)
    return cropped_images


def sliding_window_crop(images, size, stride):
    """
    使用滑动窗口裁剪图像。

    参数:
        images (list of numpy.ndarray): 要裁剪的图像列表。
        size (tuple): 目标裁剪大小 (高度, 宽度)。
        stride (int): 滑动窗口步长。

    返回:
        list of list of numpy.ndarray: 每张图像裁剪后的多个片段列表。
    """
    cropped_images = [[] for _ in images]
    h, w = images[0].shape[:2]
    target_h, target_w = size
    if h < target_h or w < target_w:
        raise ValueError("目标大小大于输入图像大小。")
    for top in range(0, h - target_h + 1, stride):
        for left in range(0, w - target_w + 1, stride):
            for idx, img in enumerate(images):
                cropped = img[top:top + target_h, left:left + target_w]
                cropped_images[idx].append(cropped)
    return cropped_images


def darker(images, percentage=0.9):
    """
    使图像变暗。
    """
    return [np.clip((img * percentage).astype(np.uint8), 0, 255) for img in images]


def brighter(images, percentage=1.2):
    """
    增加图像亮度。
    """
    return [np.clip((img * percentage).astype(np.uint8), 0, 255) for img in images]


def zoom(images, scale_range=(1.0, 1.5)):
    """
    随机缩放图像。
    """
    min_scale, max_scale = scale_range
    scale_factor = random.uniform(min_scale, max_scale)
    resized_images = []
    for idx, img in enumerate(images):
        # 对原始图像和标签都采用相同插值方法
        resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        resized_images.append(np.clip(resized, 0, 255))
    return resized_images


if __name__ == '__main__':
    # 定义路径
    path = './data_generation/simulate_data/img'  # 原始图像路径
    label_path = './data_generation/simulate_data/label'  # 标签图像路径
    output_path = './data_generation/simulate_data/aug_img'  # 输出图像路径
    output_label_path = './data_generation/simulate_data/aug_label'  # 输出标签路径

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)

    # 获取所有文件
    file_dir = os.listdir(path)

    for j in range(len(file_dir)):
        tar_path = os.path.join(path, file_dir[j])
        label_tar_path = os.path.join(label_path, os.path.splitext(file_dir[j])[0] + '.png')

        # 加载图像和标签
        img = cv2.imread(tar_path, cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(label_tar_path, cv2.IMREAD_GRAYSCALE)

        if img is None or label_img is None:
            print(f"加载图像失败: {file_dir[j]}。跳过...")
            continue

        # 定义图像组（索引0为标签, 索引1为原始图像）
        images = [label_img, img]

        # 旋转和中心裁剪
        np.random.seed(42)
        rotate_angles = random.sample(range(181), 7)  # 随机选择7个不重复的角度

        file_base_name = os.path.splitext(file_dir[j])[0]

        for angle in rotate_angles:
            # 旋转图像并避免黑边
            rotated_images = rotate(images, angle)
            try:
                cropped_rotated = center_crop(rotated_images, (128, 128))
            except ValueError as e:
                print(f"中心裁剪失败: {e} 对于文件 {file_dir[j]}，角度 {angle}。跳过当前角度...")
                continue

            # 定义文件名
            save_name_center = f'{file_base_name}_r{angle}_center.png'

            # 保存中心裁剪后的图像：索引1为原始图像，索引0为标签
            cv2.imwrite(os.path.join(output_path, save_name_center), cropped_rotated[1])
            cv2.imwrite(os.path.join(output_label_path, save_name_center), cropped_rotated[0])

        # ----------------- 对原始图像进行滑动窗口裁剪 -----------------
        stride = 128  # 定义步长
        try:
            sliding_crops = sliding_window_crop(images, (128, 128), stride)  # 对两个图像进行滑动窗口裁剪
        except ValueError as e:
            print(f"滑动窗口裁剪失败: {e} 对于文件 {file_dir[j]}。跳过滑动窗口裁剪...")
            sliding_crops = []

        num_slides = len(sliding_crops[0]) if sliding_crops else 0

        for idx in range(num_slides):
            # 获取每个裁剪片段
            slide_label = sliding_crops[0][idx]
            slide_img = sliding_crops[1][idx]

            # 定义文件名
            save_name_slide = f'{file_base_name}_slide{idx}.png'

            # 保存滑动窗口裁剪后的图像
            cv2.imwrite(os.path.join(output_path, save_name_slide), slide_img)
            cv2.imwrite(os.path.join(output_label_path, save_name_slide), slide_label)

    # 第二轮数据增强：处理已保存的中心裁剪和滑动窗口裁剪后的图像
    second_output_files = os.listdir(output_path)
    for second_augment_file in second_output_files:
        input_tar_path = os.path.join(output_path, second_augment_file)
        input_label_tar_path = os.path.join(output_label_path, second_augment_file)

        # 定义基准文件名（不含扩展名）
        output_tar_base = os.path.splitext(second_augment_file)[0]

        # 加载原始增强图像组（仅灰度图，索引0为标签，索引1为图像）
        img_second = cv2.imread(input_tar_path, cv2.IMREAD_GRAYSCALE)
        label_img_second = cv2.imread(input_label_tar_path, cv2.IMREAD_GRAYSCALE)

        if img_second is None or label_img_second is None:
            print(f"加载增强图像失败: {second_augment_file}。跳过...")
            continue

        images_second = [label_img_second, img_second]

        # ----------------- 翻转（对 image 和 label 同步翻转） -----------------
        flipped_images = flip(images_second)
        cv2.imwrite(os.path.join(output_path, f'{output_tar_base}_fli.png'), flipped_images[1])
        cv2.imwrite(os.path.join(output_label_path, f'{output_tar_base}_fli.png'), flipped_images[0])

        # ----------------- 亮度调整 -----------------
        # 对原始图像应用亮度调整，而 label 保持不变
        darker_image = darker([images_second[1]])[0]
        brighter_image = brighter([images_second[1]])[0]
        cv2.imwrite(os.path.join(output_path, f'{output_tar_base}_darker.png'), darker_image)
        cv2.imwrite(os.path.join(output_path, f'{output_tar_base}_brighter.png'), brighter_image)
        cv2.imwrite(os.path.join(output_label_path, f'{output_tar_base}_darker.png'), images_second[0])
        cv2.imwrite(os.path.join(output_label_path, f'{output_tar_base}_brighter.png'), images_second[0])

        # ----------------- 高斯模糊 -----------------
        # 仅对原始图像应用高斯模糊，label 保持不变
        blurred_img15 = cv2.GaussianBlur(images_second[1], (3, 3), 1.5)
        blurred_img10 = cv2.GaussianBlur(images_second[1], (3, 3), 1.0)
        cv2.imwrite(os.path.join(output_path, f'{output_tar_base}_blur15.png'), blurred_img15)
        cv2.imwrite(os.path.join(output_path, f'{output_tar_base}_blur10.png'), blurred_img10)
        cv2.imwrite(os.path.join(output_label_path, f'{output_tar_base}_blur15.png'), images_second[0])
        cv2.imwrite(os.path.join(output_label_path, f'{output_tar_base}_blur10.png'), images_second[0])

        # ----------------- 随机缩放 -----------------
        # 对随机缩放的操作保持两者同步（image 和 label 均缩放，以保持对应关系）
        zoomed_images = zoom(images_second, scale_range=(1.0, 1.2))
        cv2.imwrite(os.path.join(output_path, f'{output_tar_base}_zoom.png'), zoomed_images[1])
        cv2.imwrite(os.path.join(output_label_path, f'{output_tar_base}_zoom.png'), zoomed_images[0])

    print("数据增强完成！")
