#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M2_BASE_MASKS - 直接使用CAM生成二值掩码（不使用CRF后处理）

输入: outputs/cams/*.npy - CAM文件
输出: outputs/base_pseudo/*.png - 二值掩码 (0表示背景，1表示前景)
"""

import os
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
import argparse
from config import BASE_MASK_CONFIG, CAM_DIR, SEGMENTATION_DIR
import cv2

def load_cam(cam_path):
    """
    加载并预处理类激活图(CAM)文件。
    
    参数:
        cam_path: CAM .npy文件的路径
        
    返回:
        numpy.ndarray: 预处理后的CAM，值范围为[0, 1]
    """
    try:
        cam = np.load(cam_path)
        
        # 检查数组是否为空
        if cam.size == 0:
            print(f"Warning: CAM array in {cam_path} is empty")
            return None
        
        # 检查无效值
        if np.isnan(cam).any() or np.isinf(cam).any():
            print(f"Warning: CAM in {cam_path} contains NaN or infinite values, replaced")
            cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 确保CAM是2D数组
        if len(cam.shape) != 2:
            print(f"Warning: CAM in {cam_path} has incorrect shape: {cam.shape}")
            if len(cam.shape) > 2:
                cam = cam[:, :, 0]  # 如果是多通道，只取第一个通道
            elif len(cam.shape) == 1 and cam.shape[0] > 1:
                # 尝试将1D数组重塑为2D
                n = int(np.sqrt(cam.shape[0]))
                if n*n == cam.shape[0]:  # 是否是完美平方数
                    cam = cam.reshape(n, n)
                    print(f"  Reshaped 1D array to {cam.shape}")
                else:
                    print(f"  Cannot reshape 1D array to a square 2D array")
                    return None
        
        # 确保CAM值在[0, 1]范围内
        if cam.max() > 1.0:
            cam = cam / 255.0 if cam.max() > 100 else cam / cam.max()
        
        # 增强高激活区域的对比度
        min_val = cam.min()
        max_val = cam.max()
        if max_val > min_val:
            cam = (cam - min_val) / (max_val - min_val)  # 归一化到[0,1]
        
        print(f"Successfully loaded CAM: {cam_path}, shape: {cam.shape}, range: [{cam.min():.3f}, {cam.max():.3f}]")
        return cam
        
    except Exception as e:
        print(f"Error loading CAM file {cam_path}: {e}")
        return None

def detect_uncertain_regions(cam, mask, kernel_size=5, sigma=1.8):
    """Detect uncertain regions based on edge detection."""
    # Normalize CAM values to range [0, 1] and enhance low values using power function
    cam_norm = np.power(cam / cam.max(), 0.8)
    
    # Get edge magnitude using Sobel operator
    edges_x = cv2.Sobel(cam_norm, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(cam_norm, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    
    # Apply Gaussian blur to smooth edges
    edge_magnitude = cv2.GaussianBlur(edge_magnitude, (0, 0), sigma)
    
    # Normalize edge magnitude to [0, 1]
    if edge_magnitude.max() > 0:
        edge_magnitude = edge_magnitude / edge_magnitude.max()
    
    # Find transition areas - regions where CAM values are in the moderate range
    transition_mask = np.logical_and(cam_norm >= 0.25, cam_norm <= 0.75)
    
    # Combine edge information with transition areas
    uncertain_regions = np.logical_or(edge_magnitude > 0.2, transition_mask)
    uncertain_regions = uncertain_regions.astype(np.uint8) * 255
    
    # Create initial uncertain mask
    uncertain_mask = uncertain_regions.copy()
    
    # Use morphology operations to clean up the uncertain mask
    current_uncertain_percent = np.sum(uncertain_mask > 0) / uncertain_mask.size * 100
    
    # If uncertain regions are too small, expand them
    if current_uncertain_percent < 10:
        # Use a larger dilation structure for expansion
        kernel = np.ones((3, 3), np.uint8)  # Using fixed size 3x3 kernel instead of variable size
        uncertain_mask = cv2.dilate(uncertain_mask, kernel, iterations=2)
    
    # Calculate current percentage of uncertain regions
    current_uncertain_percent = np.sum(uncertain_mask > 0) / uncertain_mask.size * 100
    
    # If uncertain regions are too large, shrink them
    if current_uncertain_percent > 15:
        kernel = np.ones((3, 3), np.uint8)  # Using fixed size 3x3 kernel
        uncertain_mask = cv2.erode(uncertain_mask, kernel, iterations=1)
    
    # If foreground percentage is too small, try to expand it
    foreground_percent = np.sum(mask == 1) / mask.size * 100
    if foreground_percent < 25:
        # Create a temporary mask to expand foreground
        temp_mask = mask.copy()
        kernel = np.ones((3, 3), np.uint8)  # Using fixed size 3x3 kernel
        temp_mask = cv2.dilate(temp_mask.astype(np.uint8), kernel, iterations=1)
        
    return uncertain_mask > 0

def apply_simple_threshold(cam, threshold=0.15, morph_kernel_size=3, adaptive_threshold=False, target_fg_percent=30.0, max_fg_percent=40.0):
    """Apply simple thresholding to generate a binary mask from the CAM."""
    # Enhance CAM contrast using power function to better distinguish foreground
    enhanced_cam = np.power(cam, 0.7)
    
    # Apply initial threshold
    binary_mask = (enhanced_cam > threshold).astype(np.uint8)
    
    # If adaptive threshold is enabled, try to find a threshold that gives a good foreground percentage
    if adaptive_threshold:
        current_fg_percent = binary_mask.mean() * 100
        print(f"Using adaptive threshold: {threshold:.3f}")
        
        # Dynamically adjust threshold to try to reach target foreground percentage
        if current_fg_percent < target_fg_percent / 2:  # Too little foreground
            for new_threshold in np.arange(threshold, 0.2, -0.05):  # More aggressive lowering
                temp_mask = (enhanced_cam > new_threshold).astype(np.uint8)
                temp_fg_percent = temp_mask.mean() * 100
                if temp_fg_percent >= target_fg_percent or temp_fg_percent >= max_fg_percent:
                    binary_mask = temp_mask
                    break
    
    # Apply morphological operations to clean up the mask
    if morph_kernel_size > 0:
        kernel = np.ones((3, 3), np.uint8)  # Using fixed size 3x3 kernel
        # Opening (erosion followed by dilation) to remove small noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        # Closing (dilation followed by erosion) to fill small holes
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Create a trimap by detecting uncertain regions
    uncertain_mask = detect_uncertain_regions(cam, binary_mask, kernel_size=3)
    
    return binary_mask

def numpy_otsu_threshold(image):
    """
    使用改进的Otsu阈值方法
    """
    # 将输入归一化到[0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # 计算图像直方图
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 类别权重
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    
    # 避免除零
    w1[w1 == 0] = 1
    w2[w2 == 0] = 1
    
    # 类别均值
    mean1 = np.cumsum(hist * bin_centers) / w1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / w2[::-1])[::-1]
    
    # 类间方差
    variance = w1[:-1] * w2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # 找到最大类间方差对应的阈值
    idx = np.argmax(variance)
    threshold = bin_centers[idx]
    
    # 返回归一化的阈值
    return max(0.3, min(0.9, threshold / 255.0))  # 限制阈值范围在[0.3, 0.9]之间

def process_cam_to_mask(cam_path, output_dir, threshold=0.7, adaptive_threshold=True, morph_kernel_size=5, target_fg_percent=15.0, max_fg_percent=25.0, uncertainty_threshold=0.2, edge_width=2):
    """
    处理单个CAM文件并保存结果掩码
    
    参数:
        cam_path: CAM文件路径
        output_dir: 输出目录
        threshold: 基础阈值
        adaptive_threshold: 是否使用自适应阈值
        morph_kernel_size: 形态学操作核大小
        target_fg_percent: 目标前景比例
        max_fg_percent: 最大前景比例
        uncertainty_threshold: 不确定性阈值
        edge_width: 边缘宽度
    """
    # 加载并预处理CAM
    cam = load_cam(cam_path)
    if cam is None:
        print(f"Error: Cannot process CAM file {cam_path}")
        return False
    
    # 获取文件路径
    cam_file = Path(cam_path).name
    base_name = Path(cam_path).stem
    if base_name.endswith('_cam'):
        base_name = base_name[:-4]  # 移除_cam后缀
    
    try:
        # 如果需要，应用自适应阈值
        if adaptive_threshold:
            otsu_threshold = numpy_otsu_threshold(cam)
            threshold = otsu_threshold
            print(f"Using adaptive threshold: {threshold:.3f}")
        
        # 应用阈值处理，传递所有参数
        mask = apply_simple_threshold(
            cam, 
            threshold, 
            morph_kernel_size, 
            adaptive_threshold,
            target_fg_percent,
            max_fg_percent
        )
        
        # 验证掩码值
        unique_values = np.unique(mask)
        foreground_percent = (mask == 1).mean() * 100
        uncertain_percent = (mask == 255).mean() * 100
        print(f"Mask for {base_name}: unique values={unique_values}")
        print(f"Foreground: {foreground_percent:.2f}%, Uncertain: {uncertain_percent:.2f}%")
        
        # 保存掩码
        output_path = Path(output_dir) / f"{base_name}.png"
        img = Image.fromarray(mask.astype(np.uint8))
        img.save(str(output_path))
        
        # 验证保存的掩码
        saved_mask = np.array(Image.open(str(output_path)))
        saved_unique = np.unique(saved_mask)
        print(f"Saved mask values: {saved_unique}")
        
        print(f"Processed: {base_name}")
        return True
        
    except Exception as e:
        print(f"Error processing {base_name}: {e}")
        return False

def main():
    """
    主函数，处理所有CAM文件并生成三值掩码
    """
    parser = argparse.ArgumentParser(description='Generate trimaps from CAMs without CRF')
    parser.add_argument('--cam_dir', type=str, default=str(CAM_DIR), help='Directory containing CAM .npy files')
    parser.add_argument('--output_dir', type=str, default=str(SEGMENTATION_DIR), help='Directory to save output masks')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for binary mask generation')
    parser.add_argument('--adaptive', action='store_true', default=True, help='Use adaptive (Otsu) thresholding')
    parser.add_argument('--morph_size', type=int, default=3, help='Kernel size for morphological operations')
    parser.add_argument('--target_fg', type=float, default=25.0, help='Target foreground percentage')
    parser.add_argument('--max_fg', type=float, default=35.0, help='Maximum foreground percentage')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.15, help='Threshold for uncertain regions')
    parser.add_argument('--edge_width', type=int, default=3, help='Width of edge regions')
    args = parser.parse_args()
    
    # 设置路径
    cam_dir = args.cam_dir
    output_dir = args.output_dir
    
    # 如果输出目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CAM文件
    cam_files = [f for f in Path(cam_dir).glob("*.npy")]
    
    if not cam_files:
        print(f"Error: No .npy files found in {cam_dir}")
        return
    
    print(f"Found {len(cam_files)} CAM files")
    print(f"Using {'adaptive' if args.adaptive else 'fixed'} thresholding, threshold={args.threshold if not args.adaptive else 'auto'}")
    
    # 处理每个CAM文件
    success_count = 0
    for cam_file in cam_files:
        if process_cam_to_mask(
            cam_file, 
            output_dir, 
            args.threshold, 
            args.adaptive, 
            args.morph_size, 
            args.target_fg, 
            args.max_fg,
            args.uncertainty_threshold,
            args.edge_width
        ):
            success_count += 1
    
    print(f"Processing completed. Successfully processed {success_count}/{len(cam_files)} files.")
    print(f"Masks saved to {output_dir}/ directory")

if __name__ == "__main__":
    main() 