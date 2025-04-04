#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M2_BASE_MASKS - 直接使用CAM生成二值掩码（不使用CRF后处理）

输入: outputs/cams/*.npy - CAM文件
输出: outputs/base_pseudo/*.png - 二值掩码 (0表示背景，1表示前景)
"""

import os
import numpy as np
import cv2
from pathlib import Path
import argparse
from config import BASE_MASK_CONFIG, CAM_DIR, SEGMENTATION_DIR

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

def apply_simple_threshold(cam, threshold=0.5, morph_kernel_size=5):
    """
    对CAM应用简单阈值处理以生成二值掩码
    
    参数:
        cam: 类激活图数组，形状为(H, W)，值在[0, 1]范围内
        threshold: 阈值，默认为0.5
        morph_kernel_size: 形态学操作的核大小
        
    返回:
        numpy.ndarray: 形状为(H, W)的二值掩码，值为0或1
    """
    if cam is None:
        return None
    
    # 应用阈值
    binary_mask = (cam > threshold).astype(np.uint8)
    
    # 应用形态学操作以填充小空洞并平滑边缘
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    return binary_mask

def process_cam_to_mask(cam_path, output_dir, threshold=0.5, adaptive_threshold=True, morph_kernel_size=5):
    """
    处理单个CAM文件并保存结果掩码（不使用CRF）
    
    参数:
        cam_path: CAM .npy文件的路径
        output_dir: 保存结果掩码的目录
        threshold: 阈值，默认为0.5
        adaptive_threshold: 是否使用自适应阈值
        morph_kernel_size: 形态学操作的核大小
        
    返回:
        bool: 处理成功返回True，否则返回False
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
            # 将CAM转换为uint8以用于Otsu阈值
            cam_uint8 = (cam * 255).astype(np.uint8)
            otsu_threshold, _ = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold = otsu_threshold / 255.0  # 转换回[0,1]范围
            print(f"Using adaptive threshold: {threshold:.3f}")
        
        # 应用阈值处理
        mask = apply_simple_threshold(cam, threshold, morph_kernel_size)
        
        # 确保掩码是严格的二值图像(0和1)，而不是(0和255)
        # 这一步很重要，它确保保存的掩码文件能被正确读取为二值图像
        binary_mask = mask.astype(np.uint8)
        
        # 验证掩码
        unique_values = np.unique(binary_mask)
        foreground_percent = (binary_mask == 1).mean() * 100
        print(f"Mask for {base_name}: unique values={unique_values}, foreground={foreground_percent:.2f}%")
        
        # 将二值掩码(0,1)保存为PNG图像(0,255)
        output_path = Path(output_dir) / f"{base_name}.png"
        
        # 保存前确认掩码值在[0,1]范围内
        if binary_mask.max() > 1:
            print(f"Warning: Mask contains values > 1, max={binary_mask.max()}")
            binary_mask = (binary_mask > 0).astype(np.uint8)
        
        # 将二值掩码保存为具有标准前景值(255)的PNG
        cv2.imwrite(str(output_path), (binary_mask * 255).astype(np.uint8))
        
        # 验证掩码是否正确保存
        saved_mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
        if saved_mask is not None:
            saved_unique = np.unique(saved_mask)
            print(f"Saved mask values: {saved_unique}")
            # 确认保存的掩码只有两个值(0和255)
            if len(saved_unique) > 2:
                print(f"Warning: Saved mask has unexpected values: {saved_unique}")
        
        print(f"Processed: {base_name}")
        return True
    except Exception as e:
        print(f"Error processing {base_name}: {e}")
        return False

def main():
    """
    主函数，处理所有CAM文件并生成二值掩码（不使用CRF）
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Generate binary masks from CAMs without CRF')
    parser.add_argument('--cam_dir', type=str, default=str(CAM_DIR), help='Directory containing CAM .npy files')
    parser.add_argument('--output_dir', type=str, default=str(SEGMENTATION_DIR), help='Directory to save output masks')
    parser.add_argument('--threshold', type=float, default=BASE_MASK_CONFIG["threshold"], help='Threshold for binary mask generation')
    parser.add_argument('--adaptive', action='store_true', default=BASE_MASK_CONFIG["adaptive_threshold"], help='Use adaptive (Otsu) thresholding')
    parser.add_argument('--morph_size', type=int, default=BASE_MASK_CONFIG["morph_kernel_size"], help='Kernel size for morphological operations')
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
        if process_cam_to_mask(cam_file, output_dir, args.threshold, args.adaptive, args.morph_size):
            success_count += 1
    
    print(f"Processing completed. Successfully processed {success_count}/{len(cam_files)} files.")
    print(f"Masks saved to {output_dir}/ directory")

if __name__ == "__main__":
    main() 