#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析DL/outputs/base_pseudo目录中的伪标签前景背景比例

输入: DL/outputs/base_pseudo/*.png - 二值掩码图像
输出: 所有图像的前景背景比例统计
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

# 伪标签目录
PSEUDO_LABEL_DIR = Path("DL/outputs/base_pseudo")

def analyze_pseudo_labels(directory):
    """
    分析目录中所有伪标签图像的前景背景比例
    
    参数:
        directory: 伪标签目录路径
    
    返回:
        统计信息字典
    """
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return None
    
    # 获取所有PNG文件
    mask_files = list(Path(directory).glob("*.png"))
    if not mask_files:
        print(f"Error: No PNG files found in {directory}")
        return None
    
    print(f"Found {len(mask_files)} mask files in {directory}")
    
    # 统计变量
    stats = {
        "total_files": len(mask_files),
        "foreground_ratios": [],
        "background_ratios": [],
        "uncertain_ratios": [],
        "avg_foreground": 0,
        "avg_background": 0,
        "avg_uncertain": 0,
        "min_foreground": 100,
        "max_foreground": 0,
        "foreground_distribution": defaultdict(int)
    }
    
    # 处理每个掩码文件
    for mask_file in mask_files:
        try:
            # 读取掩码图像
            mask = np.array(Image.open(mask_file))
            
            # 统计前景、背景和不确定区域像素数
            total_pixels = mask.size
            foreground_pixels = np.sum(mask == 1)
            background_pixels = np.sum(mask == 0)
            uncertain_pixels = np.sum(mask == 255) if 255 in mask else 0
            
            # 计算比例
            foreground_ratio = foreground_pixels / total_pixels * 100
            background_ratio = background_pixels / total_pixels * 100
            uncertain_ratio = uncertain_pixels / total_pixels * 100
            
            # 更新统计信息
            stats["foreground_ratios"].append(foreground_ratio)
            stats["background_ratios"].append(background_ratio)
            stats["uncertain_ratios"].append(uncertain_ratio)
            
            # 记录前景比例分布
            foreground_bin = int(foreground_ratio / 5) * 5  # 5%为一个区间
            stats["foreground_distribution"][foreground_bin] += 1
            
            # 更新最小/最大前景比例
            stats["min_foreground"] = min(stats["min_foreground"], foreground_ratio)
            stats["max_foreground"] = max(stats["max_foreground"], foreground_ratio)
            
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
    
    # 计算平均值
    if stats["foreground_ratios"]:
        stats["avg_foreground"] = sum(stats["foreground_ratios"]) / len(stats["foreground_ratios"])
        stats["avg_background"] = sum(stats["background_ratios"]) / len(stats["background_ratios"])
        stats["avg_uncertain"] = sum(stats["uncertain_ratios"]) / len(stats["uncertain_ratios"])
    
    return stats

def plot_distribution(stats):
    """绘制前景比例分布直方图"""
    # 准备数据
    bins = sorted(stats["foreground_distribution"].keys())
    counts = [stats["foreground_distribution"][bin] for bin in bins]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    plt.bar(bins, counts, width=4, align='center')
    plt.title('Foreground Pixel Ratio Distribution')
    plt.xlabel('Foreground Ratio (%)')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加平均线
    plt.axvline(x=stats["avg_foreground"], color='r', linestyle='--', 
                label=f'Average: {stats["avg_foreground"]:.2f}%')
    plt.legend()
    
    # 保存图表
    plt.savefig('foreground_distribution.png')
    print(f"Distribution plot saved as foreground_distribution.png")

def main():
    """主函数"""
    print(f"Analyzing pseudo label directory: {PSEUDO_LABEL_DIR}")
    
    # 分析伪标签
    stats = analyze_pseudo_labels(PSEUDO_LABEL_DIR)
    
    if stats:
        # 打印统计结果
        print("\nPseudo Label Statistics:")
        print(f"Total images: {stats['total_files']}")
        print(f"Average foreground ratio: {stats['avg_foreground']:.2f}%")
        print(f"Average background ratio: {stats['avg_background']:.2f}%")
        print(f"Average uncertain area ratio: {stats['avg_uncertain']:.2f}%")
        print(f"Minimum foreground ratio: {stats['min_foreground']:.2f}%")
        print(f"Maximum foreground ratio: {stats['max_foreground']:.2f}%")
        
        # 打印前景分布
        print("\nForeground Ratio Distribution:")
        for bin in sorted(stats["foreground_distribution"].keys()):
            count = stats["foreground_distribution"][bin]
            percent = count / stats["total_files"] * 100
            print(f"{bin}-{bin+5}%: {count} images ({percent:.1f}%)")
        
        # 绘制分布图
        plot_distribution(stats)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 