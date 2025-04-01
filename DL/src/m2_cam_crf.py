
"""
M2_CAM_CRF - 使用全连接CRF对CAM进行后处理以生成二值掩码

输入: outputs/cams/*.npy - CAM文件
输出: outputs/pseudo_masks/*.png - 二值掩码 (0表示背景，1表示前景)
"""

import os
import numpy as np
import cv2
from pathlib import Path
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

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

def get_image_path(cam_filename, image_dir):
    """
    为CAM文件找到对应的原始图像路径。
    
    参数:
        cam_filename: CAM文件名
        image_dir: 包含原始图像的目录
        
    返回:
        Path: 原始图像文件的路径，如果未找到则为None
    """
    
    base_name = Path(cam_filename).stem
    if base_name.endswith('_cam'):
        base_name = base_name[:-4]  
    
   
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = Path(image_dir) / f"{base_name}{ext}"
        if img_path.exists():
            return img_path
    
    return None

def apply_crf(image, cam, config=None):
    """
    应用全连接条件随机场(CRF)来优化CAM。
    
    参数:
        image: 原始图像数组，形状为(H, W, 3)
        cam: 类激活图数组，形状为(H, W)，值在[0, 1]范围内
        config: CRF参数字典
        
    返回:
        numpy.ndarray: 形状为(H, W)的二值掩码，值为0或1
    """
    if config is None:
        config = {
            'pos_w': 3,       # 位置权重
            'pos_xy_std': 3,  # 位置核标准差
            'bi_w': 4,        # 双边权重
            'bi_xy_std': 67,  # 双边空间标准差
            'bi_rgb_std': 3,  # 双边颜色标准差
            'iterations': 5   # 推理迭代次数
        }
    
    # 如果需要，将图像转换为uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 如果需要，将灰度图转换为BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 检查CAM数组是否有效
    if cam is None or cam.size == 0:
        print("Error: CAM array is empty")
        return np.zeros((h, w), dtype=np.uint8)
        
    # 确保CAM不包含NaN或无限值
    if np.isnan(cam).any() or np.isinf(cam).any():
        print("Warning: CAM contains NaN or infinite values, replaced with 0")
        cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 确保CAM是2D数组
    if len(cam.shape) != 2:
        print(f"Warning: CAM shape incorrect {cam.shape}, attempting to convert")
        if len(cam.shape) > 2:
            cam = cam[:, :, 0]  # 如果是多通道，只取第一个通道
        elif len(cam.shape) == 1:
            # 如果是1D数组，可能需要重构或无法处理
            print("Error: Cannot process 1D CAM array")
            return np.zeros((h, w), dtype=np.uint8)
    
    # 打印调试信息
    print(f"CAM shape: {cam.shape}, min: {cam.min()}, max: {cam.max()}, dtype: {cam.dtype}")
    print(f"Target shape for resize: ({w}, {h})")
    
    try:
        # 尝试调整CAM大小以匹配图像尺寸
        cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    except cv2.error as e:
        print(f"OpenCV resize error: {e}")
        print(f"Attempting alternative method to resize")
        # 使用另一种OpenCV调整大小的方法
        cam_float32 = cam.astype(np.float32)
        cam_resized = cv2.resize(cam_float32, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 使用Otsu方法进行简单阈值处理
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    _, otsu_mask = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_threshold = otsu_mask.astype(float) / 255.0
    
    # 将阈值化CAM转换为概率图（软化边界）
    fg_prob = np.zeros_like(cam_resized)
    # 将高激活区域标记为高概率前景
    fg_prob[cam_resized > 0.7] = 0.9    # 强前景(90%置信度)
    # 将中等激活区域标记为中等概率前景
    fg_prob[(cam_resized > 0.3) & (cam_resized <= 0.7)] = 0.75  # 中等前景(75%置信度)
    # 将低激活区域标记为背景
    fg_prob[cam_resized <= 0.3] = 0.1   # 背景(10%置信度)
    
    # 初始化CRF概率数组
    probs = np.zeros((2, h, w), dtype=np.float32)
    probs[0] = 1.0 - fg_prob  # 背景概率
    probs[1] = fg_prob        # 前景概率
    
    # 创建CRF模型
    d = dcrf.DenseCRF2D(w, h, 2)  # 宽度，高度，标签数
    
    # 设置一元势能
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)
    
    # 添加与位置无关的项
    d.addPairwiseGaussian(
        sxy=config['pos_xy_std'],  # 空间标准差
        compat=config['pos_w']     # 兼容性值
    )
    
    # 添加外观核
    d.addPairwiseBilateral(
        sxy=config['bi_xy_std'],    # 空间标准差
        srgb=config['bi_rgb_std'],  # 颜色标准差
        rgbim=image,                # 原始RGB图像
        compat=config['bi_w']       # 兼容性值
    )
    
    # 执行推理
    Q = d.inference(config['iterations'])
    
    # 获取MAP估计
    MAP = np.argmax(Q, axis=0).reshape((h, w))
    
    # 简单的后处理 - 基本闭合操作以填充小洞
    kernel = np.ones((5, 5), np.uint8)
    MAP = cv2.morphologyEx(MAP.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    return MAP.astype(np.uint8)

def process_cam_with_crf(cam_path, image_dir, output_dir, crf_config=None):
    """
    使用CRF处理单个CAM文件并保存结果掩码。
    
    参数:
        cam_path: CAM .npy文件的路径
        image_dir: 包含原始图像的目录
        output_dir: 保存结果掩码的目录
        crf_config: CRF参数的字典
        
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
    
    # 查找原始图像
    image_path = get_image_path(cam_file, image_dir)
    if image_path is None:
        print(f"Warning: Cannot find image corresponding to {base_name}")
        return False
    
    # 加载原始图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Cannot load image {image_path}")
        return False
    
    # 应用CRF
    try:
        mask = apply_crf(image, cam, crf_config)
        
        # 保存结果
        output_path = Path(output_dir) / f"{base_name}.png"
        cv2.imwrite(str(output_path), (mask * 255).astype(np.uint8))
        
        print(f"Processed: {base_name}")
        return True
    except Exception as e:
        print(f"Error processing {base_name}: {e}")
        return False

def main():
    """
    主函数，使用CRF处理所有CAM文件并生成二值掩码。
    """
    # 设置路径
    cam_dir = "outputs/cams"
    image_dir = "data/images"
    output_dir = "outputs/pseudo_masks"
    
    # 如果输出目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)
    

    crf_config = {
        'pos_w': 3,         # 标准位置权重
        'pos_xy_std': 5,    # 位置标准差，用于更好的空间一致性
        'bi_w': 7,          # 双边权重，用于更好的边缘附着
        'bi_xy_std': 80,    # 更大的空间标准差，用于更好的区域连贯性
        'bi_rgb_std': 13,   # 更高的颜色标准差，以更容忍颜色变化
        'iterations': 10    # 增加迭代次数，以实现更好的收敛
    }
    
    # 获取所有CAM文件
    cam_files = [f for f in Path(cam_dir).glob("*.npy")]
    
    if not cam_files:
        print(f"Error: No .npy files found in {cam_dir}")
        return
    
    print(f"Found {len(cam_files)} CAM files")
    
    # 处理每个CAM文件
    success_count = 0
    for cam_file in cam_files:
        if process_cam_with_crf(cam_file, image_dir, output_dir, crf_config):
            success_count += 1
    
    print(f"Processing completed. Successfully processed {success_count}/{len(cam_files)} files.")
    print(f"Masks saved to {output_dir}/ directory")

if __name__ == "__main__":
    main() 