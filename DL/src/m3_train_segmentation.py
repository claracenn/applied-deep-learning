"""
M3_TRAIN_SEGMENTATION - 使用伪标签训练DeepLab-LargeFOV + ResNet50进行语义分割

本脚本实现了使用两种伪标签（带CRF和不带CRF）训练DeepLab-LargeFOV + ResNet50模型，
并计算两种方法在训练集和验证集上的mIoU性能指标。

输入:
    - outputs/pseudo_masks/*.png: CRF后处理的伪标签 (仅用于训练集)
    - outputs/base_pseudo/*.png: 不带CRF处理的伪标签 (仅用于训练集)
    - data/images/: 原始图像数据集
    - data/annotations/: 真实标签 (用于验证集)

输出:
    - models/segmentor/: 保存训练好的模型
    - outputs/results/: 保存评估结果和训练日志

    # 仅使用基础伪标签(不带CRF)
    python src/m3_train_segmentation.py --base_only

    # 仅使用CRF处理后的伪标签
    python src/m3_train_segmentation.py --crf_only
    
    
"""

import os
import time
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet50
from pathlib import Path
import argparse
import json
from datetime import datetime
from scipy import ndimage
import random

# 导入配置文件
from config import (
    SEGMENTATION_CONFIG, 
    SEGMENTATION_PATHS, 
    PROJECT_ROOT, 
    OUTPUT_ROOT,
    MODEL_ROOT,
    DATA_ROOT,
    IMAGE_DIR,
    ANNOTATION_DIR,
    PSEUDO_MASK_DIR,
    SEGMENTATION_DIR
)

# 导入utils中的分割函数
from utils import set_seed, get_dataloaders

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置
class SegmentationConfig:
    def __init__(self):
        # 使用配置文件中的路径
        self.img_dir = str(IMAGE_DIR)
        self.base_mask_dir = str(SEGMENTATION_DIR)
        self.crf_mask_dir = str(PSEUDO_MASK_DIR)
        self.gt_mask_dir = str(ANNOTATION_DIR)
        self.model_dir = str(MODEL_ROOT / "segmentor")
        self.result_dir = str(OUTPUT_ROOT / "results")
        
        # 从配置文件加载其他设置
        self.backbone = SEGMENTATION_CONFIG["backbone"]
        self.atrous_rates = SEGMENTATION_CONFIG["atrous_rates"]
        self.num_classes = SEGMENTATION_CONFIG["num_classes"]
        self.batch_size = SEGMENTATION_CONFIG["batch_size"]
        self.num_epochs = SEGMENTATION_CONFIG["num_epochs"]
        self.learning_rate = SEGMENTATION_CONFIG["learning_rate"]
        self.weight_decay = SEGMENTATION_CONFIG["weight_decay"]
        self.image_size = SEGMENTATION_CONFIG["image_size"]
        self.save_every = SEGMENTATION_CONFIG["save_every"]
        self.eval_every = SEGMENTATION_CONFIG["eval_every"]
        
        # 设置设备 - 简单方式
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 快速模式配置
        self.fast_mode = False
        self.fast_samples = 20  # 快速模式中每类使用的样本数
        self.fast_image_size = (128, 128)  # 快速模式中使用的图像尺寸
        self.fast_epochs = 1  # 快速模式中的epoch数

# 简单的进度打印函数，替代tqdm
def print_progress(current, total, prefix='', suffix='', decimals=1, length=30, fill='█'):
    """
    创建简单的命令行进度条
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if current == total:
        print()

# 数据集类
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, image_list=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_list = image_list  # 指定的图像列表
        
        # 获取所有有对应掩码的图像
        self.image_paths = []
        self.mask_paths = []
        
        # 支持的图像扩展名
        img_extensions = ['.jpg', '.jpeg', '.png']
        
        # 如果提供了图像列表，则只使用列表中的图像
        if image_list:
            for img_name in image_list:
                base_name = Path(img_name).stem  # 去掉扩展名和路径
                
                # 查找对应的掩码文件
                mask_path = self.mask_dir / f"{base_name}.png"
                if mask_path.exists():
                    # 查找对应的图像文件
                    for ext in img_extensions:
                        img_path = self.img_dir / f"{base_name}{ext}"
                        if img_path.exists():
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)
                            break
        else:
            # 检查mask_dir是否存在
            if not self.mask_dir.exists():
                raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")
                
            # 首先获取所有掩码文件
            mask_files = [f for f in self.mask_dir.glob('*.png')]
            
            # 对于每个掩码，查找对应的图像
            for mask_path in mask_files:
                base_name = mask_path.stem  # 去掉扩展名
                
                # 查找对应的图像文件
                for ext in img_extensions:
                    img_path = self.img_dir / f"{base_name}{ext}"
                    if img_path.exists():
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                        break
        
        # 预先验证图像，移除无效的图像
        valid_indices = []
        for i, (img_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            try:
                # 使用PIL尝试读取图像，确保它是有效的
                img = Image.open(str(img_path))
                if img is None:
                    print(f"Warning: Cannot read image {img_path}, skipping")
                    continue
                
                # 尝试读取掩码，确保它是有效的
                mask = Image.open(str(mask_path))
                if mask is None:
                    print(f"Warning: Cannot read mask {mask_path}, skipping")
                    continue
                
                valid_indices.append(i)
            except Exception as e:
                print(f"Error: Processing image {img_path} and mask {mask_path} failed: {e}")
                continue
        
        # 只保留有效的路径
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.mask_paths = [self.mask_paths[i] for i in valid_indices]
        
        print(f"Found {len(self.image_paths)} valid image-mask pairs from {mask_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            # 使用PIL读取图像和掩码
            image = Image.open(str(img_path)).convert('RGB')
            mask_img = Image.open(str(mask_path))
            
            # 如果掩码是彩色的，转换为灰度
            if mask_img.mode != 'L':
                mask_img = mask_img.convert('L')
            
            # 转换为NumPy数组进行处理
            image_np = np.array(image)
            mask = np.array(mask_img)
            
            # 检查掩码是否为空，输出调试信息
            if idx < 5:  # 只对前几个样本打印调试信息
                print(f"Mask {idx} statistics: min={mask.min()}, max={mask.max()}, unique values={np.unique(mask)}")
            
            # 处理掩码 - 增强版
            # 1. 检查是否是CAM风格的热图 (通常有多个强度值)
            unique_values = np.unique(mask)
            
            # 如果可能是CAM图像（不是二值图像）
            if len(unique_values) > 2 or (mask.max() > 1 and mask.min() < mask.max()):
                # CAM处理逻辑
                if idx < 5:
                    print(f"Mask {idx} might be a CAM heatmap (range: {mask.min()}-{mask.max()}, unique values: {len(unique_values)})")
                
                # 将掩码归一化到0-255范围
                if mask.max() <= 1:  # 可能是0-1归一化的
                    mask = (mask * 255).astype(np.uint8)
                
                # 使用简单的阈值法，而不是Otsu
                threshold = np.median(mask[mask > 0]) if np.any(mask > 0) else 127
                binary_mask = (mask > threshold).astype(np.uint8)
                
                # 应用简单的形态学操作
                binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3))).astype(np.uint8)
                binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((3,3))).astype(np.uint8)
                
                # 检查前景像素占比
                foreground_ratio = np.mean(binary_mask) * 100
                if idx < 5:
                    print(f"CAM binarized mask {idx}: foreground pixel ratio={foreground_ratio:.2f}%")
                
                # 如果前景像素太少，可能需要降低阈值
                if foreground_ratio < 1.0:  # 低于1%的前景
                    # 使用更低的固定阈值
                    binary_mask = (mask > mask.max() * 0.3).astype(np.uint8)
                    foreground_ratio = np.mean(binary_mask) * 100
                    if idx < 5:
                        print(f"After re-thresholding mask {idx}: foreground pixel ratio={foreground_ratio:.2f}%")
                
                mask = binary_mask
                
            # 如果掩码仍然没有前景像素
            if np.max(mask) == 0 or np.mean(mask) < 0.001:  # 几乎没有前景
                # 尝试反转
                if np.mean(mask) > 0.99:  # 几乎全是255/1
                    mask = 1 - mask
                    if idx < 5:
                        print(f"After inversion mask {idx}: foreground pixel ratio={np.mean(mask)*100:.2f}%")
                else:
                    # 最后手段：创建一个简单的中心区域掩码
                    h, w = mask.shape
                    new_mask = np.zeros_like(mask)
                    center_h, center_w = h//2, w//2
                    radius = min(h, w) // 3
                    y, x = np.ogrid[:h, :w]
                    # 创建圆形区域
                    mask_area = ((y - center_h)**2 + (x - center_w)**2) <= radius**2
                    new_mask[mask_area] = 1
                    mask = new_mask
                    if idx < 5:
                        print(f"Created circular center mask {idx}: foreground pixel ratio={np.mean(mask)*100:.2f}%")
            
            # 确保掩码为uint8类型
            mask = mask.astype(np.uint8)
            
            # 应用变换
            if self.transform:
                # 将numpy数组转回PIL图像用于torchvision变换
                image_pil = Image.fromarray(image_np)
                image = self.transform(image_pil)
            else:
                # 手动转换为tensor
                image = torch.from_numpy(image_np.transpose((2, 0, 1))).float() / 255.0
            
            # 处理掩码变换 - 修复版本
            if self.mask_transform:
                # 保证mask作为numpy数组传入mask_transform
                mask_tensor = self.mask_transform(mask)
                return {"image": image, "mask": mask_tensor, "path": str(img_path)}
            else:
                # 没有mask_transform时的直接转换
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.uint8)
                mask_tensor = torch.from_numpy(mask).long()
                return {"image": image, "mask": mask_tensor, "path": str(img_path)}
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回一个替代样本（空图像和掩码）
            dummy_image = torch.zeros(3, 224, 224)
            dummy_mask = torch.zeros((224, 224), dtype=torch.long)
            
            return {"image": dummy_image, "mask": dummy_mask, "path": str(img_path)}

# 快速模式数据集采样函数
def create_fast_mode_subset(dataset, num_samples):
    """
    从完整数据集中随机选择少量样本，用于快速模式
    """
    if len(dataset) <= num_samples:
        return dataset
    
    # 随机选择指定数量的样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)

# 掩码变换 - 修复: 确保二值掩码在ToTensor后不会变成浮点数0
class BinaryMaskToTensor:
    def __call__(self, mask):
        """
        将掩码转换为PyTorch tensor，处理多种可能的输入类型
        
        Args:
            mask: 可以是numpy数组或PyTorch tensor
            
        Returns:
            torch.Tensor: 二值掩码，形状为 [1, H, W] 或 [H, W]
        """
        # 1. 检查输入类型
        if isinstance(mask, torch.Tensor):
            # 如果已经是张量，只需确保类型正确
            mask_tensor = mask.long()
        else:
            # 确保掩码是二值的，值为0和1
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)
            
            # 转换为张量
            mask_tensor = torch.from_numpy(mask).long()
        
        # 2. 确保维度正确
        if mask_tensor.dim() == 2:
            # 有些操作需要通道维度
            mask_tensor_with_channel = mask_tensor.unsqueeze(0)
        else:
            mask_tensor_with_channel = mask_tensor
            
        return mask_tensor_with_channel

# 创建自定义的Resize转换，保持掩码值
class MaskResize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, mask):
        """
        对掩码进行缩放，保持掩码值，处理不同类型的输入
        
        Args:
            mask: 可以是numpy数组或PyTorch tensor
            
        Returns:
            torch.Tensor: 缩放后的二值掩码，形状为 [H, W]
        """
        # 1. 确保输入是PyTorch tensor
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).float()
        else:
            mask_tensor = mask.float()  # 确保是浮点类型用于插值
        
        # 2. 确保维度正确，需要[B, C, H, W]格式
        if mask_tensor.dim() == 2:  # [H, W]
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif mask_tensor.dim() == 3:  # [C, H, W]
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, C, H, W]
        
        # 3. 使用最近邻插值进行缩放，保持掩码值
        resized = F.interpolate(
            mask_tensor,
            size=self.size,
            mode='nearest'
        )
        
        # 4. 移除多余的维度，确保输出是[H, W]
        resized = resized.squeeze().long()  # 移除批次和通道维度
        
        # 如果只有一个像素，可能会挤压所有维度
        if resized.dim() == 0:
            resized = resized.unsqueeze(0).unsqueeze(0)  # 恢复为[1, 1]
            
        return resized  # 返回2D张量 [H, W]

# DeepLab-LargeFOV + ResNet50 模型
class DeepLabLargeFOV(nn.Module):
    def __init__(self, num_classes=2, atrous_rates=(6, 12, 18, 24), fast_mode=False):
        super(DeepLabLargeFOV, self).__init__()
        
        # 使用ResNet50作为骨干网络
        base = resnet50(pretrained=True)
        self.backbone_features = nn.Sequential(*list(base.children())[:-2])
        in_channels = 2048  # ResNet50的输出通道数
        
        # 快速模式下使用更小的ASPP
        if fast_mode:
            # 简化的ASPP，只使用更少的卷积分支
            self.aspp = ASPP(in_channels, atrous_rates[:2], fast_mode=True)
        else:
            # 完整的ASPP
            self.aspp = ASPP(in_channels, atrous_rates)
        
        # 分类器头部
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        input_size = x.size()
        
        # 经过骨干网络提取特征
        x = self.backbone_features(x)
        
        # 应用ASPP
        x = self.aspp(x)
        
        # 分类
        x = self.classifier(x)
        
        # 上采样到原始输入尺寸
        x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)
        
        return x

# ASPP模块 (Atrous Spatial Pyramid Pooling)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, fast_mode=False):
        super(ASPP, self).__init__()
        
        out_channels = 256
        
        # 1x1卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 空洞卷积分支
        self.conv_atrous = nn.ModuleList()
        for rate in atrous_rates:
            self.conv_atrous.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 合并所有分支
        # 快速模式下简化通道数
        if fast_mode:
            concat_channels = (len(atrous_rates) + 2) * out_channels
            self.conv_merge = nn.Sequential(
                nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)  # 减少dropout率加速收敛
            )
        else:
            concat_channels = (len(atrous_rates) + 2) * out_channels
            self.conv_merge = nn.Sequential(
                nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
    
    def forward(self, x):
        # 应用各个分支
        features = [self.conv1x1(x)]
        
        for conv in self.conv_atrous:
            features.append(conv(x))
        
        # 全局平均池化分支
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.size()[2:], mode='bilinear', align_corners=True)
        features.append(global_feat)
        
        # 合并所有特征
        x = torch.cat(features, dim=1)
        x = self.conv_merge(x)
        
        return x

# 评估函数：计算mIoU
def compute_miou(model, dataloader, device, num_classes=2):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # 前景类别像素计数
    foreground_pixel_count = 0
    total_pixel_count = 0
    
    with torch.no_grad():
        print("Evaluating...")
        total_batches = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            # 打印进度
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print_progress(batch_idx + 1, total_batches, prefix='Progress:', suffix='Complete')
            
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            paths = batch["path"]  # 获取图像路径，用于保存可视化结果
            
            # 统计前景像素比例
            if batch_idx < 5:  # 只检查前几个批次
                foreground_pixel_count += (masks == 1).sum().item()
                total_pixel_count += masks.numel()
            
            # 前向传播
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 更新混淆矩阵
            for pred, mask in zip(preds, masks):
                pred = pred.cpu().numpy().flatten()
                mask = mask.cpu().numpy().flatten()
                
                # 确保预测和真实标签中包含有效的类别值
                valid_mask = (mask < num_classes) & (mask >= 0)
                valid_pred = (pred < num_classes) & (pred >= 0)
                
                valid_indices = valid_mask & valid_pred
                if valid_indices.sum() > 0:
                    confusion_matrix += np.bincount(
                        num_classes * mask[valid_indices] + pred[valid_indices],
                        minlength=num_classes**2
                    ).reshape(num_classes, num_classes)
    
    # 打印前景像素占比统计
    if total_pixel_count > 0:
        foreground_percent = 100 * foreground_pixel_count / total_pixel_count
        print(f"Foreground pixels (class 1) percentage: {foreground_percent:.2f}%")
    
    # 计算IoU
    iou_per_class = np.zeros(num_classes)
    for i in range(num_classes):
        # 交集: TP
        intersection = confusion_matrix[i, i]
        # 并集: TP + FP + FN
        union = (confusion_matrix[i, :].sum() + confusion_matrix[:, i].sum() - intersection)
        
        if union > 0:
            iou_per_class[i] = intersection / union
    
    # 打印每个类别的IoU
    for i in range(num_classes):
        print(f"IoU for class {i}: {iou_per_class[i]:.4f}")
    
    # 直接对所有类别计算平均IoU，不再只取非零IoU
    # 二分类任务就是对背景和前景类取平均
    mean_iou = np.mean(iou_per_class)
    print(f"Average IoU across all classes: {mean_iou:.4f}")
    
    # 如果有类别IoU为0，额外报告
    zero_iou_classes = np.where(iou_per_class == 0)[0]
    if len(zero_iou_classes) > 0:
        print(f"Warning: Classes {zero_iou_classes} have IoU of 0")
    
    return mean_iou, iou_per_class

# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, config, mask_type="base"):
    device = config.device
    model.to(device)
    
    # 创建结果目录
    model_dir = Path(config.model_dir)
    result_dir = Path(config.result_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    result_dir.mkdir(exist_ok=True, parents=True)
    
    # 日志字典
    log = {
        "train_loss": [],
        "test_loss": [],
        "train_miou": [],
        "val_miou": [],  # 添加验证集mIoU
        "test_miou": [],
        "mask_type": mask_type,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
    }
    
    best_miou = 0.0
    start_time = time.time()
    
    # 从训练集中分出一小部分作为验证集
    train_size = int(0.9 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_subset, val_subset = random_split(
        train_loader.dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed if hasattr(config, 'seed') else 42)
    )
    
    # 创建验证集数据加载器
    val_loader = DataLoader(
        val_subset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers
    )
    
    # 使用剩余的训练数据创建新的训练加载器
    train_subset_loader = DataLoader(
        train_subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers
    )
    
    # 检查首个批次是否有前景类
    try:
        check_labels_batch = next(iter(train_subset_loader))
        all_masks = check_labels_batch["mask"]
        
        # 检查批次大小是否为0
        if all_masks.size(0) == 0:
            print("Error: Batch size is 0, please check the dataset or reduce batch size!")
            return model, log
            
        foreground_pixels = (all_masks == 1).sum().item()
        total_pixels = all_masks.numel()
        foreground_percent = 100 * foreground_pixels / total_pixels if total_pixels > 0 else 0
        
        print(f"Foreground pixel ratio in first batch: {foreground_percent:.2f}%")
        if foreground_percent < 0.1:
            print(f"Warning: Very low foreground pixel ratio ({foreground_percent:.2f}%), model may only learn to predict background!")
        
        if torch.max(all_masks) == 0:
            print(f"Severe warning: No foreground class detected in first batch, mask processing may be problematic!")
    except StopIteration:
        print("Error: Unable to get first batch, training set may be empty!")
        return model, log
    except Exception as e:
        print(f"Error checking batch: {e}")
    
    # 使用带权重的损失函数处理不平衡类别
    if config.num_classes == 2:
        # 估计类别权重来处理不平衡问题
        weight = torch.FloatTensor([0.1, 0.9]).to(device)  # 假设前景类(1)比背景类(0)少得多
        weighted_criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        weighted_criterion = criterion
    
    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        total_batches = len(train_subset_loader)
        
        for batch_idx, batch in enumerate(train_subset_loader):
            # 每隔一些批次打印进度
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print_progress(batch_idx + 1, total_batches, prefix='Training:', 
                              suffix=f'Loss: {train_loss/(batch_idx+1):.4f}')
            
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 使用带权重的损失函数
            loss = weighted_criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            train_loss += loss.item()
        
        # 计算平均训练损失
        train_loss /= len(train_subset_loader)
        log["train_loss"].append(train_loss)
        
        # 定期评估模型
        if (epoch + 1) % config.eval_every == 0:
            # 在训练集上评估
            train_miou, train_iou_per_class = compute_miou(model, train_subset_loader, device, config.num_classes)
            log["train_miou"].append(train_miou)
            
            # 在验证集上评估
            val_miou, val_iou_per_class = compute_miou(model, val_loader, device, config.num_classes)
            log["val_miou"].append(val_miou)
            
            # 在测试集上评估
            test_miou, test_iou_per_class = compute_miou(model, test_loader, device, config.num_classes)
            log["test_miou"].append(test_miou)
            
            print(f"Epoch {epoch+1}/{config.num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train mIoU: {train_miou:.4f}, "
                  f"Val mIoU: {val_miou:.4f}, "
                  f"Test mIoU: {test_miou:.4f}")
            
            # 添加每个类别的IoU到日志
            if epoch == 0 or (epoch + 1) % 5 == 0:  # 每5个epoch记录一次详细IoU
                class_ious = {}
                for i in range(config.num_classes):
                    class_ious[f"class_{i}_iou"] = float(test_iou_per_class[i])
                print(f"Class IoU details: {class_ious}")
            
            # 保存最佳模型（使用验证集mIoU作为指标）
            if val_miou > best_miou:
                best_miou = val_miou
                model_path = model_dir / f"best_model_{mask_type}.pth"
                torch.save(model.state_dict(), str(model_path))
                print(f"Saving best model, Val mIoU: {best_miou:.4f}")
            
            # 使用验证集mIoU来调整学习率
            scheduler.step(val_miou)
    
    # 保存训练日志
    training_time = time.time() - start_time
    log["training_time"] = training_time
    log["best_miou"] = best_miou
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = result_dir / f"training_log_{mask_type}_{timestamp}.json"
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
    
    print(f"Training completed in {training_time/60:.2f} minutes. Best Val mIoU: {best_miou:.4f}")
    
    return model, log

# 添加加载官方数据集划分的函数
def load_official_dataset_split(trainval_file, test_file, test_ratio=0.2, seed=42):
    """
    从Oxford-IIIT Pet数据集官方划分文件加载训练和测试集，
    从test.txt中随机选择test_ratio比例的样本作为测试集
    
    Args:
        trainval_file: 训练集和验证集列表文件路径
        test_file: 测试集列表文件路径
        test_ratio: 从test.txt中选择的比例作为测试集
        seed: 随机种子，确保测试集选择的可重现性
        
    Returns:
        train_images, test_images: 训练集和测试集的图像名称列表
    """
    trainval_path = Path(trainval_file)
    test_path = Path(test_file)
    
    if not trainval_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Dataset split files not found: {trainval_path} or {test_path}")
    
    # 读取trainval.txt文件作为训练集
    train_images = []
    with open(trainval_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  # 跳过注释和空行
            parts = line.strip().split()
            if parts:
                # 格式: Image CLASS-ID SPECIES BREED ID
                image_name = parts[0]
                train_images.append(image_name)
    
    # 读取test.txt文件
    all_test_images = []
    with open(test_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  # 跳过注释和空行
            parts = line.strip().split()
            if parts:
                image_name = parts[0]
                all_test_images.append(image_name)
    
    # 设置随机种子以确保可重现性
    random.seed(seed)
    
    # 随机选择test_ratio比例的样本作为测试集
    test_size = int(len(all_test_images) * test_ratio)
    # 随机打乱后选择前test_size个
    random.shuffle(all_test_images)
    test_images = all_test_images[:test_size]
    
    print(f"Dataset split loaded: {len(train_images)} training images, {len(test_images)} test images")
    print(f"Test set is {test_ratio*100:.1f}% of test.txt file, using random seed {seed}")
    return train_images, test_images

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate segmentation models with different pseudo masks')
    parser.add_argument('--base_only', action='store_true', help='Only train and evaluate with base pseudo masks')
    parser.add_argument('--crf_only', action='store_true', help='Only train and evaluate with CRF pseudo masks')
    parser.add_argument('--fast_mode', action='store_true', help='Run in fast mode to verify model functionality')
    parser.add_argument('--epochs', type=int, default=SEGMENTATION_CONFIG["num_epochs"], help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=SEGMENTATION_CONFIG["batch_size"], help='Batch size for training')
    parser.add_argument('--lr', type=float, default=SEGMENTATION_CONFIG["learning_rate"], help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of test.txt to use as test set')
    parser.add_argument('--num_workers', type=int, default=SEGMENTATION_CONFIG["num_workers"], help='Number of data loading workers')
    parser.add_argument('--crf_dir', type=str, default=None, help='Directory with CRF preset masks to use (e.g., outputs/pseudo_masks/preset_A)')
    parser.add_argument('--trimap_dir', type=str, default=str(DATA_ROOT / 'annotations/trimaps'), help='真实标签(trimap)目录')
    parser.add_argument('--trainval_file', type=str, default=str(DATA_ROOT / 'annotations/trainval.txt'), help='训练集列表文件')
    parser.add_argument('--test_file', type=str, default=str(DATA_ROOT / 'annotations/test.txt'), help='测试集列表文件')
    args = parser.parse_args()
    
    # 确保所有必要的目录都存在
    print("检查并创建必要的目录...")
    for path in [
        Path(SEGMENTATION_DIR),  # 基础伪标签
        Path(PSEUDO_MASK_DIR),   # CRF处理后的伪标签
        Path(MODEL_ROOT / "segmentor"),  # 模型保存目录
        Path(OUTPUT_ROOT / "results"),   # 结果保存目录
    ]:
        if not path.exists():
            print(f"创建目录: {path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"目录已存在: {path}")
    
    # 检查真实标签目录
    trimap_dir = Path(args.trimap_dir)
    if not trimap_dir.exists() or not any(trimap_dir.glob('*.png')):
        print(f"警告: 真实标签目录 {trimap_dir} 不存在或为空")
    else:
        print(f"找到真实标签目录: {trimap_dir}, 包含 {len(list(trimap_dir.glob('*.png')))} 个标签文件")
    
    # 检查伪标签目录是否包含PNG文件
    base_masks = list(Path(SEGMENTATION_DIR).glob('*.png'))
    crf_masks = list(Path(PSEUDO_MASK_DIR).glob('*.png'))
    
    print(f"基础伪标签数量: {len(base_masks)}")
    print(f"CRF处理后伪标签数量: {len(crf_masks)}")
    
    if len(base_masks) == 0 and not args.crf_only:
        print("警告: 基础伪标签目录为空，可能无法训练基础模型")
    
    if len(crf_masks) == 0 and not args.base_only:
        print("警告: CRF处理后伪标签目录为空，可能无法训练CRF模型")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化配置
    config = SegmentationConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.gt_mask_dir = args.trimap_dir  # 设置真实标签目录
    
    # 设置数据加载器线程数
    num_workers = 0 if args.fast_mode else args.num_workers  # 快速模式下使用0个worker以避免多进程问题
    
    # 快速模式设置
    if args.fast_mode:
        print("\n===== Running in FAST MODE to verify model functionality =====")
        config.fast_mode = True
        config.num_epochs = config.fast_epochs
        config.image_size = config.fast_image_size
        config.batch_size = 4  # 减小批量大小
        config.learning_rate = 5e-4  # 增大学习率加速收敛
        config.save_every = config.num_epochs  # 只在最后保存一次
        config.eval_every = 1  # 每个epoch都评估
    
    # 使用指定的CRF预设掩码
    if args.crf_dir and Path(args.crf_dir).exists() and any(Path(args.crf_dir).glob('*.png')):
        print(f"\n===== Using CRF preset masks from {args.crf_dir} =====")
        if not args.base_only:
            config.crf_mask_dir = args.crf_dir
            print(f"Will use CRF preset masks from: {args.crf_dir}")
    
    # 创建结果目录
    Path(config.model_dir).mkdir(exist_ok=True, parents=True)
    Path(config.result_dir).mkdir(exist_ok=True, parents=True)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    mask_transform = transforms.Compose([
        BinaryMaskToTensor(),
        MaskResize(config.image_size)
    ])
    
    # 使用官方数据集划分，从test.txt中选择一部分作为测试集
    print("加载官方数据集划分...")
    train_images, test_images = load_official_dataset_split(
        args.trainval_file,
        args.test_file,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 创建结果汇总
    results = {"base": {}, "crf": {}}
    
    # 训练使用基础掩码的模型
    if not args.crf_only:
        print("\n===== Training with base pseudo masks =====")
        # 创建训练集 - 使用基础伪标签
        train_dataset = SegmentationDataset(
            config.img_dir, 
            config.base_mask_dir, 
            transform, 
            mask_transform,
            image_list=train_images  # 只使用训练集图像
        )
        
        # 创建测试集 - 使用真实标签
        test_dataset = SegmentationDataset(
            config.img_dir, 
            config.gt_mask_dir,  # 使用真实标签(trimaps)
            transform, 
            mask_transform,
            image_list=test_images  # 只使用测试集图像
        )
        
        # 快速模式下只使用少量样本
        if config.fast_mode:
            train_dataset = create_fast_mode_subset(train_dataset, config.fast_samples)
            test_dataset = create_fast_mode_subset(test_dataset, config.fast_samples // 5)
            print(f"Fast mode: Using {len(train_dataset)} samples for training and {len(test_dataset)} for testing")
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
        
        # 创建模型、损失函数和优化器
        model_base = DeepLabLargeFOV(num_classes=config.num_classes, atrous_rates=config.atrous_rates, 
                                     fast_mode=config.fast_mode)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model_base.parameters(), lr=config.learning_rate, 
                                     weight_decay=1e-3)
        
        # 训练模型
        model_base, log_base = train_model(model_base, train_loader, test_loader, criterion, optimizer, config, "base")
        
        # 保存结果
        results["base"] = {
            "train_miou": log_base["train_miou"][-1] if log_base["train_miou"] else None,
            "test_miou": log_base["test_miou"][-1] if log_base["test_miou"] else None,
            "best_miou": log_base["best_miou"] if "best_miou" in log_base else None
        }
    
    # 训练使用CRF掩码的模型
    if not args.base_only:
        print("\n===== Training with CRF pseudo masks =====")
        # 创建训练集 - 使用CRF伪标签
        train_dataset = SegmentationDataset(
            config.img_dir, 
            config.crf_mask_dir, 
            transform, 
            mask_transform,
            image_list=train_images  # 只使用训练集图像
        )
        
        # 创建测试集 - 使用真实标签
        test_dataset = SegmentationDataset(
            config.img_dir, 
            config.gt_mask_dir,  # 使用真实标签(trimaps)
            transform, 
            mask_transform,
            image_list=test_images  # 只使用测试集图像
        )
        
        # 快速模式下只使用少量样本
        if config.fast_mode:
            train_dataset = create_fast_mode_subset(train_dataset, config.fast_samples)
            test_dataset = create_fast_mode_subset(test_dataset, config.fast_samples // 5)
            print(f"Fast mode: Using {len(train_dataset)} samples for training and {len(test_dataset)} for testing")
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
        
        # 创建模型、损失函数和优化器
        model_crf = DeepLabLargeFOV(num_classes=config.num_classes, atrous_rates=config.atrous_rates,
                                   fast_mode=config.fast_mode)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model_crf.parameters(), lr=config.learning_rate, 
                                     weight_decay=1e-3)
        
        # 训练模型
        model_crf, log_crf = train_model(model_crf, train_loader, test_loader, criterion, optimizer, config, "crf")
        
        # 保存结果
        results["crf"] = {
            "train_miou": log_crf["train_miou"][-1] if log_crf["train_miou"] else None,
            "test_miou": log_crf["test_miou"][-1] if log_crf["test_miou"] else None,
            "best_miou": log_crf["best_miou"] if "best_miou" in log_crf else None
        }
    
    # 打印结果对比
    print("\n===== Results =====")
    print(f"Base Pseudo Masks - Train mIoU: {results['base'].get('train_miou', 'N/A')}")
    print(f"Base Pseudo Masks - Val mIoU: {results['base'].get('val_miou', 'N/A')}")
    print(f"Base Pseudo Masks - Test mIoU: {results['base'].get('test_miou', 'N/A')}")
    print(f"CRF Pseudo Masks - Train mIoU: {results['crf'].get('train_miou', 'N/A')}")
    print(f"CRF Pseudo Masks - Val mIoU: {results['crf'].get('val_miou', 'N/A')}")
    print(f"CRF Pseudo Masks - Test mIoU: {results['crf'].get('test_miou', 'N/A')}")
    
    # 保存结果对比
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(config.result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存详细的JSON结果
    comparison_path = result_dir / f"comparison_results_{timestamp}.json"
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存CSV格式结果，包含训练集、验证集和测试集的mIoU
    csv_path = result_dir / f"miou_comparison_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        # 写入表头
        f.write("Method,Train_mIoU,Val_mIoU,Test_mIoU\n")
        # 写入基础方法结果
        f.write(f"Base,{results['base'].get('train_miou', 0):.4f}," +
                f"{results['base'].get('val_miou', 0):.4f}," +
                f"{results['base'].get('test_miou', 0):.4f}\n")
        # 写入CRF方法结果
        f.write(f"CRF,{results['crf'].get('train_miou', 0):.4f}," +
                f"{results['crf'].get('val_miou', 0):.4f}," +
                f"{results['crf'].get('test_miou', 0):.4f}\n")
    
    print(f"Results comparison saved to {comparison_path}")
    print(f"mIoU comparison CSV saved to {csv_path}")

if __name__ == "__main__":
    main() 