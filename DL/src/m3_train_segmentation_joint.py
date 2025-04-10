"""
M3_TRAIN_SEGMENTATION_JOINT - 联合训练分割模型和ScoreNet

本脚本实现了分割模型和ScoreNet的端到端联合训练。
训练过程中，ScoreNet会动态生成置信度图，用于调整伪标签的质量。

输入:
    - outputs/cams/*.npy: CAM特征图
    - outputs/base_pseudo/*.png: 初始伪标签
    - data/images/: 原始图像数据集

输出:
    - models/joint/: 保存训练好的联合模型
    - outputs/results/joint/: 保存评估结果和训练日志
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from datetime import datetime
import json
from PIL import Image
import torchvision.transforms as transforms
import random
from scipy import ndimage

from models_joint import JointSegmentationModel
from joint_loss import JointLoss
from m3_train_segmentation import (
    set_seed, print_progress, compute_miou, load_official_dataset_split,
    create_fast_mode_subset, SegmentationDataset
)

# 掩码变换 - 确保二值掩码在ToTensor后不会变成浮点数0
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

class JointConfig:
    """联合训练配置"""
    def __init__(self):
        # 数据路径
        self.img_dir = "data/images"
        self.cam_dir = "outputs/cams"
        self.initial_mask_dir = "outputs/base_pseudo"
        self.model_dir = "models/joint"
        self.result_dir = "outputs/results/joint"
        self.trainval_file = "data/annotations/trainval.txt"
        self.test_file = "data/annotations/test.txt"
        self.gt_mask_dir = "data/annotations/trimaps"  # 添加真实标签路径
        
        # 模型配置
        self.num_classes = 2
        self.score_channels = 16  # 更合理的通道数
        self.atrous_rates = (6, 12, 18, 24)  # 与DeepLabLargeFOV相同
        
        # 训练配置
        self.batch_size = 8
        self.num_epochs = 5  # 与m3_train_segmentation一致
        self.learning_rate = 1e-4  # 与m3_train_segmentation一致
        self.weight_decay = 1e-4  # 与m3_train_segmentation一致
        self.test_ratio = 0.2   # 测试集比例
        self.image_size = (224, 224)
        self.num_workers = 4
        
        # 损失函数配置 - 保持原来的权重
        self.use_dice = True
        self.dice_weight = 0.8     # 略微减小Dice损失权重
        self.score_smoothness_weight = 0.02  # 增加平滑度权重
        self.score_sparsity_weight = 0.03    # 降低稀疏性权重
        self.score_reg_weight = 0.01  # 大幅降低
        
        # 类别权重
        self.seg_weights = [0.3, 0.7]  # 确保关注前景类别
        
        # 评估配置
        self.eval_every = 1
        self.save_every = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

class JointDataset(Dataset):
    """联合训练数据集，同时加载图像、CAM和初始伪标签，使用与m3_train_segmentation.py相同的处理方法"""
    def __init__(self, img_dir, cam_dir, mask_dir, transform=None, mask_transform=None, image_list=None):
        self.img_dir = Path(img_dir)
        self.cam_dir = Path(cam_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_list = image_list  # 指定的图像列表
        
        # 获取所有匹配的文件
        self.image_paths = []
        self.cam_paths = []
        self.mask_paths = []
        
        # 支持的图像扩展名
        img_extensions = ['.jpg', '.jpeg', '.png']
        
        # 如果提供了图像列表，则只使用列表中的图像
        if image_list:
            for img_name in image_list:
                base_name = Path(img_name).stem  # 去掉扩展名和路径
                
                # 查找对应的掩码文件
                mask_path = self.mask_dir / f"{base_name}.png"
                # 查找对应的CAM文件
                cam_path = self.cam_dir / f"{base_name}_cam.npy"
                
                if mask_path.exists() and cam_path.exists():
                    # 查找对应的图像文件
                    for ext in img_extensions:
                        img_path = self.img_dir / f"{base_name}{ext}"
                        if img_path.exists():
                            self.image_paths.append(img_path)
                            self.cam_paths.append(cam_path)
                            self.mask_paths.append(mask_path)
                            break
        else:
            # 检查mask_dir是否存在
            if not self.mask_dir.exists():
                raise FileNotFoundError(f"掩码目录不存在: {self.mask_dir}")
                
            # 首先获取所有掩码文件
            mask_files = [f for f in self.mask_dir.glob('*.png')]
            
            # 对于每个掩码，查找对应的图像和CAM
            for mask_path in mask_files:
                base_name = mask_path.stem  # 去掉扩展名
                
                # 查找对应的CAM文件
                cam_path = self.cam_dir / f"{base_name}_cam.npy"
                if not cam_path.exists():
                    continue
                
                # 查找对应的图像文件
                for ext in img_extensions:
                    img_path = self.img_dir / f"{base_name}{ext}"
                    if img_path.exists():
                        self.image_paths.append(img_path)
                        self.cam_paths.append(cam_path)
                        self.mask_paths.append(mask_path)
                        break
        
        # 预先验证图像，移除无效的图像
        valid_indices = []
        for i, (img_path, cam_path, mask_path) in enumerate(zip(self.image_paths, self.cam_paths, self.mask_paths)):
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
                
                # 尝试读取CAM，确保它是有效的
                cam = np.load(str(cam_path))
                if cam is None or cam.size == 0:
                    print(f"Warning: Cannot read CAM {cam_path}, skipping")
                    continue
                
                valid_indices.append(i)
            except Exception as e:
                print(f"Error: Processing {img_path}, {mask_path}, {cam_path} failed: {e}")
                continue
        
        # 只保留有效的路径
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.cam_paths = [self.cam_paths[i] for i in valid_indices]
        self.mask_paths = [self.mask_paths[i] for i in valid_indices]
        
        print(f"Found {len(self.image_paths)} valid image-mask-cam triplets")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # 加载图像和掩码
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            cam_path = self.cam_paths[idx]
            
            # 使用PIL读取图像和掩码
            image = Image.open(str(img_path)).convert('RGB')
            mask_img = Image.open(str(mask_path))
            
            # 如果掩码是彩色的，转换为灰度
            if mask_img.mode != 'L':
                mask_img = mask_img.convert('L')
            
            # 加载CAM
            cam = np.load(str(cam_path))
            
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
            
            # 处理CAM数据
            # 确保CAM是2D数组
            if len(cam.shape) != 2:
                if len(cam.shape) > 2:
                    cam = cam[:, :, 0]  # 如果是多通道，只取第一个通道
            
            # 确保CAM值在[0, 1]范围内
            if cam.max() > 1.0:
                cam = cam / 255.0 if cam.max() > 100 else cam / cam.max()
            
            # 应用变换
            if self.transform:
                # 将numpy数组转回PIL图像用于torchvision变换
                image_pil = Image.fromarray(image_np)
                image = self.transform(image_pil)
            else:
                # 手动转换为tensor
                image = torch.from_numpy(image_np.transpose((2, 0, 1))).float() / 255.0
            
            # 处理掩码变换
            if self.mask_transform:
                # 保证mask作为numpy数组传入mask_transform
                mask_tensor = self.mask_transform(mask)
            else:
                # 没有mask_transform时的直接转换
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.uint8)
                mask_tensor = torch.from_numpy(mask).long()
            
            # 调整CAM大小以匹配图像
            target_size = (image.shape[2], image.shape[1])  # (W, H)
            cam_pil = Image.fromarray(cam)
            cam_pil = cam_pil.resize(target_size, Image.BILINEAR)
            cam = np.array(cam_pil)
            cam_tensor = torch.from_numpy(cam).float().unsqueeze(0)  # [1, H, W]
            
            return {
                "image": image,
                "cam": cam_tensor,
                "mask": mask_tensor,
                "path": str(img_path)
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回一个替代样本（空图像和掩码）
            dummy_size = (224, 224)
            dummy_image = torch.zeros(3, *dummy_size)
            dummy_mask = torch.zeros(dummy_size, dtype=torch.long)
            dummy_cam = torch.zeros(1, *dummy_size, dtype=torch.float)
            
            return {
                "image": dummy_image,
                "cam": dummy_cam,
                "mask": dummy_mask,
                "path": "invalid_sample"
            }

def update_pseudo_labels(initial_masks, conf_map, threshold=0.05):
    """
    基于置信度图动态更新伪标签 - 修复版
    
    Args:
        initial_masks (torch.Tensor): 初始伪标签 [B, H, W]
        conf_map (torch.Tensor): ScoreNet生成的置信度图 [B, 1, H, W]
        threshold (float): 置信度阈值，默认0.05 (使用相对保守的阈值)
    
    Returns:
        torch.Tensor: 更新后的伪标签 [B, H, W]
    """
    # 确保conf_map的维度正确 [B, 1, H, W] -> [B, H, W]
    if conf_map.dim() == 4:
        conf_map = conf_map.squeeze(1)
    
    # 复制初始掩码
    refined_masks = initial_masks.clone()
    
    # 第一阶段优先保留原始前景
    # 仅过滤掉非常低置信度的区域，阈值设置极低
    very_low_conf = (conf_map < threshold)
    refined_masks[very_low_conf & (initial_masks == 1)] = 0
    
    # 确保每个样本中至少保留一些前景像素
    batch_size = initial_masks.size(0)
    for i in range(batch_size):
        # 检查当前样本是否有足够的前景像素
        foreground_ratio = (refined_masks[i] == 1).float().mean()
        initial_foreground_ratio = (initial_masks[i] == 1).float().mean()
        
        # 如果前景像素比例太低，恢复一些原始掩码中的前景
        if foreground_ratio < 0.01 and initial_foreground_ratio > 0:
            # 对于前景比例低于1%的样本，至少保留原始掩码的30%前景区域
            conf_values, _ = torch.sort(conf_map[i].flatten(), descending=True)
            dynamic_threshold = conf_values[int(len(conf_values) * 0.3)]
            high_conf = conf_map[i] >= dynamic_threshold
            
            # 恢复一些原始前景
            refined_masks[i] = torch.where(
                high_conf & (initial_masks[i] == 1),
                torch.ones_like(refined_masks[i]),
                refined_masks[i]
            )
    
    return refined_masks

# 添加一个专门处理trimaps真实标签的函数
def compute_miou_with_trimaps(model, dataloader, device, num_classes=2):
    """
    计算mIoU时专门处理trimaps格式的真实标签（1:前景，2:背景，3:不确定区域）
    
    Args:
        model: 分割模型
        dataloader: 数据加载器
        device: 运行设备
        num_classes: 分类数量
        
    Returns:
        float: 平均IoU
        np.ndarray: 每个类别的IoU
    """
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
            
            # 检查是否是trimaps格式（值为1,2,3）
            is_trimaps_format = False
            unique_values = torch.unique(masks)
            if 2 in unique_values or 3 in unique_values:
                is_trimaps_format = True
                
            if is_trimaps_format:
                # 转换trimaps: 1->1(前景), 2->0(背景), 3->忽略
                valid_mask = (masks != 3)  # 忽略值为3的区域
                
                # 创建新掩码: 背景(2)映射为0，前景(1)映射为1
                converted_masks = torch.zeros_like(masks)
                converted_masks[masks == 1] = 1  # 前景
                # 背景(2)默认为0，所以不需要额外操作
                
                # 使用转换后的掩码和有效区域
                masks = converted_masks
            
            # 统计前景像素比例
            if batch_idx < 5:  # 只检查前几个批次
                foreground_pixel_count += (masks == 1).sum().item()
                total_pixel_count += masks.numel()
            
            # 前向传播
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 更新混淆矩阵 - 支持trimaps
            for pred, mask in zip(preds, masks):
                pred = pred.cpu().numpy().flatten()
                mask = mask.cpu().numpy().flatten()
                
                # 如果是trimaps格式，忽略标记为3的区域
                if is_trimaps_format:
                    valid_indices = (mask != 3)
                    pred = pred[valid_indices]
                    mask = mask[valid_indices]
                
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
    
    # 计算平均IoU
    mean_iou = np.mean(iou_per_class)
    print(f"Average IoU across all classes: {mean_iou:.4f}")
    
    # 如果有类别IoU为0，额外报告
    zero_iou_classes = np.where(iou_per_class == 0)[0]
    if len(zero_iou_classes) > 0:
        print(f"Warning: Classes {zero_iou_classes} have IoU of 0")
    
    return mean_iou, iou_per_class

def train_joint_model(model, train_loader, val_loader, test_loader, criterion, optimizer, config):
    """
    联合训练函数
    
    Args:
        model: 联合分割模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        config: 配置信息
    """
    device = config.device
    model.to(device)
    criterion.to(device)  # 确保损失函数也在正确的设备上
    
    # 创建输出目录
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # 日志字典
    log = {
        "train_loss": [],
        "test_miou": [],
        "train_miou": [],
        "val_miou": [],  # 添加验证集mIoU
        "loss_components": [],
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
    }
    
    best_miou = 0.0
    start_time = time.time()
    
    # 学习率调度器 - 与m3_train_segmentation一致
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=2, factor=0.5
    )
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        epoch_loss_components = {"seg": 0.0, "score_reg": 0.0}
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            images = batch["image"].to(device)
            initial_masks = batch["mask"].to(device)
            cams = batch["cam"].to(device)
            
            # 1. 使用当前的ScoreNet生成置信度图
            seg_pred, conf_map = model(images, cams)
            
            # 2. 基于置信度图动态更新伪标签
            with torch.no_grad():
                refined_masks = update_pseudo_labels(initial_masks, conf_map)
            
            # 3. 使用更新后的伪标签计算损失
            loss, loss_dict = criterion(seg_pred, refined_masks, conf_map)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            epoch_losses.append(loss_dict['total_loss'])
            epoch_loss_components['seg'] += loss_dict['seg_loss']
            epoch_loss_components['score_reg'] += loss_dict['score_reg_loss']
            
            # 打印进度
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                print_progress(batch_idx + 1, total_batches, prefix='Training:',
                             suffix=f'Loss: {avg_loss:.4f}')
        
        # 计算平均损失
        if epoch_losses:  # 确保有有效的损失值
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            # 使用总批次数来计算平均组件损失
            avg_components = {k: v / total_batches for k, v in epoch_loss_components.items()}
            
            log["train_loss"].append(avg_epoch_loss)
            log["loss_components"].append(avg_components)
        
        # 评估模型
        if (epoch + 1) % config.eval_every == 0:
            # 在训练集上评估
            model.eval()
            with torch.no_grad():
                # 对训练集，仍然使用原始的compute_miou
                train_miou, train_iou_per_class = compute_miou(
                    model.segmentor, train_loader, device, config.num_classes
                )
            log["train_miou"].append(train_miou)
            
            # 在验证集上评估
            with torch.no_grad():
                val_miou, val_iou_per_class = compute_miou(
                    model.segmentor, val_loader, device, config.num_classes
                )
            log["val_miou"].append(val_miou)
            
            # 在测试集上评估 - 使用支持trimaps的评估函数
            with torch.no_grad():
                # 使用支持trimaps的特殊评估函数
                test_miou, test_iou_per_class = compute_miou_with_trimaps(
                    model.segmentor, test_loader, device, config.num_classes
                )
            log["test_miou"].append(test_miou)
            
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Train Loss: {avg_epoch_loss:.4f}")
            print(f"Train mIoU: {train_miou:.4f}")
            print(f"Val mIoU: {val_miou:.4f}")  # 打印验证集mIoU
            print(f"Test mIoU: {test_miou:.4f}")
            print("Loss Components:")
            for k, v in avg_components.items():
                print(f"  {k}: {v:.4f}")
            
            # 保存最佳模型 - 基于验证集mIoU
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(),
                         os.path.join(config.model_dir, "best_model_joint.pth"))
                print(f"Saved best model with Val mIoU: {best_miou:.4f}")
            
            # 使用验证集mIoU来调整学习率，而不是测试集
            scheduler.step(val_miou)
    
    # 保存训练日志
    training_time = time.time() - start_time
    log["training_time"] = training_time
    log["best_miou"] = best_miou
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.result_dir, f"training_log_joint_{timestamp}.json")
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
    
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best Val mIoU: {best_miou:.4f}")
    
    # 打印最终的train、val、test三个miou结果
    if log["train_miou"]:
        print("\n最终评估结果:")
        print(f"Train mIoU: {log['train_miou'][-1]:.4f}")
        print(f"Val mIoU: {log['val_miou'][-1]:.4f}")
        print(f"Test mIoU: {log['test_miou'][-1]:.4f}")
    
    print(f"Training log saved to {log_path}")
    
    return model, log

def main():
    parser = argparse.ArgumentParser(description='Joint training of segmentation model and ScoreNet')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of test.txt to use as test set')
    parser.add_argument('--trainval_file', type=str, default="data/annotations/trainval.txt", help='训练集列表文件')
    parser.add_argument('--test_file', type=str, default="data/annotations/test.txt", help='测试集列表文件')
    parser.add_argument('--pretrained_segmentor', type=str, help='Path to pretrained segmentor')
    parser.add_argument('--pretrained_scorenet', type=str, help='Path to pretrained scorenet')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--trimap_dir', type=str, default="data/annotations/trimaps", help='真实标签(trimap)目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化配置
    config = JointConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.test_ratio = args.test_ratio
    config.trainval_file = args.trainval_file
    config.test_file = args.test_file
    config.num_workers = args.num_workers
    config.gt_mask_dir = args.trimap_dir  # 添加真实标签路径
    
    # 创建必要的目录
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # 检查并更新路径
    for path_name, path_value in [
        ('img_dir', config.img_dir), 
        ('cam_dir', config.cam_dir), 
        ('initial_mask_dir', config.initial_mask_dir),
        ('gt_mask_dir', config.gt_mask_dir),
        ('trainval_file', config.trainval_file),
        ('test_file', config.test_file)
    ]:
        # 检查路径是否存在
        if not os.path.exists(path_value):
            # 尝试添加DL前缀
            dl_path = os.path.join("DL", path_value)
            if os.path.exists(dl_path):
                print(f"找到替代路径: {dl_path} 替换 {path_value}")
                setattr(config, path_name, dl_path)
            else:
                print(f"警告: 路径不存在: {path_value}")
    
    # 使用官方数据集划分，从test.txt中选择一部分作为测试集
    print("\n===== 加载官方数据集划分 =====")
    train_images, test_images = load_official_dataset_split(
        config.trainval_file, 
        config.test_file,
        test_ratio=config.test_ratio,
        seed=args.seed
    )
    
    # 创建与m3_train_segmentation相同的数据变换
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
    
    # 创建训练和测试数据集
    train_dataset = JointDataset(
        config.img_dir,
        config.cam_dir,
        config.initial_mask_dir,
        transform=transform,
        mask_transform=mask_transform,
        image_list=train_images
    )
    
    # 使用真实标签目录作为测试集的真值来源
    print(f"\n===== 使用真实标签进行测试评估 =====")
    print(f"真实标签目录: {config.gt_mask_dir}")
    
    # 检查真实标签目录是否存在
    if not os.path.exists(config.gt_mask_dir):
        print(f"警告: 真实标签目录不存在: {config.gt_mask_dir}")
        # 尝试查找常见的trimaps路径
        possible_paths = [
            "DL/data/annotations/trimaps",
            "data/annotations/trimaps",
            "annotations/trimaps",
            "DL/annotations/trimaps"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config.gt_mask_dir = path
                print(f"找到可用的真实标签目录: {path}")
                break
    
    # 检查真实标签文件数量
    try:
        trimaps_count = len(list(Path(config.gt_mask_dir).glob('*.png')))
        print(f"真实标签文件数量: {trimaps_count}")
    except:
        print("无法读取真实标签文件")
        trimaps_count = 0
    
    # 专用于测试的数据集，使用真实标签而不是伪标签
    test_dataset = SegmentationDataset(
        config.img_dir,
        config.gt_mask_dir,  # 使用真实标签目录
        transform,
        mask_transform,
        image_list=test_images
    )
    
    # 如果测试集为空，可能是找不到真实标签，尝试使用训练集分割
    if len(test_dataset) == 0:
        print(f"警告: 使用真实标签的测试集为空，尝试从训练集分离一部分作为测试集")
        # 创建一个使用伪标签的训练数据集
        train_seg_dataset = SegmentationDataset(
            config.img_dir,
            config.initial_mask_dir,
            transform,
            mask_transform,
            image_list=train_images
        )
        
        # 从训练集中分割测试集
        test_size = int(0.2 * len(train_seg_dataset))
        train_size = len(train_seg_dataset) - test_size
        _temp_train, test_dataset = torch.utils.data.random_split(
            train_seg_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"从训练数据集分离出的测试集大小: {len(test_dataset)}")
    
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 使用与m3_train_segmentation.py相同比例划分数据集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # 打印数据加载器信息
    print(f"\n数据加载器信息:")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    model = JointSegmentationModel(
        num_classes=config.num_classes,
        score_channels=config.score_channels,
        atrous_rates=config.atrous_rates
    )
    
    # 加载预训练模型（如果有）
    if args.pretrained_segmentor or args.pretrained_scorenet:
        model.load_pretrained(args.pretrained_segmentor, args.pretrained_scorenet)
    
    # 创建损失函数
    criterion = JointLoss(
        num_classes=config.num_classes,
        seg_weights=config.seg_weights,  # 使用配置的类别权重
        use_dice=config.use_dice,
        dice_weight=config.dice_weight,
        score_smoothness_weight=config.score_smoothness_weight,
        score_sparsity_weight=config.score_sparsity_weight,
        score_reg_weight=config.score_reg_weight
    )
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    
    # 训练模型
    model, log = train_joint_model(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, config
    )

if __name__ == "__main__":
    main() 