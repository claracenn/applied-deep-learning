
"""
M3_TRAIN_SEGMENTATION - 使用伪标签训练DeepLab-LargeFOV + ResNet50进行语义分割

本脚本实现了使用两种伪标签（带CRF和不带CRF）训练DeepLab-LargeFOV + ResNet50模型，
并计算两种方法在训练集和验证集上的mIoU性能指标。

输入:
    - outputs/pseudo_masks/*.png: CRF后处理的伪标签
    - outputs/base_pseudo/*.png: 不带CRF处理的伪标签
    - data/images/: 原始图像数据集

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
import cv2
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
        self.img_dir = "data/images"
        self.base_mask_dir = "outputs/base_pseudo"
        self.crf_mask_dir = "outputs/pseudo_masks"
        self.model_dir = "models/segmentor"
        self.result_dir = "outputs/results"
        
        # 模型配置
        self.backbone = "resnet50"  # 只使用ResNet50
        self.atrous_rates = (6, 12, 18, 24)  # 空洞卷积率
        self.num_classes = 2  # 背景和前景
        
        # 训练配置
        self.batch_size = 8
        self.num_epochs = 5
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        self.image_size = (224, 224)  # (height, width)
        
        # 保存与输出配置
        self.save_every = 5  # 每隔多少个epoch保存一次模型
        self.eval_every = 1   # 每隔多少个epoch评估一次模型
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
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        
        # 获取所有有对应掩码的图像
        self.image_paths = []
        self.mask_paths = []
        
        # 支持的图像扩展名
        img_extensions = ['.jpg', '.jpeg', '.png']
        
        # 首先获取所有掩码文件
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        # 对于每个掩码，查找对应的图像
        for mask_file in mask_files:
            base_name = Path(mask_file).stem  # 去掉扩展名
            
            # 查找对应的图像文件
            for ext in img_extensions:
                img_path = self.img_dir / f"{base_name}{ext}"
                if img_path.exists():
                    self.image_paths.append(img_path)
                    self.mask_paths.append(self.mask_dir / mask_file)
                    break
        
        # 预先验证图像，移除无效的图像
        valid_indices = []
        for i, (img_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            try:
                # 尝试读取图像，确保它是有效的
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Cannot read image {img_path}, skipping")
                    continue
                
                # 尝试读取掩码，确保它是有效的
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
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
            # 读取图像和掩码
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Cannot read image: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # 尝试读取彩色图像并转换为灰度（处理CAM可视化图像）
                mask_color = cv2.imread(str(mask_path))
                if mask_color is not None:
                    # 如果是彩色的CAM图像，转换为灰度
                    mask = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
                else:
                    raise ValueError(f"Cannot read mask: {mask_path}")
            
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
                
                # 利用Otsu自适应阈值进行二值化
                if mask.max() <= 1:  # 可能是0-1归一化的
                    mask = (mask * 255).astype(np.uint8)
                
                # 使用Otsu阈值法
                _, binary_mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 应用形态学操作以改善掩码质量
                kernel = np.ones((3, 3), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                
                # 检查前景像素占比
                foreground_ratio = np.mean(binary_mask) * 100
                if idx < 5:
                    print(f"CAM binarized mask {idx}: foreground pixel ratio={foreground_ratio:.2f}%")
                
                # 如果前景像素太少，可能需要降低阈值
                if foreground_ratio < 1.0:  # 低于1%的前景
                    # 使用更低的固定阈值
                    _, binary_mask = cv2.threshold(mask, mask.max() * 0.3, 1, cv2.THRESH_BINARY)
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
                image = self.transform(image)
            
            if self.mask_transform:
                # 不需要再执行squeeze操作，因为我们自定义的BinaryMaskToTensor已经返回了正确形状的张量
                mask = self.mask_transform(mask)
                # 确保掩码是2D张量
                if mask.dim() > 2:
                    mask = mask.squeeze()
            else:
                # 确保mask是二值的，且为long类型
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.uint8)
                mask = torch.from_numpy(mask).long()
            
            return {"image": image, "mask": mask, "path": str(img_path)}
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回一个替代样本（空图像和掩码）
            # 这里创建一个与数据集中其他样本相同形状的空样本
            if self.transform:
                dummy_image = torch.zeros(3, *self.transform(np.zeros((224, 224, 3), dtype=np.uint8)).shape[1:])
            else:
                dummy_image = torch.zeros(3, 224, 224)
                
            # 创建一个224x224的2D掩码
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
        # 确保掩码是二值的，值为0和1
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        # 转换为张量，保持二值特性
        # 注意: 添加通道维度以匹配Resize期望的格式
        mask_tensor = torch.from_numpy(mask).long()
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0) # 增加通道维度
        return mask_tensor

# 创建自定义的Resize转换，保持掩码值
class MaskResize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, mask_tensor):
        # 使用最近邻插值进行缩放，保持掩码值
        resized = F.interpolate(
            mask_tensor.float().unsqueeze(0),  # 添加批次维度
            size=self.size,
            mode='nearest'
        )
        # 移除批次维度，确保移除通道维度
        resized = resized.squeeze(0).squeeze(0).long()
        return resized  # 应该是2D张量 [H, W]

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
def train_model(model, train_loader, val_loader, criterion, optimizer, config, mask_type="base"):
    device = config.device
    model.to(device)
    
    # 创建结果目录
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # 日志字典
    log = {
        "train_loss": [],
        "val_miou": [],
        "train_miou": [],
        "mask_type": mask_type,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
    }
    
    best_miou = 0.0
    start_time = time.time()
    
    # 检查首个批次是否有前景类
    try:
        check_labels_batch = next(iter(train_loader))
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
        # 继续执行
    
    # 使用带权重的损失函数处理不平衡类别
    if config.num_classes == 2:
        # 估计类别权重来处理不平衡问题
        weight = torch.FloatTensor([0.1, 0.9]).to(device)  # 假设前景类(1)比背景类(0)少得多
        weighted_criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        weighted_criterion = criterion
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
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
        train_loss /= len(train_loader)
        log["train_loss"].append(train_loss)
        
        # 定期评估模型
        if (epoch + 1) % config.eval_every == 0:
            # 在验证集上评估
            val_miou, val_iou_per_class = compute_miou(model, val_loader, device, config.num_classes)
            log["val_miou"].append(val_miou)
            
            # 在训练集上评估
            train_miou, train_iou_per_class = compute_miou(model, train_loader, device, config.num_classes)
            log["train_miou"].append(train_miou)
            
            print(f"Epoch {epoch+1}/{config.num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train mIoU: {train_miou:.4f}, "
                  f"Val mIoU: {val_miou:.4f}")
            
            # 添加每个类别的IoU到日志
            if epoch == 0 or (epoch + 1) % 5 == 0:  # 每5个epoch记录一次详细IoU
                class_ious = {}
                for i in range(config.num_classes):
                    class_ious[f"class_{i}_iou"] = float(val_iou_per_class[i])
                print(f"Class IoU details: {class_ious}")
            
            # 保存最佳模型
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), 
                           os.path.join(config.model_dir, f"best_model_{mask_type}.pth"))
                print(f"Saving best model, mIoU: {best_miou:.4f}")
    
    # 保存训练日志
    training_time = time.time() - start_time
    log["training_time"] = training_time
    log["best_miou"] = best_miou
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.result_dir, f"training_log_{mask_type}_{timestamp}.json")
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
    
    print(f"Training completed in {training_time/60:.2f} minutes. Best mIoU: {best_miou:.4f}")
    
    return model, log

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate segmentation models with different pseudo masks')
    parser.add_argument('--base_only', action='store_true', help='Only train and evaluate with base pseudo masks')
    parser.add_argument('--crf_only', action='store_true', help='Only train and evaluate with CRF pseudo masks')
    parser.add_argument('--fast_mode', action='store_true', help='Run in fast mode to verify model functionality')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--crf_dir', type=str, default=None, help='Directory with CRF preset masks to use (e.g., outputs/pseudo_masks/preset_A)')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化配置
    config = SegmentationConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
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
    if args.crf_dir and os.path.exists(args.crf_dir) and os.listdir(args.crf_dir):
        print(f"\n===== Using CRF preset masks from {args.crf_dir} =====")
        if not args.base_only:
            config.crf_mask_dir = args.crf_dir
            print(f"Will use CRF preset masks from: {args.crf_dir}")
    
    # 创建结果目录
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(config.image_size)
    ])
    
    mask_transform = transforms.Compose([
        BinaryMaskToTensor(),
        MaskResize(config.image_size)
    ])
    
    # 创建结果汇总
    results = {"base": {}, "crf": {}}
    
    # 训练使用基础掩码的模型
    if not args.crf_only:
        print("\n===== Training with base pseudo masks =====")
        base_dataset = SegmentationDataset(config.img_dir, config.base_mask_dir, transform, mask_transform)
        
        # 快速模式下只使用少量样本
        if config.fast_mode:
            base_dataset = create_fast_mode_subset(base_dataset, config.fast_samples)
            print(f"Fast mode: Using {len(base_dataset)} samples from base pseudo masks")
        
        # 分割为训练集和验证集
        train_size = int(config.train_ratio * len(base_dataset))
        val_size = len(base_dataset) - train_size
        train_dataset, val_dataset = random_split(base_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
        
        # 创建模型、损失函数和优化器
        model_base = DeepLabLargeFOV(num_classes=config.num_classes, atrous_rates=config.atrous_rates, 
                                     fast_mode=config.fast_mode)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model_base.parameters(), lr=config.learning_rate, 
                                     weight_decay=config.weight_decay)
        
        # 训练模型
        model_base, log_base = train_model(model_base, train_loader, val_loader, criterion, optimizer, config, "base")
        
        # 保存结果
        results["base"] = {
            "train_miou": log_base["train_miou"][-1] if log_base["train_miou"] else None,
            "val_miou": log_base["val_miou"][-1] if log_base["val_miou"] else None,
            "best_miou": log_base["best_miou"] if "best_miou" in log_base else None
        }
    
    # 训练使用CRF掩码的模型
    if not args.base_only:
        print("\n===== Training with CRF pseudo masks =====")
        crf_dataset = SegmentationDataset(config.img_dir, config.crf_mask_dir, transform, mask_transform)
        
        # 快速模式下只使用少量样本
        if config.fast_mode:
            crf_dataset = create_fast_mode_subset(crf_dataset, config.fast_samples)
            print(f"Fast mode: Using {len(crf_dataset)} samples from CRF pseudo masks")
        
        # 分割为训练集和验证集
        train_size = int(config.train_ratio * len(crf_dataset))
        val_size = len(crf_dataset) - train_size
        train_dataset, val_dataset = random_split(crf_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
        
        # 创建模型、损失函数和优化器
        model_crf = DeepLabLargeFOV(num_classes=config.num_classes, atrous_rates=config.atrous_rates,
                                   fast_mode=config.fast_mode)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model_crf.parameters(), lr=config.learning_rate, 
                                     weight_decay=config.weight_decay)
        
        # 训练模型
        model_crf, log_crf = train_model(model_crf, train_loader, val_loader, criterion, optimizer, config, "crf")
        
        # 保存结果
        results["crf"] = {
            "train_miou": log_crf["train_miou"][-1] if log_crf["train_miou"] else None,
            "val_miou": log_crf["val_miou"][-1] if log_crf["val_miou"] else None,
            "best_miou": log_crf["best_miou"] if "best_miou" in log_crf else None
        }
    
    # 打印结果对比
    print("\n===== Results =====")
    print(f"Base Pseudo Masks - Train mIoU: {results['base'].get('train_miou', 'N/A')}")
    print(f"Base Pseudo Masks - Val mIoU: {results['base'].get('val_miou', 'N/A')}")
    print(f"CRF Pseudo Masks - Train mIoU: {results['crf'].get('train_miou', 'N/A')}")
    print(f"CRF Pseudo Masks - Val mIoU: {results['crf'].get('val_miou', 'N/A')}")
    
    # 保存结果对比
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(config.result_dir, f"comparison_results_{timestamp}.json")
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存CSV格式结果
    csv_path = os.path.join(config.result_dir, f"miou_comparison_{timestamp}.csv")
    with open(csv_path, 'w') as f:
        f.write("train mIoU,train+CRF mIoU,val mIoU,val+CRF mIoU\n")
        f.write(f"{results['base'].get('train_miou', 0):.4f},{results['crf'].get('train_miou', 0):.4f}," +
                f"{results['base'].get('val_miou', 0):.4f},{results['crf'].get('val_miou', 0):.4f}\n")
    
    print(f"Results comparison saved to {comparison_path}")
    print(f"mIoU comparison CSV saved to {csv_path}")

if __name__ == "__main__":
    main() 