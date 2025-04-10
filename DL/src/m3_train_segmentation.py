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
    IMAGE_DIR,
    SEGMENTATION_DIR,
    PSEUDO_MASK_DIR,
    ANNOTATION_DIR,
    MODEL_ROOT,
    OUTPUT_ROOT
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
        # 从配置文件中导入设置
        from config import (
            SEGMENTATION_CONFIG, 
            SEGMENTATION_PATHS, 
            IMAGE_DIR,
            SEGMENTATION_DIR,
            PSEUDO_MASK_DIR,
            ANNOTATION_DIR,
            MODEL_ROOT,
            OUTPUT_ROOT
        )
        
        # 使用配置文件中的路径
        self.img_dir = SEGMENTATION_PATHS["img_dir"]
        self.base_mask_dir = SEGMENTATION_PATHS["base_mask_dir"]
        self.crf_mask_dir = SEGMENTATION_PATHS["crf_mask_dir"]
        self.gt_mask_dir = SEGMENTATION_PATHS["gt_mask_dir"]
        self.model_dir = SEGMENTATION_PATHS["model_dir"] 
        self.result_dir = SEGMENTATION_PATHS["result_dir"]
        
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
        
        # 命令行参数相关属性
        self.base_only = SEGMENTATION_CONFIG["base_only"]
        self.crf_only = SEGMENTATION_CONFIG["crf_only"]
        self.seed = SEGMENTATION_CONFIG["seed"]
        self.patience = SEGMENTATION_CONFIG["patience"]
        
        # 损失函数参数
        self.foreground_weight = SEGMENTATION_CONFIG["foreground_weight"]
        self.background_weight = SEGMENTATION_CONFIG["background_weight"]

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
            mask = Image.open(str(mask_path)).convert('L')
            
            # 转换为numpy数组
            mask_np = np.array(mask)
            
            # 处理trimap值：
            # 1: 前景 → 1
            # 2: 背景 → 0
            # 3: 未分类 → 255 (忽略的标签值)
            binary_mask = np.zeros_like(mask_np)
            binary_mask[mask_np == 1] = 1  # 前景
            binary_mask[mask_np == 2] = 0  # 背景
            binary_mask[mask_np == 3] = 255  # 未分类区域标记为255
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            if self.mask_transform:
                mask_pil = Image.fromarray(binary_mask.astype(np.uint8))
                mask_tensor = self.mask_transform(mask_pil)
            else:
                mask_tensor = torch.from_numpy(binary_mask).long()
            
            return {"image": image, "mask": mask_tensor, "path": str(img_path)}
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回一个替代样本（空图像和掩码）
            dummy_image = torch.zeros(3, 224, 224)
            dummy_mask = torch.zeros(224, 224, dtype=torch.long)
            
            return {"image": dummy_image, "mask": dummy_mask, "path": str(img_path)}

# 掩码变换 - 修复: 确保二值掩码在ToTensor后不会变成浮点数0
class BinaryMaskToTensor:
    def __call__(self, mask):
        """
        将掩码转换为PyTorch tensor，处理多种可能的输入类型
        """
        # 如果输入是PIL Image，先转换为numpy数组
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # 如果已经是tensor，确保类型正确
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.long()
        else:
            # 确保掩码是二值的
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask).long()
        
        # 确保维度正确 - 不需要通道维度，CrossEntropyLoss期望(N, H, W)
        if mask_tensor.dim() == 3:  # 如果有通道维度，去掉它
            mask_tensor = mask_tensor.squeeze(0)
            
        return mask_tensor

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
    def __init__(self, num_classes=2, atrous_rates=(6, 12, 18, 24)):
        super(DeepLabLargeFOV, self).__init__()
        
        # 使用ResNet50作为骨干网络
        base = resnet50(pretrained=True)
        
        # 保存各个层，以便访问中间层特征
        self.layer0 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        # 冻结前两层
        for layer in [self.layer0, self.layer1]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # 获取中间层的通道数 (ResNet50 layer3的输出通道数是1024)
        mid_channels = 1024
        
        # 最终输出通道数
        in_channels = 2048  # ResNet50 layer4的输出通道数
        
        # 添加辅助分支 - 使用正确的通道数
        self.aux_branch = nn.Sequential(
            nn.Conv2d(mid_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # 完整的ASPP
        self.aspp = ASPP(in_channels, atrous_rates)
        
        # 优化后的分类器头部
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # 增加dropout率
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),  # 额外的卷积层
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        input_size = x.size()
        
        # 逐层提取特征
        x = self.layer0(x)   # 1/4
        x = self.layer1(x)   # 1/4
        x = self.layer2(x)   # 1/8
        
        # 提取中间层特征用于辅助分支
        aux_features = self.layer3(x)  # 1/16, 1024 通道
        
        # 最后一层特征
        x = self.layer4(aux_features)  # 1/32, 2048 通道
        
        # 应用ASPP
        x = self.aspp(x)
        
        # 分类
        main_out = self.classifier(x)
        
        # 计算辅助输出
        if self.training:
            aux_out = self.aux_branch(aux_features)
            aux_out = F.interpolate(aux_out, size=input_size[2:], mode='bilinear', align_corners=True)
        else:
            aux_out = None
        
        # 上采样到原始输入尺寸
        main_out = F.interpolate(main_out, size=input_size[2:], mode='bilinear', align_corners=True)
        
        if self.training:
            return main_out, aux_out
        else:
            return main_out

# 改进的ASPP模块 (Atrous Spatial Pyramid Pooling)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        
        out_channels = 256
        
        # 1x1卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)  # 增加dropout
        )
        
        # 空洞卷积分支
        self.conv_atrous = nn.ModuleList()
        for rate in atrous_rates:
            self.conv_atrous.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2)  # 增加dropout
            ))
        
        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)  # 增加dropout
        )
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 合并所有分支
        concat_channels = (len(atrous_rates) + 2) * out_channels  # +2是1x1卷积和全局池化分支
        self.conv_merge = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)  # 保持较高的dropout率
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
        
        # 应用通道注意力
        att = self.channel_attention(x)
        x = x * att
        
        return x

# 评估函数：计算mIoU
def compute_miou(model, dataloader, device, num_classes=2):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # 前景类别像素计数（不包括未分类区域）
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
            
            # 统计前景像素比例（不包括未分类区域）
            valid_mask = masks != 255  # 255是未分类区域
            if batch_idx < 5:  # 只检查前几个批次
                foreground_pixels = (masks[valid_mask] == 1).sum().item()
                total_pixels = valid_mask.sum().item()
                foreground_pixel_count += foreground_pixels
                total_pixel_count += total_pixels
            
            # 前向传播
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 更新混淆矩阵，忽略未分类区域
            for pred, mask in zip(preds, masks):
                pred = pred.cpu().numpy().flatten()
                mask = mask.cpu().numpy().flatten()
                
                # 只考虑有效的（非未分类）区域
                valid_indices = mask != 255
                
                if valid_indices.sum() > 0:
                    # 确保预测和真实标签中包含有效的类别值
                    valid_mask = (mask[valid_indices] < num_classes) & (mask[valid_indices] >= 0)
                    valid_pred = (pred[valid_indices] < num_classes) & (pred[valid_indices] >= 0)
                    
                    final_valid = valid_mask & valid_pred
                    if final_valid.sum() > 0:
                        confusion_matrix += np.bincount(
                            num_classes * mask[valid_indices][final_valid] + pred[valid_indices][final_valid],
                            minlength=num_classes**2
                        ).reshape(num_classes, num_classes)
    
    # 打印前景像素占比统计（不包括未分类区域）
    if total_pixel_count > 0:
        foreground_percent = 100 * foreground_pixel_count / total_pixel_count
        print(f"Foreground pixels (class 1) percentage (excluding ignored regions): {foreground_percent:.2f}%")
    
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
        "val_miou": [],
        "test_miou": [],
        "mask_type": mask_type,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
    }
    
    best_miou = 0.0
    patience = config.patience if hasattr(config, 'patience') else 5  # 使用配置的早停耐心值
    no_improve = 0  # 未改善计数器
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
    
    # 准备损失函数
    # 主损失：带权重的交叉熵损失
    if config.num_classes == 2:
        # 从配置获取类别权重，如果有的话
        if hasattr(config, 'foreground_weight') and hasattr(config, 'background_weight'):
            weight = torch.FloatTensor([config.background_weight, config.foreground_weight]).to(device)
        else:
            # 更平衡的权重设置
            # 假设前景约占30%，背景约占70%
            weight = torch.FloatTensor([0.4, 0.6]).to(device)  # 稍微增加前景权重
        
        weighted_criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
    else:
        weighted_criterion = criterion
    
    # 辅助损失：标准交叉熵
    aux_criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # 学习率调度器 - 使用改进的余弦退火策略
    if hasattr(config, 'scheduler') and config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5,  # 初始周期
            T_mult=2,  # 每个周期后的倍增因子
            eta_min=getattr(config, 'min_lr', 1e-6)
        )
    else:
        # 使用OneCycleLR，但调整参数
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate * 3,  # 降低最大学习率
            epochs=config.num_epochs,
            steps_per_epoch=len(train_subset_loader),
            pct_start=0.3,  # 30%的时间用于预热
            div_factor=10.0,  # 初始学习率为max_lr/10
            final_div_factor=1e3,  # 最终学习率为初始学习率/1000
            anneal_strategy='cos'
        )
    
    # 混合精度训练
    if torch.cuda.is_available():
        try:
            # 尝试使用新API (PyTorch 2.0+)
            from torch.amp import GradScaler, autocast
            scaler = GradScaler()
            autocast_fn = autocast
        except ImportError:
            # 兼容旧API
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            autocast_fn = autocast
    else:
        scaler = None
        autocast_fn = None
    
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
            
            # 混合精度训练
            if scaler is not None and config.mixed_precision:
                with autocast_fn():
                    # 前向传播
                    outputs = model(images)
                    
                    # 处理主输出和辅助输出
                    if isinstance(outputs, tuple):
                        main_output, aux_output = outputs
                        # 主损失
                        main_loss = weighted_criterion(main_output, masks)
                        # 辅助损失
                        aux_loss = aux_criterion(aux_output, masks)
                        # 总损失：主损失 + 0.4 * 辅助损失
                        loss = main_loss + 0.4 * aux_loss
                    else:
                        # 如果没有辅助输出
                        loss = weighted_criterion(outputs, masks)
                
                # 反向传播和优化
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 无混合精度训练的情况
                # 前向传播
                outputs = model(images)
                
                # 处理主输出和辅助输出
                if isinstance(outputs, tuple):
                    main_output, aux_output = outputs
                    # 主损失
                    main_loss = weighted_criterion(main_output, masks)
                    # 辅助损失
                    aux_loss = aux_criterion(aux_output, masks)
                    # 总损失：主损失 + 0.4 * 辅助损失
                    loss = main_loss + 0.4 * aux_loss
                else:
                    # 如果没有辅助输出
                    loss = weighted_criterion(outputs, masks)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 记录损失
            train_loss += loss.item()
            
            # 更新学习率（如果使用OneCycleLR）
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
        
        # 计算平均训练损失
        train_loss /= len(train_subset_loader)
        log["train_loss"].append(train_loss)
        
        # 更新学习率（如果使用CosineAnnealingWarmRestarts）
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()
        
        # 定期评估模型
        if (epoch + 1) % config.eval_every == 0 or epoch == config.num_epochs - 1:
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
            
            # 早停检查 (基于验证集性能)
            if val_miou > best_miou:
                best_miou = val_miou
                no_improve = 0
                # 保存最佳模型
                model_path = model_dir / f"best_model_{mask_type}.pth"
                torch.save(model.state_dict(), str(model_path))
                print(f"Saving best model, Val mIoU: {best_miou:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
    
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
    # 从配置文件中加载默认参数值
    from config import SEGMENTATION_CONFIG, SEGMENTATION_PATHS
    
    # 创建参数解析器，所有值的默认值来自配置文件
    parser = argparse.ArgumentParser(description='Train and evaluate segmentation models with different pseudo masks')
    
    # 命令行参数 - 功能标志
    parser.add_argument('--base_only', action='store_true', help='Only train and evaluate with base pseudo masks', 
                        default=SEGMENTATION_CONFIG['base_only'])
    parser.add_argument('--crf_only', action='store_true', help='Only train and evaluate with CRF pseudo masks',
                        default=SEGMENTATION_CONFIG['crf_only'])
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=SEGMENTATION_CONFIG['num_epochs'], 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=SEGMENTATION_CONFIG['batch_size'], 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=SEGMENTATION_CONFIG['learning_rate'], 
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=SEGMENTATION_CONFIG['seed'], 
                        help='Random seed')
    parser.add_argument('--test_ratio', type=float, default=SEGMENTATION_CONFIG['test_ratio'], 
                        help='Ratio of test.txt to use as test set')
    parser.add_argument('--num_workers', type=int, default=SEGMENTATION_CONFIG['num_workers'], 
                        help='Number of data loading workers')
    parser.add_argument('--auxloss', action='store_true', help='Enable auxiliary loss', 
                        default=False)
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training', 
                        default=False)
    
    # 路径参数
    parser.add_argument('--crf_dir', type=str, default=None, 
                        help='Directory with CRF preset masks to use (e.g., outputs/pseudo_masks/preset_A)')
    parser.add_argument('--trimap_dir', type=str, default=SEGMENTATION_PATHS['gt_mask_dir'], 
                        help='真实标签(trimap)目录')
    parser.add_argument('--trainval_file', type=str, default=SEGMENTATION_PATHS['trainval_file'], 
                        help='训练集列表文件')
    parser.add_argument('--test_file', type=str, default=SEGMENTATION_PATHS['test_file'], 
                        help='测试集列表文件')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"Set global random seed to {args.seed}")
    
    # 确保所有必要的目录都存在
    print("检查并创建必要的目录...")
    for path in [
        Path(SEGMENTATION_DIR),  # 基础伪标签目录
        Path(PSEUDO_MASK_DIR),   # CRF处理后的伪标签目录
        Path(SEGMENTATION_PATHS['model_dir']),  # 模型保存目录
        Path(SEGMENTATION_PATHS['result_dir']),   # 结果保存目录
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
    
    # 初始化配置
    config = SegmentationConfig()
    
    # 从命令行参数更新配置
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.gt_mask_dir = args.trimap_dir  # 设置真实标签目录
    config.seed = args.seed  # 将种子传递给config对象
    config.base_only = args.base_only
    config.crf_only = args.crf_only
    config.use_auxloss = args.auxloss
    config.mixed_precision = args.mixed_precision
    
    # 打印启用的优化功能
    if config.use_auxloss:
        print("\n===== 启用辅助损失 =====")
    if config.mixed_precision:
        print("\n===== 启用混合精度训练 =====")
    
    # 使用指定的CRF预设掩码
    if args.crf_dir and Path(args.crf_dir).exists() and any(Path(args.crf_dir).glob('*.png')):
        print(f"\n===== Using CRF preset masks from {args.crf_dir} =====")
        if not args.base_only:
            config.crf_mask_dir = args.crf_dir
            print(f"Will use CRF preset masks from: {args.crf_dir}")
    
    # 创建结果目录
    Path(config.model_dir).mkdir(exist_ok=True, parents=True)
    Path(config.result_dir).mkdir(exist_ok=True, parents=True)
    
    # 数据变换 - 增强数据增强策略
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 增加输入分辨率
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # 增加尺度变化范围
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),  # 增加旋转角度
        transforms.ColorJitter(
            brightness=0.4, 
            contrast=0.4, 
            saturation=0.4, 
            hue=0.2
        ),  # 增强色彩抖动
        transforms.RandomAffine(
            degrees=30, 
            translate=(0.2, 0.2), 
            scale=(0.8, 1.2),
            shear=15
        ),  # 增强仿射变换
        # 添加更多数据增强
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),  # 随机高斯模糊
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # 随机锐化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 掩码变换 - 确保与图像变换对应
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomRotation(30),  # 与图像相同的旋转
        transforms.RandomAffine(
            degrees=30,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2),
            shear=15
        ),  # 与图像相同的仿射变换
        BinaryMaskToTensor(),
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
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # 创建模型、损失函数和优化器
        model_base = DeepLabLargeFOV(num_classes=config.num_classes, atrous_rates=config.atrous_rates)
        # 交叉熵损失，忽略255（未分类区域）
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # 使用AdamW优化器，增加权重衰减
        optimizer = torch.optim.AdamW(
            model_base.parameters(), 
            lr=config.learning_rate,
            weight_decay=SEGMENTATION_CONFIG['weight_decay'] * 1.5,  # 增加权重衰减来减少过拟合
            betas=(0.9, 0.999)
        )
        
        # 训练模型
        model_base, log_base = train_model(model_base, train_loader, test_loader, criterion, optimizer, config, "base")
        
        # 保存结果
        results["base"] = {
            "train_miou": log_base["train_miou"][-1] if log_base["train_miou"] else None,
            "val_miou": log_base["val_miou"][-1] if log_base["val_miou"] else None,
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
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # 创建模型、损失函数和优化器
        model_crf = DeepLabLargeFOV(num_classes=config.num_classes, atrous_rates=config.atrous_rates)
        
        # 交叉熵损失，忽略255（未分类区域）
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # 使用AdamW优化器，增加权重衰减
        optimizer = torch.optim.AdamW(
            model_crf.parameters(), 
            lr=config.learning_rate,
            weight_decay=SEGMENTATION_CONFIG['weight_decay'] * 1.5,  # 增加权重衰减来减少过拟合
            betas=(0.9, 0.999)
        )
        
        # 训练模型
        model_crf, log_crf = train_model(model_crf, train_loader, test_loader, criterion, optimizer, config, "crf")
        
        # 保存结果
        results["crf"] = {
            "train_miou": log_crf["train_miou"][-1] if log_crf["train_miou"] else None,
            "val_miou": log_crf["val_miou"][-1] if log_crf["val_miou"] else None,
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