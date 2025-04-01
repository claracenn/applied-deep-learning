"""
M3_TRAIN_FULLY_SUPERVISED - 使用真实标签训练DeepLab-LargeFOV + ResNet50进行语义分割

本脚本实现了使用真实标签训练DeepLab-LargeFOV + ResNet50模型，作为弱监督方法的比较基线。
使用与弱监督方法相同的网络架构，以确保公平比较。

输入:
    - data/ground_truth/: 真实分割标签的目录
    - data/images/: 原始图像数据集

输出:
    - models/segmentor/: 保存训练好的模型
    - outputs/results/: 保存评估结果和训练日志

用法:
    # 标准训练模式
    python src/m3_train_fully_supervised.py
    
    
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

# 导入现有模块
from m3_train_segmentation import (
    set_seed, print_progress, BinaryMaskToTensor, MaskResize,
    DeepLabLargeFOV, compute_miou
)

# 配置
class FullySupConfig:
    def __init__(self):
        self.img_dir = "data/images"
        self.mask_dir = "data/annotations/trimaps"  # 真实标签的目录
        self.model_dir = "models/segmentor/fully_supervised"  # 单独的模型子目录
        self.result_dir = "outputs/results/fully_supervised"  # 单独的结果子目录
        
        # 模型配置
        self.backbone = "resnet50"  # 只使用ResNet50
        self.atrous_rates = (6, 12, 18, 24)  # 空洞卷积率
        self.num_classes = 2  # 背景和前景
        
        # 训练配置
        self.batch_size = 8
        self.num_epochs = 10
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        self.image_size = (224, 224)  # (height, width)
        
        # 保存与输出配置
        self.eval_every = 1   # 每隔多少个epoch评估一次模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 快速模式配置
        self.fast_mode = False
        self.fast_samples = 20  # 快速模式中每类使用的样本数
        self.fast_image_size = (128, 128)  # 快速模式中使用的图像尺寸
        self.fast_epochs = 1  # 快速模式中的epoch数

# 数据集类 - 专门用于加载真实标签
class FullySupSegDataset(Dataset):
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
                    print(f"Warning: Cannot read image: {img_path}, skipping")
                    continue
                
                # 尝试读取掩码，确保它是有效的
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Cannot read mask: {mask_path}, skipping")
                    continue
                
                valid_indices.append(i)
            except Exception as e:
                print(f"Error: Error processing image {img_path} and mask {mask_path}: {e}")
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
                raise ValueError(f"Cannot read mask: {mask_path}")
            
            # 打印一些样本的统计信息
            if idx < 5:  # 只对前几个样本打印调试信息
                print(f"Mask {idx} statistics: min={mask.min()}, max={mask.max()}, unique values={np.unique(mask)}")
                # 检查前景像素占比
                foreground_ratio = np.mean(mask > 0) * 100
                print(f"Mask {idx}: foreground pixel ratio={foreground_ratio:.2f}%")
            
            # 处理真实标签 - 确保二值化
            if np.max(mask) > 1:
                # 处理trimap格式 (1=背景, 2/3=前景)
                # 将值1映射为背景(0)，将值2和3映射为前景(1)
                binary_mask = np.zeros_like(mask)
                binary_mask[mask > 1] = 1  # 只有2和3被视为前景
                mask = binary_mask
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            if self.mask_transform:
                mask = self.mask_transform(mask)
                # 确保掩码是2D张量
                if mask.dim() > 2:
                    mask = mask.squeeze()
            else:
                mask = torch.from_numpy(mask).long()
            
            return {"image": image, "mask": mask, "path": str(img_path)}
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回一个替代样本（空图像和掩码）
            if self.transform:
                dummy_image = torch.zeros(3, *self.transform(np.zeros((224, 224, 3), dtype=np.uint8)).shape[1:])
            else:
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

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, config, model_type="fully_supervised"):
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
        "mask_type": model_type,
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
                           os.path.join(config.model_dir, f"best_model_{model_type}.pth"))
                print(f"Saving best model, mIoU: {best_miou:.4f}")
    
    # 保存训练日志
    training_time = time.time() - start_time
    log["training_time"] = training_time
    log["best_miou"] = best_miou
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config.result_dir, f"training_log_{model_type}_{timestamp}.json")
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
    
    print(f"Training completed in {training_time/60:.2f} minutes. Best mIoU: {best_miou:.4f}")
    
    return model, log

def save_comparison_results(results_fs, results_ws=None, save_dir="outputs/results"):
    """
    保存全监督方法与弱监督方法的mIoU对比结果
    
    参数:
        results_fs: 全监督方法的结果
        results_ws: 弱监督方法的结果
        save_dir: 保存结果的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集结果
    train_mious = [results_fs.get("train_miou", 0)]
    val_mious = [results_fs.get("val_miou", 0)]
    labels = ["Fully Supervised"]
    
    # 如果有弱监督结果，添加到比较中
    if results_ws is not None:
        if "base" in results_ws and results_ws["base"].get("best_miou") is not None:
            train_mious.append(results_ws["base"].get("train_miou", 0))
            val_mious.append(results_ws["base"].get("val_miou", 0))
            labels.append("Weakly Supervised (No CRF)")
            
        if "crf" in results_ws and results_ws["crf"].get("best_miou") is not None:
            train_mious.append(results_ws["crf"].get("train_miou", 0))
            val_mious.append(results_ws["crf"].get("val_miou", 0))
            labels.append("Weakly Supervised (With CRF)")
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存比较结果为CSV
    csv_path = os.path.join(save_dir, f"comparison_results_fullyVSweakly_{timestamp}.csv")
    with open(csv_path, 'w') as f:
        f.write("train,val\n")
        for train_miou, val_miou in zip(train_mious, val_mious):
            f.write(f"{train_miou:.4f},{val_miou:.4f}\n")
    
    # 输出比较结果
    print("\n===== mIoU Comparison Results =====")
    print(f"{'Method':<15}{'Train':<10}{'Val':<10}")
    print("="*35)
    for label, train_miou, val_miou in zip(labels, train_mious, val_mious):
        print(f"{label:<15}{train_miou:.4f}{val_miou:.4f}")
    print("="*35)
    
    print(f"Comparison results saved to {csv_path}")

def load_weakly_supervised_results(result_dir="outputs/results"):
    """
    加载已有的弱监督方法结果
    
    参数:
        result_dir: 保存结果的目录
        
    返回:
        字典: 弱监督方法的结果
    """
    results = {"base": {}, "crf": {}}
    
    # 使用弱监督训练的通用结果目录
    weakly_result_dir = os.path.join(os.path.dirname(result_dir), "")
    
    # 查找最新的比较结果文件
    comparison_files = [f for f in os.listdir(weakly_result_dir) 
                        if f.startswith("comparison_results_") and 
                        f.endswith(".json") and 
                        "fullyVSweakly" not in f]
    
    if not comparison_files:
        print("No comparison result files found for weakly supervised methods")
        
        # 尝试从训练日志中查找结果
        log_files = [f for f in os.listdir(weakly_result_dir) 
                    if f.startswith("training_log_") and f.endswith(".json")]
        
        if log_files:
            log_files.sort(reverse=True)
            for log_file in log_files:
                try:
                    with open(os.path.join(weakly_result_dir, log_file), 'r') as f:
                        log_data = json.load(f)
                    
                    if "mask_type" in log_data and "best_miou" in log_data:
                        mask_type = log_data["mask_type"]
                        if mask_type == "base":
                            results["base"]["best_miou"] = log_data["best_miou"]
                            print(f"Loaded base mask results from log: mIoU = {log_data['best_miou']:.4f}")
                        elif mask_type == "crf":
                            results["crf"]["best_miou"] = log_data["best_miou"]
                            print(f"Loaded CRF mask results from log: mIoU = {log_data['best_miou']:.4f}")
                except Exception as e:
                    print(f"Error reading log file: {e}")
        
        return results
    
    # 按时间戳排序
    comparison_files.sort(reverse=True)
    latest_file = os.path.join(weakly_result_dir, comparison_files[0])
    
    try:
        with open(latest_file, 'r') as f:
            comparison_data = json.load(f)
        
        # 从比较结果中提取弱监督方法的结果
        if 'Weakly Supervised (No CRF)' in comparison_data:
            results["base"]["best_miou"] = comparison_data['Weakly Supervised (No CRF)']
        elif 'base' in comparison_data:
            results["base"]["best_miou"] = comparison_data['base']
        
        if 'Weakly Supervised (With CRF)' in comparison_data:
            results["crf"]["best_miou"] = comparison_data['Weakly Supervised (With CRF)']
        elif 'crf' in comparison_data:
            results["crf"]["best_miou"] = comparison_data['crf']
            
        print(f"Loaded weakly supervised method results: {latest_file}")
    except Exception as e:
        print(f"Failed to load weakly supervised method results: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train segmentation model with fully supervised learning')
    parser.add_argument('--fast_mode', action='store_true', help='Run in fast mode to verify model functionality')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--skip_comparison', action='store_true', help='Skip weakly-supervised comparison')
    parser.add_argument('--gt_dir', type=str, default=None, help='Directory with ground truth segmentation masks')
    parser.add_argument('--weakly_result_dir', type=str, default="outputs/results", 
                       help='Directory containing weakly supervised results')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化配置
    config = FullySupConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # 如果指定了真实标签目录
    if args.gt_dir:
        config.mask_dir = args.gt_dir
    
    # 设置数据加载器线程数
    num_workers = 0 if args.fast_mode else args.num_workers
    
    # 快速模式设置
    if args.fast_mode:
        print("\n===== Running in FAST MODE to verify model functionality =====")
        config.fast_mode = True
        config.num_epochs = config.fast_epochs
        config.image_size = config.fast_image_size
        config.batch_size = 4
        config.learning_rate = 5e-4
        config.eval_every = 1
    
    # 创建结果目录
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    
    # 输出清晰的目录说明
    print(f"\nModels will be saved to: {config.model_dir}")
    print(f"Results will be saved to: {config.result_dir}")
    
    # 检查真实标签目录是否存在
    if not os.path.exists(config.mask_dir):
        print(f"Error: Ground truth mask directory does not exist {config.mask_dir}")
        print(f"Please prepare ground truth data first or specify the correct directory with --gt_dir")
        return
    
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
    
    # 创建全监督数据集
    print("\n===== Creating Fully Supervised Dataset =====")
    full_dataset = FullySupSegDataset(config.img_dir, config.mask_dir, transform, mask_transform)
    
    # 快速模式下只使用少量样本
    if config.fast_mode:
        full_dataset = create_fast_mode_subset(full_dataset, config.fast_samples)
        print(f"Fast mode: Using {len(full_dataset)} samples")
    
    # 分割为训练集和验证集
    train_size = int(config.train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
    
    # 创建模型、损失函数和优化器
    print("\n===== Training Fully Supervised Model =====")
    model = DeepLabLargeFOV(num_classes=config.num_classes, atrous_rates=config.atrous_rates, 
                           fast_mode=config.fast_mode)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                 weight_decay=config.weight_decay)
    
    # 训练模型
    model, log = train_model(model, train_loader, val_loader, criterion, optimizer, config, "fully_supervised")
    
    # 保存最终结果
    results_fs = {
        "best_miou": log["best_miou"],
        "train_miou": log["train_miou"][-1] if log["train_miou"] else None,
        "val_miou": log["val_miou"][-1] if log["val_miou"] else None
    }
    
    # 进行方法比较
    if not args.skip_comparison:
        print("\n===== Comparing Fully Supervised vs Weakly Supervised Methods =====")
        # 加载弱监督方法结果
        results_ws = load_weakly_supervised_results(args.weakly_result_dir)
        
        # 如果成功加载弱监督结果，则保存比较结果
        if (results_ws["base"].get("best_miou") is not None or 
            results_ws["crf"].get("best_miou") is not None):
            # 保存比较结果
            save_comparison_results(results_fs, results_ws, config.result_dir)
        else:
            print("No valid weakly supervised method results found, skipping comparison")
            save_comparison_results(results_fs, None, config.result_dir)
    else:
        # 只输出全监督结果
        save_comparison_results(results_fs, None, config.result_dir)
    
    # 添加比较说明
    print("\n===== Experiment Completed =====")
    print(f"Fully supervised model best mIoU: {results_fs['best_miou']:.4f}")
    print(f"\nTo view the fully supervised model, check: {config.model_dir}/best_model_fully_supervised.pth")
    print(f"Fully supervised training logs saved in: {config.result_dir}")
    print(f"Comparison analysis results saved in: {config.result_dir}")

if __name__ == "__main__":
    main() 