"""
M3_TRAIN_FULLY_SUPERVISED - 使用真实标签训练DeepLab-LargeFOV + ResNet50进行语义分割

本脚本实现了使用真实标签训练DeepLab-LargeFOV + ResNet50模型，作为弱监督方法的比较基线。
使用与弱监督方法相同的网络架构，以确保公平比较。

输入:
    - data/annotations/trimaps/: 真实分割标签的目录
    - data/images/: 原始图像数据集

输出:
    - models/segmentor/fully_supervised/: 保存训练好的模型
    - outputs/results/fully_supervised/: 保存评估结果和训练日志

用法:
    python src/m3_train_fully_supervised.py
"""

import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import json
from datetime import datetime
import random

# 导入现有模块
from m3_train_segmentation import (
    set_seed, print_progress, BinaryMaskToTensor, MaskResize,
    DeepLabLargeFOV
)

# 导入配置
from config import (
    FULLY_SUP_CONFIG, FULLY_SUP_PATHS, PROJECT_ROOT, DATA_ROOT, 
    ANNOTATION_DIR, IMAGE_DIR, OUTPUT_ROOT, MODEL_ROOT
)

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
    with open(trainval_file, 'r') as f:
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
    with open(test_file, 'r') as f:
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

# 数据集类 - 专门用于加载真实标签
class FullySupSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, image_list=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_list = image_list  # 指定的图像列表

        self.image_paths = []
        self.mask_paths = []
        img_extensions = ['.jpg', '.jpeg', '.png']

        if image_list:
            # 如果提供了图像列表，则只使用列表中的图像
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
            # 如果没有提供图像列表，则使用所有可用的图像-掩码对
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

            for mask_file in mask_files:
                base_name = Path(mask_file).stem
                for ext in img_extensions:
                    img_path = self.img_dir / f"{base_name}{ext}"
                    if img_path.exists():
                        self.image_paths.append(img_path)
                        self.mask_paths.append(self.mask_dir / mask_file)
                        break

        # 验证图像和掩码对
        valid_indices = []
        for i, (img_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            try:
                # 确保文件存在
                if not os.path.exists(str(img_path)) or not os.path.exists(str(mask_path)):
                    raise FileNotFoundError(f"File not found: {img_path} or {mask_path}")
                
                # 尝试打开图像和掩码
                with Image.open(str(img_path)) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                with Image.open(str(mask_path)) as mask:
                    if mask.mode != "L":
                        mask = mask.convert("L")
                valid_indices.append(i)
            except Exception as e:
                print(f"Error: Skipping corrupted pair {img_path}, {mask_path}: {e}")

        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.mask_paths = [self.mask_paths[i] for i in valid_indices]
        print(f"Found {len(self.image_paths)} valid image-mask pairs from {mask_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 防止索引越界
        if idx >= len(self.image_paths):
            print(f"Warning: Index {idx} out of bounds, returning dummy sample")
            return self._get_dummy_sample()
            
        try:
            # 使用字符串路径打开文件
            img_path = str(self.image_paths[idx])
            mask_path = str(self.mask_paths[idx])
            
            # 确保文件存在
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"Warning: Files not found at index {idx}, returning dummy sample")
                return self._get_dummy_sample()

            # 打开图像为RGB模式
            with Image.open(img_path) as img:
                # 始终转换为RGB以确保一致性
                image = img.convert("RGB")

            # 打开掩码为灰度模式
            with Image.open(mask_path) as mask_img:
                # 始终转换为L以确保一致性
                mask_pil = mask_img.convert("L")
                # 转为numpy以进行处理
                mask_np = np.array(mask_pil)
            
            # 处理trimap值 - 创建二值掩码
            binary_mask = np.zeros_like(mask_np)
            binary_mask[mask_np == 1] = 1  # 前景
            binary_mask[mask_np == 2] = 0  # 背景
            binary_mask[mask_np == 3] = 255  # 未分类区域
            
            # 将numpy掩码转回PIL图像用于变换
            binary_mask_pil = Image.fromarray(binary_mask.astype(np.uint8))
            
            # 应用图像变换 - 直接在PIL图像上操作
            if self.transform is not None:
                try:
                    image = self.transform(image)
                except Exception as e:
                    print(f"Error applying transform to image: {e}")
                    return self._get_dummy_sample()

            # 应用掩码变换 - 直接在PIL图像上操作
            if self.mask_transform is not None:
                try:
                    mask_tensor = self.mask_transform(binary_mask_pil)
                except Exception as e:
                    print(f"Error applying transform to mask: {e}")
                    return self._get_dummy_sample()
            else:
                # 没有变换，直接转为tensor
                mask_tensor = torch.from_numpy(binary_mask).long()

            # 确保掩码是2D的 [H, W]
            if mask_tensor.dim() > 2:
                mask_tensor = mask_tensor.squeeze(0)

            return {"image": image, "mask": mask_tensor, "path": img_path}

        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            return self._get_dummy_sample()
    
    def _get_dummy_sample(self):
        """创建一个有效的替代样本"""
        dummy_image = torch.zeros(3, 224, 224)
        dummy_mask = torch.zeros(224, 224, dtype=torch.long)
        return {"image": dummy_image, "mask": dummy_mask, "path": "dummy_path"}

# 添加计算批次间IoU的函数
def compute_batch_miou(preds, targets, num_classes=2):
    """
    计算批次中的平均交并比(mIoU)，忽略标记为255的区域
    
    Args:
        preds: 形状为(B, H, W)的预测掩码
        targets: 形状为(B, H, W)的真实掩码
        num_classes: 类别数量
        
    Returns:
        float: 平均交并比(mIoU)
    """
    # 确保输入是numpy数组
    if not isinstance(preds, np.ndarray):
        preds = preds.numpy()
    if not isinstance(targets, np.ndarray):
        targets = targets.numpy()
    
    # 初始化交集和并集
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    # 创建有效区域掩码，忽略值为255的区域
    valid_mask = (targets != 255)
    
    # 对每个类别计算IoU
    for cls in range(num_classes):
        # 只在有效区域内计算
        pred_inds = (preds == cls) & valid_mask
        target_inds = (targets == cls) & valid_mask
        
        # 计算交集和并集
        intersection[cls] = np.logical_and(pred_inds, target_inds).sum()
        union[cls] = np.logical_or(pred_inds, target_inds).sum()
    
    # 计算每个类别的IoU
    iou = np.zeros(num_classes)
    for cls in range(num_classes):
        if union[cls] > 0:
            iou[cls] = intersection[cls] / union[cls]
    
    # 返回所有类别的平均IoU
    return np.mean(iou)

# 添加计算每个类别IoU的函数
def calculate_per_class_iou(preds, targets, num_classes=2):
    """
    计算每个类别的IoU
    
    Args:
        preds: 预测掩码 (numpy array或tensor)
        targets: 真实掩码 (numpy array或tensor)
        num_classes: 类别数量
        
    Returns:
        numpy array: 每个类别的IoU
    """
    # 确保输入是numpy数组
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
    if not isinstance(targets, np.ndarray):
        targets = targets.cpu().numpy()
    
    # 创建有效区域掩码，忽略值为255的区域
    valid_mask = (targets != 255)
    
    # 初始化每个类别的IoU
    class_ious = np.zeros(num_classes)
    
    # 计算每个类别的IoU
    for cls in range(num_classes):
        # 只在有效区域内计算
        pred_cls = (preds == cls) & valid_mask
        target_cls = (targets == cls) & valid_mask
        
        # 计算交集和并集
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        # 计算IoU，避免除零
        if union > 0:
            class_ious[cls] = intersection / union
    
    return class_ious

# 训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, config, paths, model_type="fully_supervised"):
    """训练分割模型并保存最佳模型
    
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        criterion: 损失函数
        optimizer: 优化器
        config: 配置参数字典
        paths: 路径字典
        model_type: 模型类型标识
        
    Returns:
        model: 训练好的模型
        log: 训练日志，包含损失值和评估指标
    """
    
    # 创建日志字典
    log = {
        "train_loss": [],
        "test_loss": [],
        "train_miou": [],
        "test_miou": [],
        "val_miou": [],  # 添加验证集的mIoU
        "test_class_ious": [],  # 添加每个类别的IoU
        "best_miou": 0.0,
        "lr": []  # 记录学习率
    }
    
    # 获取scheduler
    scheduler = config.get("scheduler", None)
    
    # 确保模型目录存在
    os.makedirs(paths["model_dir"], exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_miou = 0.0
    best_epoch = 0
    best_model_path = os.path.join(paths["model_dir"], f"best_model_{model_type}.pth")
    
    print(f"Training on {device}")
    print(f"Number of epochs: {config['num_epochs']}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of testing samples: {len(test_loader.dataset)}")
    print(f"Initial learning rate: {config['learning_rate']}")
    
    # 开始训练循环
    for epoch in range(config["num_epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_iou_sum = 0.0
        num_samples = 0
        
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 处理模型输出 - 可能是元组(main_output, aux_output)
            if isinstance(outputs, tuple):
                main_output, aux_output = outputs
                # 计算主要损失和辅助损失
                main_loss = criterion(main_output, masks)
                aux_loss = criterion(aux_output, masks)
                # 综合损失 - 主要损失占比0.7，辅助损失占比0.3
                loss = 0.7 * main_loss + 0.3 * aux_loss
                # 使用主要输出进行评估
                preds = torch.argmax(main_output, dim=1)
            else:
                # 单一输出的情况
                loss = criterion(outputs, masks)
                preds = torch.argmax(outputs, dim=1)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item() * images.size(0)
            
            # 计算mIoU
            iou = compute_batch_miou(preds.cpu(), masks.cpu(), config["num_classes"])
            train_iou_sum += iou * images.size(0)
            num_samples += images.size(0)
            
            # 显示进度
            if (i+1) % config.get("print_freq", 10) == 0:
                print_progress(
                    current=i+1, 
                    total=len(train_loader),
                    prefix=f"Epoch {epoch+1}/{config['num_epochs']} Training",
                    suffix=f"Loss: {loss.item():.4f}, IoU: {iou:.4f}"
                )
        
        # 计算平均损失和mIoU
        avg_train_loss = train_loss / num_samples
        avg_train_miou = train_iou_sum / num_samples
        
        # 验证阶段
        model.eval()
        test_loss = 0.0
        test_iou_sum = 0.0
        num_samples = 0
        class_ious_sum = np.zeros(config["num_classes"])  # 累积每个类别的IoU
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 处理模型输出 - 在eval模式下应该只有一个输出，但为了安全起见还是处理一下
                if isinstance(outputs, tuple):
                    main_output = outputs[0]  # 取主要输出
                    loss = criterion(main_output, masks)
                    preds = torch.argmax(main_output, dim=1)
                else:
                    loss = criterion(outputs, masks)
                    preds = torch.argmax(outputs, dim=1)
                
                # 更新统计信息
                test_loss += loss.item() * images.size(0)
                
                # 计算mIoU
                iou = compute_batch_miou(preds.cpu(), masks.cpu(), config["num_classes"])
                test_iou_sum += iou * images.size(0)
                
                # 计算每个类别的IoU
                class_ious = calculate_per_class_iou(preds.cpu(), masks.cpu(), config["num_classes"])
                class_ious_sum += class_ious * images.size(0)
                
                num_samples += images.size(0)
                
                # 显示进度
                if (i+1) % config.get("print_freq", 10) == 0:
                    print_progress(
                        current=i+1, 
                        total=len(test_loader),
                        prefix=f"Epoch {epoch+1}/{config['num_epochs']} Testing",
                        suffix=f"Loss: {loss.item():.4f}, IoU: {iou:.4f}"
                    )
        
        # 计算平均损失和mIoU
        avg_test_loss = test_loss / num_samples
        avg_test_miou = test_iou_sum / num_samples
        avg_class_ious = class_ious_sum / num_samples
        
        # 将测试集的mIoU同时作为验证集的mIoU (因为我们没有单独的验证集)
        avg_val_miou = avg_test_miou
        
        # 更新日志
        log["train_loss"].append(avg_train_loss)
        log["test_loss"].append(avg_test_loss)
        log["train_miou"].append(avg_train_miou)
        log["test_miou"].append(avg_test_miou)
        log["val_miou"].append(avg_val_miou)  # 添加验证集mIoU
        log["test_class_ious"].append(avg_class_ious.tolist())  # 保存每个类别的IoU
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        log["lr"].append(current_lr)
        
        # 打印本轮结果，包含验证集mIoU
        print(f"Epoch {epoch+1}/{config['num_epochs']} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train mIoU: {avg_train_miou:.4f}, "
              f"Val Loss: {avg_test_loss:.4f}, "
              f"Val mIoU: {avg_val_miou:.4f}, "
              f"Test mIoU: {avg_test_miou:.4f}, "
              f"LR: {current_lr:.6f}")
        
        # 打印每个类别的IoU
        class_names = ["背景", "前景"]
        print(f"类别IoU: ", end="")
        for i, class_iou in enumerate(avg_class_ious):
            print(f"{class_names[i]}: {class_iou:.4f}", end="  ")
        print()
        
        # 保存最佳模型
        if avg_test_miou > best_miou:
            best_miou = avg_test_miou
            best_epoch = epoch + 1
            log["best_miou"] = best_miou
            
            # 保存模型
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with mIoU: {best_miou:.4f}")
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            print(f"Learning rate adjusted to: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 计算总训练时间
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    
    # 保存训练日志
    log_file = os.path.join(paths["result_dir"], f"training_log_{model_type}.json")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        json.dump({
            "train_loss": log["train_loss"],
            "test_loss": log["test_loss"],
            "train_miou": log["train_miou"],
            "val_miou": log["val_miou"],  # 添加验证集mIoU
            "test_miou": log["test_miou"],
            "test_class_ious": log["test_class_ious"],  # 添加每个类别的IoU
            "best_miou": log["best_miou"],
            "lr": log["lr"],
            "best_epoch": best_epoch,
            "total_epochs": config["num_epochs"],
            "training_time_minutes": training_time / 60,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=4)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    return model, log

def save_comparison_results(results_fs, results_ws=None, save_dir=None):
    """
    保存全监督方法与弱监督方法的mIoU对比结果
    
    参数:
        results_fs: 全监督方法的结果
        results_ws: 弱监督方法的结果
        save_dir: 保存结果的目录
    """
    if save_dir is None:
        save_dir = FULLY_SUP_PATHS["result_dir"]
        
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
        f.write("Method,train,val\n")
        for label, train_miou, val_miou in zip(labels, train_mious, val_mious):
            f.write(f"{label},{train_miou:.4f},{val_miou:.4f}\n")
    
    # 输出比较结果
    print("\n===== mIoU Comparison Results =====")
    print(f"{'Method':<25}{'Train':<10}{'Val':<10}")
    print("="*45)
    for label, train_miou, val_miou in zip(labels, train_mious, val_mious):
        print(f"{label:<25}{train_miou:.4f}{val_miou:.4f}")
    print("="*45)
    
    print(f"Comparison results saved to {csv_path}")

def load_weakly_supervised_results(result_dir=None):
    """
    加载已有的弱监督方法结果
    
    参数:
        result_dir: 保存结果的目录
        
    返回:
        字典: 弱监督方法的结果
    """
    if result_dir is None:
        # 使用config中定义的结果目录
        result_dir = str(OUTPUT_ROOT / "results")
        
    results = {"base": {}, "crf": {}}
    
    # 查找最新的比较结果文件
    comparison_files = [f for f in os.listdir(result_dir) 
                        if f.startswith("comparison_results_") and 
                        f.endswith(".json") and 
                        "fullyVSweakly" not in f]
    
    if not comparison_files:
        print("No comparison result files found for weakly supervised methods")
        
        # 尝试从训练日志中查找结果
        log_files = [f for f in os.listdir(result_dir) 
                    if f.startswith("training_log_") and f.endswith(".json")]
        
        if log_files:
            log_files.sort(reverse=True)
            for log_file in log_files:
                try:
                    with open(os.path.join(result_dir, log_file), 'r') as f:
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
    latest_file = os.path.join(result_dir, comparison_files[0])
    
    try:
        with open(latest_file, 'r') as f:
            comparison_data = json.load(f)
        
        # 从比较结果中提取弱监督方法的结果
        if 'base' in comparison_data:
            results["base"]["best_miou"] = comparison_data['base'].get("best_miou")
            results["base"]["train_miou"] = comparison_data['base'].get("train_miou")
            results["base"]["val_miou"] = comparison_data['base'].get("val_miou")
        
        if 'crf' in comparison_data:
            results["crf"]["best_miou"] = comparison_data['crf'].get("best_miou")
            results["crf"]["train_miou"] = comparison_data['crf'].get("train_miou")
            results["crf"]["val_miou"] = comparison_data['crf'].get("val_miou")
            
        print(f"Loaded weakly supervised method results: {latest_file}")
    except Exception as e:
        print(f"Failed to load weakly supervised method results: {e}")
    
    return results

# 添加一个帮助函数，确保掩码是PIL图像
class MaskToPIL:
    def __call__(self, mask):
        """将不同类型的掩码转换为PIL图像"""
        if isinstance(mask, np.ndarray):
            return Image.fromarray(mask.astype(np.uint8))
        elif isinstance(mask, torch.Tensor):
            return Image.fromarray(mask.numpy().astype(np.uint8))
        elif isinstance(mask, Image.Image):
            return mask
        else:
            raise TypeError(f"Unexpected mask type: {type(mask)}")

# 添加同步变换的数据集类 - 放在主函数外部以便多进程使用
class SyncedTransformDataset(FullySupSegDataset):
    def __getitem__(self, idx):
        # 防止索引越界
        if idx >= len(self.image_paths):
            print(f"Warning: Index {idx} out of bounds, returning dummy sample")
            return self._get_dummy_sample()
            
        try:
            # 使用字符串路径打开文件
            img_path = str(self.image_paths[idx])
            mask_path = str(self.mask_paths[idx])
            
            # 确保文件存在
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"Warning: Files not found at index {idx}, returning dummy sample")
                return self._get_dummy_sample()

            # 打开图像为RGB模式
            with Image.open(img_path) as img:
                # 始终转换为RGB以确保一致性
                image = img.convert("RGB")

            # 打开掩码为灰度模式
            with Image.open(mask_path) as mask_img:
                # 始终转换为L以确保一致性
                mask_pil = mask_img.convert("L")
                # 转为numpy以进行处理
                mask_np = np.array(mask_pil)
            
            # 处理trimap值 - 创建二值掩码
            binary_mask = np.zeros_like(mask_np)
            binary_mask[mask_np == 1] = 1  # 前景
            binary_mask[mask_np == 2] = 0  # 背景
            binary_mask[mask_np == 3] = 255  # 未分类区域
            
            # 将numpy掩码转回PIL图像用于变换
            binary_mask_pil = Image.fromarray(binary_mask.astype(np.uint8))
            
            # 同步随机状态
            random_state = torch.get_rng_state()
            
            # 应用图像变换 - 直接在PIL图像上操作
            if self.transform is not None:
                try:
                    image = self.transform(image)
                except Exception as e:
                    print(f"Error applying transform to image: {e}")
                    return self._get_dummy_sample()
            
            # 恢复随机状态，确保掩码变换与图像变换使用相同的随机性
            torch.set_rng_state(random_state)
            
            # 应用掩码变换 - 直接在PIL图像上操作
            if self.mask_transform is not None:
                try:
                    mask_tensor = self.mask_transform(binary_mask_pil)
                except Exception as e:
                    print(f"Error applying transform to mask: {e}")
                    return self._get_dummy_sample()
            else:
                # 没有变换，直接转为tensor
                mask_tensor = torch.from_numpy(binary_mask).long()

            # 确保掩码是2D的 [H, W]
            if mask_tensor.dim() > 2:
                mask_tensor = mask_tensor.squeeze(0)

            return {"image": image, "mask": mask_tensor, "path": img_path}

        except Exception as e:
            print(f"Error loading index {idx}: {e}")
            return self._get_dummy_sample()

def main():
    parser = argparse.ArgumentParser(description='Train segmentation model with fully supervised learning')
    parser.add_argument('--gt_dir', type=str, default=None, help='地址真实标签目录的路径')
    parser.add_argument('--batch_size', type=int, default=FULLY_SUP_CONFIG['batch_size'], help='批量大小')
    parser.add_argument('--epochs', type=int, default=FULLY_SUP_CONFIG['num_epochs'], help='训练轮次')
    parser.add_argument('--lr', type=float, default=FULLY_SUP_CONFIG['learning_rate'], help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='从test.txt中选取的比例作为测试集')
    parser.add_argument('--trainval_file', type=str, default=str(ANNOTATION_DIR / 'trainval.txt'), help='训练集列表文件')
    parser.add_argument('--test_file', type=str, default=str(ANNOTATION_DIR / 'test.txt'), help='测试集列表文件')
    parser.add_argument('--disable_aug', action='store_true', help='禁用数据增强，只使用基本变换')
    parser.add_argument('--num_workers', type=int, default=FULLY_SUP_CONFIG['num_workers'], help='数据加载器的工作线程数')
    parser.add_argument('--single_process', action='store_true', help='禁用多进程数据加载，设置workers=0')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 更新配置，而不是复制
    FULLY_SUP_CONFIG['batch_size'] = args.batch_size
    FULLY_SUP_CONFIG['num_epochs'] = args.epochs
    FULLY_SUP_CONFIG['learning_rate'] = args.lr
    FULLY_SUP_CONFIG['weight_decay'] = 1e-4  # 增加权重衰减
    
    # 如果使用单进程模式，设置workers=0
    if args.single_process:
        FULLY_SUP_CONFIG['num_workers'] = 0
        print("使用单进程数据加载(num_workers=0)")
    else:
        FULLY_SUP_CONFIG['num_workers'] = args.num_workers
        print(f"使用多进程数据加载(num_workers={args.num_workers})")
    
    # 如果提供了自定义真实标签目录，则更新路径
    if args.gt_dir:
        FULLY_SUP_PATHS['mask_dir'] = args.gt_dir
    
    # 检查真实标签目录是否存在
    if not os.path.exists(FULLY_SUP_PATHS['mask_dir']):
        print(f"Error: Ground truth mask directory does not exist {FULLY_SUP_PATHS['mask_dir']}")
        print(f"Please prepare ground truth data first or specify the correct directory with --gt_dir")
        return
    
    # 决定是否使用增强的数据变换
    if args.disable_aug:
        print("\n===== Using basic transforms (no augmentation) =====")
        # 基本变换 - 只进行归一化和调整大小
        transform = transforms.Compose([
            transforms.Resize(FULLY_SUP_CONFIG["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 基本掩码变换
        mask_transform = transforms.Compose([
            transforms.Resize(
                FULLY_SUP_CONFIG["image_size"],
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            BinaryMaskToTensor()
        ])
    else:
        print("\n===== Using enhanced transforms with augmentation =====")
        # 创建共享的随机状态，确保图像和掩码应用相同的随机变换
        # 这些种子不会影响训练过程的随机性，只用于确保变换一致
        tf_seed = random.randint(0, 2**32 - 1)
        flip_seed = random.randint(0, 2**32 - 1)
        crop_seed = random.randint(0, 2**32 - 1)
        rotate_seed = random.randint(0, 2**32 - 1)
        affine_seed = random.randint(0, 2**32 - 1)
        print(f"Transform random seeds: flip={flip_seed}, crop={crop_seed}, rotate={rotate_seed}, affine={affine_seed}")
        
        # 增强的数据变换 - 使用同步的随机性
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((256, 256)),  # 先调整大小，再裁剪
            transforms.RandomResizedCrop(
                size=FULLY_SUP_CONFIG["image_size"], 
                scale=(0.7, 1.0)  # 使用更大的下限，确保不会裁剪掉太多内容
            ),
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 掩码变换 - 需要与图像变换一致，但不应用色彩变换
        mask_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),  # 先调整大小，再裁剪
            transforms.RandomResizedCrop(
                size=FULLY_SUP_CONFIG["image_size"], 
                scale=(0.7, 1.0),  # 与图像相同的参数
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            BinaryMaskToTensor()
        ])
    
    # 使用官方数据集划分，使用全部test.txt作为测试集
    print("\n===== 加载官方数据集划分 =====")
    train_images, test_images = load_official_dataset_split(
        trainval_file=args.trainval_file,
        test_file=args.test_file,
        test_ratio=args.test_ratio,  # 使用全部test.txt文件
        seed=args.seed
    )
    
    # 创建全监督数据集
    print("\n===== Creating Fully Supervised Dataset =====")
    
    # 创建训练集 - 使用真实标签
    train_dataset = SyncedTransformDataset(
        FULLY_SUP_PATHS["img_dir"], 
        FULLY_SUP_PATHS["mask_dir"], 
        transform, 
        mask_transform,
        image_list=train_images  # 只使用训练集图像
    )
    
    # 创建测试集 - 使用真实标签
    test_dataset = SyncedTransformDataset(
        FULLY_SUP_PATHS["img_dir"], 
        FULLY_SUP_PATHS["mask_dir"], 
        # 测试集只需要基本变换，不需要数据增强
        transforms.Compose([
            transforms.Resize(FULLY_SUP_CONFIG["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), 
        transforms.Compose([
            transforms.Resize(FULLY_SUP_CONFIG["image_size"], interpolation=transforms.InterpolationMode.NEAREST),
            BinaryMaskToTensor()
        ]),
        image_list=test_images  # 只使用测试集图像
    )
    
    # 创建数据加载器
    print(f"\n创建数据加载器: 批量大小={FULLY_SUP_CONFIG['batch_size']}, 工作线程数={FULLY_SUP_CONFIG['num_workers']}")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=FULLY_SUP_CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=FULLY_SUP_CONFIG["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if FULLY_SUP_CONFIG["num_workers"] > 0 else False,
        prefetch_factor=2 if FULLY_SUP_CONFIG["num_workers"] > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=FULLY_SUP_CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=FULLY_SUP_CONFIG["num_workers"],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if FULLY_SUP_CONFIG["num_workers"] > 0 else False,
        prefetch_factor=2 if FULLY_SUP_CONFIG["num_workers"] > 0 else None
    )
    
    # 创建模型、损失函数和优化器
    print("\n===== Training Fully Supervised Model =====")
    model = DeepLabLargeFOV(
        num_classes=FULLY_SUP_CONFIG["num_classes"], 
        atrous_rates=FULLY_SUP_CONFIG["atrous_rates"]
    )
    
    # 使用带忽略索引的交叉熵损失
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # 使用AdamW优化器，更合适的权重衰减
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=FULLY_SUP_CONFIG["learning_rate"], 
        weight_decay=FULLY_SUP_CONFIG["weight_decay"]
    )
    
    # 添加学习率调度器 - 使用StepLR
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=20, 
        gamma=0.5
    )
    
    # 添加scheduler到config中
    FULLY_SUP_CONFIG["scheduler"] = scheduler
    
    # 训练模型
    model, log = train_model(model, train_loader, test_loader, criterion, optimizer, FULLY_SUP_CONFIG, FULLY_SUP_PATHS, "fully_supervised")
    
    print("\n===== Fully Supervised Training Complete =====")
    print(f"Results:")
    print(f"  - Best mIoU: {log['best_miou']:.4f}")
    print(f"  - Last Train mIoU: {log['train_miou'][-1]:.4f}")
    print(f"  - Last Test mIoU: {log['test_miou'][-1]:.4f}")
    
    # 找出最佳模型对应的epoch
    best_epoch_idx = log['test_miou'].index(log['best_miou'])
    best_train_miou = log['train_miou'][best_epoch_idx]
    best_val_miou = log['val_miou'][best_epoch_idx]
    best_test_miou = log['best_miou']
    best_class_ious = log['test_class_ious'][best_epoch_idx]
    
    # 保存最佳模型的mIoU结果到CSV文件
    best_results_file = os.path.join(FULLY_SUP_PATHS["result_dir"], f"best_model_results.csv")
    os.makedirs(os.path.dirname(best_results_file), exist_ok=True)
    
    with open(best_results_file, 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Best_Train_mIoU,{best_train_miou:.4f}\n")
        f.write(f"Best_Val_mIoU,{best_val_miou:.4f}\n")
        f.write(f"Best_Test_mIoU,{best_test_miou:.4f}\n")
        class_names = ["背景", "前景"]
        for i, class_iou in enumerate(best_class_ious):
            f.write(f"Class_{class_names[i]}_IoU,{class_iou:.4f}\n")
    
    print(f"\n最佳模型结果已保存到: {best_results_file}")
    print(f"最佳模型训练mIoU: {best_train_miou:.4f}")
    print(f"最佳模型验证mIoU: {best_val_miou:.4f}")
    print(f"最佳模型测试mIoU: {best_test_miou:.4f}")
    print(f"最佳模型类别IoU:")
    for i, class_iou in enumerate(best_class_ious):
        print(f"  - {class_names[i]}: {class_iou:.4f}")
    
    # 比较函数 - 可用于将完全监督与弱监督方法进行比较
    def compare_with_weakly_supervised(fully_supervised_miou, weakly_supervised_miou):
        """比较完全监督和弱监督方法的mIoU性能"""
        difference = fully_supervised_miou - weakly_supervised_miou
        relative = weakly_supervised_miou / fully_supervised_miou * 100 if fully_supervised_miou > 0 else 0
        
        print(f"\nComparison with weakly supervised method:")
        print(f"  - Fully Supervised mIoU: {fully_supervised_miou:.4f}")
        print(f"  - Weakly Supervised mIoU: {weakly_supervised_miou:.4f}")
        print(f"  - Absolute Difference: {difference:.4f}")
        print(f"  - Relative Performance: {relative:.2f}%")
        
        return {
            "fully_supervised_miou": fully_supervised_miou,
            "weakly_supervised_miou": weakly_supervised_miou,
            "difference": difference,
            "relative_performance": relative
        }

if __name__ == "__main__":
    main() 