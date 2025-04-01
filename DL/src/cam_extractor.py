import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
import torchvision.transforms as transforms
import cv2

from config import CLASSIFIER_CONFIG, CAM_DIR, CLASSIFIER_DIR, IMAGE_DIR, DATASET_CONFIG, FAST_TEST_CONFIG, CAM_CONFIG
from utils import get_dataloaders, visualize_cam, save_cam, set_seed

class CAMModel(nn.Module):
    """
    轻量级CAM模型，使用ResNet18作为骨干网络，支持混合精度训练
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(CAMModel, self).__init__()
        # 使用更轻量的ResNet18作为backbone
        backbone = CLASSIFIER_CONFIG.get("backbone", "resnet18")
        if backbone == "resnet18":
            base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 512  # ResNet18最后一层特征维度
        else:
            base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 2048  # ResNet50最后一层特征维度
        
        # 提取多尺度特征 (保留中间层特征)
        self.layer1 = nn.Sequential(*list(base_model.children())[:5])  # 浅层特征
        self.layer2 = base_model.layer2  # 中层特征
        self.layer3 = base_model.layer3  # 深层特征
        self.layer4 = base_model.layer4  # 最深层特征
        
        # 全局池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # 通道注意力机制 - 轻量级实现
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.feature_dim, self.feature_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim // 16, self.feature_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 边缘检测器
        self.edge_detector = nn.Conv2d(self.feature_dim, self.feature_dim, 
                                       kernel_size=3, padding=1, 
                                       groups=self.feature_dim//4,  # 组卷积
                                       bias=False)
        
        # 特征融合
        self.fusion = nn.Conv2d(self.feature_dim, self.feature_dim, 
                               kernel_size=1, bias=False)
    
    def forward(self, x, return_cam=False):
        # 提取多尺度特征
        x1 = self.layer1(x)       # 浅层特征
        x2 = self.layer2(x1)      # 中层特征
        x3 = self.layer3(x2)      # 深层特征
        features = self.layer4(x3) # 最深层特征
        
        # 应用通道注意力
        channel_att = self.channel_attention(features)
        features = features * channel_att
        
        # 应用空间注意力
        spatial_max, _ = torch.max(features, dim=1, keepdim=True)
        spatial_avg = torch.mean(features, dim=1, keepdim=True)
        spatial_concat = torch.cat([spatial_max, spatial_avg], dim=1)
        spatial_att = self.spatial_attention(spatial_concat)
        features = features * spatial_att
        
        # 边缘增强 - 检测高频信息
        edge_features = self.edge_detector(features)
        edge_weights = torch.abs(features - edge_features)
        
        # 特征融合
        features = self.fusion(features + edge_features)
        
        # 全局池化用于分类
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        logits = self.fc(pooled)
        
        if return_cam:
            # 获取梯度权重 (Grad-CAM++ 核心)
            batch_size = features.size(0)
            grad_cams = []
            
            for b in range(batch_size):
                # 为每个样本单独计算 Grad-CAM++
                cam_batch = self._compute_gradcampp(features[b:b+1], logits[b:b+1], edge_weights[b:b+1])
                grad_cams.append(cam_batch)
            
            cams = torch.cat(grad_cams, dim=0)
            probas = F.softmax(logits, dim=1)
            
            return logits, cams, probas
        
        return logits
    
    def _compute_gradcampp(self, features, logits, edge_weights):
        """计算改进的Grad-CAM++，结合边缘信息和扩展身体区域"""
        batch_size, n_channels, height, width = features.size()
        n_classes = logits.size(1)
        
        # 使用预训练权重代替梯度计算，更稳定
        class_weights = self.fc.weight  # [num_classes, feature_dim]
        
        # 生成基础CAM
        cam = torch.einsum('nc,bchw->bnhw', class_weights, features)
        
        # 获取前景类别的CAM (索引1对应狗)
        if n_classes > 1:  # 确保是多类别的情况
            # 增强前景类别的权重
            foreground_bias = CAM_CONFIG.get("foreground_bias", 1.5)
            # 仅增强前景类别
            cam[:, 1] = cam[:, 1] * foreground_bias
        
        # 边缘增强
        edge_factor = edge_weights.mean(dim=1, keepdim=True)
        edge_strength = CAM_CONFIG.get("edge_enhancement", 1.5)
        cam = cam * (1 + edge_strength * edge_factor)
        
        # 对每个类别单独进行处理
        for c in range(n_classes):
            # 归一化每个类别的CAM
            c_cam = cam[:, c:c+1]
            min_val = c_cam.min()
            max_val = c_cam.max() 
            if max_val > min_val:
                c_cam = (c_cam - min_val) / (max_val - min_val)
            cam[:, c:c+1] = c_cam
            
        return cam

class CAMExtractor:
    """
    用于训练分类器和提取类激活图(CAM)的类。
    
    属性:
        config: 配置字典
        device: 运行模型的设备
        model: ResNet50模型实例
        optimizer: 用于训练的优化器
        criterion: 损失函数
        scaler: 用于混合精度训练的梯度缩放器
        scheduler: 学习率调度器
    """
    
    def __init__(self, config=None):
        """
        初始化CAM提取器。
        
        参数:
            config: 用于训练的配置字典
        """
        self.config = config if config is not None else CLASSIFIER_CONFIG
        self.device = torch.device(self.config['device'])
        self.model_path = Path(CLASSIFIER_DIR) / 'cam_model.pth'
        
        # 创建CAM模型
        self.model = CAMModel(num_classes=self.config['num_classes'], 
                               pretrained=self.config.get('use_pretrained', True))
        
        # 转换模型到目标设备
        self.model.to(self.device)
        
        # 设置优化器和学习率调度器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # 使用带权重的交叉熵损失函数
       
        try:
            from torch.optim import Lion
            self.optimizer = Lion(
                self.model.parameters(),
                lr=config.get('lr', 0.0005),  
                weight_decay=config.get('weight_decay', 1e-4)
            )
            print("Using Lion optimizer for faster convergence")
        except ImportError:
            print("Lion optimizer not available, using AdamW")
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.get('lr', 0.0001),
                weight_decay=config.get('weight_decay', 1e-4)
            )
        
        # 设置损失函数
        if config.get('weighted_loss', True):
            # 给前景类别2倍权重
            class_weights = torch.tensor([1.0, 2.0]).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Using weighted loss function, weights:", class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 创建目录
        Path(CLASSIFIER_DIR).mkdir(parents=True, exist_ok=True)
        Path(CAM_DIR).mkdir(parents=True, exist_ok=True)
        
        # 模型路径
        self.model_path = Path(CLASSIFIER_DIR) / "classifier_model.pth"
        
       
        if hasattr(amp, 'GradScaler'):
            self.scaler = amp.GradScaler()
        else:
        
            try:
                self.scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
            except:
               
                self.scaler = amp.GradScaler()
        
        # 添加带预热的余弦退火学习率
        total_epochs = config['epochs']
        warmup_epochs = int(total_epochs * 0.1)  # 10%的训练周期用于预热
        self.scheduler = self._create_scheduler(total_epochs, warmup_epochs)
    
    def _create_scheduler(self, epochs, warmup_epochs=0):
        """
        创建带余弦退火的学习率调度器。
        
        参数:
            epochs: 训练轮次总数
            warmup_epochs: 预热轮次数
            
        返回:
            torch.optim.lr_scheduler: 学习率调度器
        """
        if self.config.get('lr_scheduler') == 'cosine':
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    # 线性预热
                    return epoch / max(1, warmup_epochs)
                else:
                    # 余弦退火
                    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            # 默认使用普通的余弦退火
            return CosineAnnealingLR(self.optimizer, T_max=epochs)
    
    def train(self, train_loader, val_loader, epochs=None):
        """
        使用混合精度训练分类模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮次
            
        返回:
            float: 最佳验证准确率
        """
        if epochs is None:
            epochs = self.config['epochs']
        
        print(f"Training classifier for {epochs} epochs with mixed precision...")
        
        best_val_acc = 0.0
        epochs_without_improvement = 0
        patience = self.config.get('early_stopping_patience', 5)
        
        # 启用混合精度训练
        use_amp = self.config.get('use_mixed_precision', True) and torch.cuda.is_available()
        scaler = self.scaler
        
        # 使用非阻塞数据传输
        non_blocking = self.config.get('non_blocking', True)
        
        # 设置CUDA环境
        if torch.cuda.is_available():
            # 设置CUDA缓存分配器以提高内存效率
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True  # 启用自动算法选择
        
        print(f"Using mixed precision: {use_amp}, batch size: {self.config['batch_size']}")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # 使用进度条更新
            print(f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(train_loader):
                # 使用非阻塞数据传输
                images = batch['image'].to(self.device, non_blocking=non_blocking)
                labels = batch['label'].to(self.device, non_blocking=non_blocking)
                
                # 清零梯度
                self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # 使用梯度缩放器进行反向传播和优化
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                # 计算训练指标
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # 定期记录训练状态
                if (batch_idx + 1) % 20 == 0 or batch_idx == len(train_loader) - 1:
                    train_acc = 100.0 * train_correct / train_total
                    avg_loss = train_loss / (batch_idx + 1)
                    print(f"  Batch {batch_idx+1}/{len(train_loader)}: loss={avg_loss:.4f}, accuracy={train_acc:.2f}%")
            
            # 更新学习率
            self.scheduler.step()
            
            # 计算整体训练指标
            train_acc = 100.0 * train_correct / train_total
            avg_loss = train_loss / len(train_loader)
            
            # 验证阶段
            val_acc = self.evaluate(val_loader)
            
            # 打印轮次摘要
            print(f"  Epoch {epoch+1}/{epochs} - Train loss: {avg_loss:.4f}, Train accuracy: {train_acc:.2f}%, Validation accuracy: {val_acc:.2f}%")
            
            # 检查是否为最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                print(f"  Saving model, validation accuracy: {val_acc:.2f}%")
                torch.save(self.model.state_dict(), self.model_path)
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epochs")
            
            # 早停检查
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # 加载最佳模型
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        if best_val_acc > 0:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        return best_val_acc
    
    def evaluate(self, dataloader):
        """
        评估模型在给定数据集上的准确率
        
        参数:
            dataloader: 数据加载器
            
        返回:
            float: 准确率百分比
        """
        self.model.eval()
        correct = 0
        total = 0
        
        # 使用更大的批量进行评估
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config.get('use_mixed_precision', True)):
            for batch in dataloader:
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def generate_cams(self, data_loader, save_dir=None):
        """
        生成和保存类激活图 (CAMs)
        
        参数:
            data_loader: 用于生成CAM的数据加载器
            save_dir: 保存CAM的目录
            
        返回:
            tuple: (CAM列表, 图像路径列表)
        """
        if save_dir is None:
            save_dir = CAM_DIR
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        
        all_cams = []
        all_image_paths = []
        
        print(f"Generating CAMs for {len(data_loader.dataset)} images...")
        
        # 启用CUDA优化
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config.get('use_mixed_precision', True)):
            for batch_idx, batch in enumerate(data_loader):
                # 定期打印进度
                if batch_idx % 10 == 0 or batch_idx == len(data_loader) - 1:
                    print(f"  Processing batch {batch_idx+1}/{len(data_loader)} ({(batch_idx+1)/len(data_loader)*100:.1f}%)")
                
                # 高效数据加载
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                image_paths = batch['image_path']
                
                # 在单次前向传播中生成所有CAM
                outputs, cams, probas = self.model(images, return_cam=True)
                _, predicted = outputs.max(1)
                
                # 批量处理CAM和保存操作
                for i, (image, cam, pred, image_path) in enumerate(zip(images, cams, predicted, image_paths)):
                    try:
                        # 获取预测类别的CAM
                        cam_pred = cam[pred].cpu().numpy()
                        
                        # 调整CAM大小到原图尺寸
                        img_size = self.config['image_size']
                        cam_pred = torch.nn.functional.interpolate(
                            torch.tensor(cam_pred).unsqueeze(0).unsqueeze(0),
                            size=img_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze().numpy()
                        
                        # 归一化和处理CAM
                        cam_pred = cam_pred - cam_pred.min()
                        cam_max = cam_pred.max()
                        if cam_max > 0:
                            cam_pred = cam_pred / cam_max
                        else:
                            cam_pred = np.zeros_like(cam_pred)
                        
                        # 应用配置的阈值处理
                        if CAM_CONFIG.get("use_adaptive_threshold", False):
                            # 使用自适应阈值
                            cam_uint8 = (cam_pred * 255).astype(np.uint8)
                            otsu_threshold, _ = cv2.threshold(
                                cam_uint8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU
                            )
                            threshold = 0.7 * (otsu_threshold / 255.0) + 0.3 * CAM_CONFIG["threshold"]
                            cam_pred[cam_pred < threshold] = 0
                        else:
                            # 使用固定阈值
                            cam_pred[cam_pred < CAM_CONFIG["threshold"]] = 0
                        
                        # 应用gamma校正和形态学处理
                        cam_pred = np.power(cam_pred, CAM_CONFIG["gamma"])
                        
                        if CAM_CONFIG.get("use_morphology", True):
                            cam_binary = (cam_pred > 0).astype(np.uint8)
                            kernel = np.ones((3, 3), np.uint8)
                            cam_binary = cv2.morphologyEx(cam_binary, cv2.MORPH_CLOSE, kernel)
                            cam_pred = cam_pred * cam_binary
                        
                        # 应用身体区域扩展
                        cam_pred = _expand_head_to_body(cam_pred)
                        
                        # 异步保存PNG和NPY文件
                        image_name = Path(image_path).stem
                        cam_image_path = save_dir / f"{image_name}_cam.png"
                        cam_data_path = save_dir / f"{image_name}_cam.npy"
                        
                        # 保存PNG图像
                        try:
                            visualize_cam(image.cpu(), cam_pred, save_path=cam_image_path)
                        except Exception as e:
                            print(f"Failed to save PNG for {image_path}: {e}")
                        
                        # 保存原始CAM数据
                        try:
                            np.save(cam_data_path, cam_pred)
                        except Exception as e:
                            print(f"Failed to save NPY for {image_path}: {e}")
                        
                        # 添加到结果列表
                        all_cams.append(cam_pred)
                        all_image_paths.append(image_path)
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
        
        print(f"CAM generation completed. {len(all_cams)} CAMs generated.")
        return all_cams, all_image_paths
    
    def load_model(self, model_path=None):
        """
        加载预训练模型。
        
        参数:
            model_path: 模型检查点的路径
            
        异常:
            FileNotFoundError: 如果找不到模型文件
        """
        if model_path is None:
            model_path = self.model_path
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        print(f"Loaded model from {model_path}")

def get_enhanced_dataloaders(
    image_dir, batch_size, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, fast_test=False
):
    """
    创建数据加载器，包含增强的数据预处理
    """
    from utils import get_dataloaders
    
    # 增强的数据预处理
    from utils import CustomDataset
    
    # 获取原始数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(
        image_dir=image_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        fast_test=fast_test,
        # 增加其他参数传递
    )
    
    return train_loader, val_loader, test_loader

def train_classifier_and_extract_cams(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, fast_test=None, skip_train_eval=False, skip_cam_gen=False):
    """
    训练分类模型并提取CAM
    
    参数:
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        fast_test: 快速测试配置
        skip_train_eval: 是否跳过训练集评估
        skip_cam_gen: 是否跳过CAM生成
        
    返回:
        CAMExtractor: 训练好的CAM提取器实例
    """
    from config import CLASSIFIER_CONFIG, FAST_TEST_CONFIG
    
    # 使用配置创建CAM提取器
    config = CLASSIFIER_CONFIG.copy()
    
    # 检查是否为快速测试模式
    if fast_test is None and FAST_TEST_CONFIG.get("enabled", False):
        fast_test = FAST_TEST_CONFIG
        print("Enabling fast test mode")
        config["epochs"] = fast_test.get("epochs", 3)
        config["batch_size"] = fast_test.get("batch_size", 16)
        
    # 创建CAM提取器
    cam_extractor = CAMExtractor(config)
    
    # 优化数据加载器的创建 - 将配置传递给数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(
        image_dir=IMAGE_DIR,
        batch_size=config["batch_size"],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        fast_test=fast_test,
        config=config  # 传递完整配置
    )
    
    # 检查是否需要训练或加载模型
    model_path = Path(CLASSIFIER_DIR) / 'cam_model.pth'
    if model_path.exists():
        print(f"Loading existing model: {model_path}")
        cam_extractor.load_model(model_path)
    else:
        print("Training new model")
        cam_extractor.train(train_loader, val_loader)
    
    # 评估测试集
    print("Evaluating test set performance...")
    test_acc = cam_extractor.evaluate(test_loader)
    print(f"Test set accuracy: {test_acc:.2f}%")
    
    if skip_train_eval:
        print("Skipping train set evaluation to save time...")
        train_acc = 0.0  # 占位值
        val_acc = 0.0    # 占位值
    else:
        try:
            # 对训练集进行评估
            print("Evaluating train and validation sets...")
            train_acc = cam_extractor.evaluate(train_loader)
            val_acc = cam_extractor.evaluate(val_loader)
        except Exception as e:
            print(f"Error during train set evaluation: {e}")
            print("Using test set accuracy as fallback...")
            train_acc = test_acc
            val_acc = test_acc
    
    # 是否跳过CAM生成
    if skip_cam_gen:
        print("Skipping CAM generation step...")
        return cam_extractor
    
    # 生成CAM
    print("Starting CAM generation...")
    try:
        # 只为训练集生成CAM
        print("Processing training dataset only...")
        cam_extractor.generate_cams(train_loader)
        print(f"CAMs saved to {CAM_DIR}")
    except Exception as e:
        print(f"Error during CAM generation: {e}")
    
    return cam_extractor

def visualize_cam(image, cam, save_path=None, alpha=0.5):
    """
    CAM可视化函数，支持边缘增强和多种热图模式
    
    参数:
        image: 输入图像张量或numpy数组
        cam: 类激活图numpy数组
        save_path: 保存可视化结果的路径
        alpha: 叠加的透明度因子
    """
    if isinstance(image, torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)
    
    image_uint8 = (image * 255).astype(np.uint8)
    
    if len(image_uint8.shape) == 2:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    
    h, w = image_uint8.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # 应用阈值处理，低于阈值的像素设为0
    cam_resized[cam_resized < CAM_CONFIG["threshold"]] = 0
    
    # 应用Canny边缘检测增强边缘
    if CAM_CONFIG["apply_canny"]:
        try:
            image_gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY) if image_uint8.shape[2] == 3 else image_uint8
            edges = cv2.Canny(image_gray, CAM_CONFIG["canny_low"], CAM_CONFIG["canny_high"])
            # 在边缘处增强CAM
            cam_resized = cam_resized.copy()  # 创建副本以避免修改原始数据
            cam_resized[edges > 0] *= 1.5  # 边缘位置增强CAM
        except Exception as e:
            print(f"Edge enhancement failed: {e}")
    
    # 使用颜色映射增强热图可视化效果
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    colormap = CAM_CONFIG.get("colormap", cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    
    # 确保形状匹配
    assert image_uint8.shape[:2] == heatmap.shape[:2], f"Shape mismatch: image={image_uint8.shape}, heatmap={heatmap.shape}"
    
    if image_uint8.shape[2] == 3:
        image_uint8_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    else:
        image_uint8_bgr = image_uint8
    
    # 应用对比度增强
    alpha_actual = CAM_CONFIG.get("overlay_alpha", alpha)
    overlay = cv2.addWeighted(image_uint8_bgr, 1-alpha_actual, heatmap, alpha_actual, 0)
    
    # 生成并排显示的结果
    result = np.zeros((h, w*2, 3), dtype=np.uint8)
    result[:, :w] = image_uint8_bgr
    result[:, w:] = overlay
    
    # 添加标题
    cv2.putText(result, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'CAM Overlay', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    try:
        if save_path:
            cv2.imwrite(str(save_path), result)
        else:
            cv2.imshow('CAM Visualization', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error saving visualization result: {e}")
        # 尝试仅保存热力图
        try:
            cv2.imwrite(str(save_path), cam_uint8)
        except:
            print(f"Saving simplified CAM also failed")

def _expand_head_to_body(cam_pred):
    """扩展头部激活到整个身体区域"""
    # 创建二值化版本找到关键区域
    binary = (cam_pred > CAM_CONFIG["threshold"]).astype(np.uint8)
    
    # 形态学膨胀操作扩展激活区域
    kernel_size = CAM_CONFIG.get("morphology_kernel_size", 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilate_iterations = CAM_CONFIG.get("dilate_iterations", 2)
    
    # 形态学膨胀，向下扩展
    dilated = cv2.dilate(binary, kernel, iterations=dilate_iterations)
    
    # 寻找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    # 找到最大的区域（可能是头部）
    if num_labels > 1:  # 确保有至少一个区域
        # 忽略背景（索引0）
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_mask = (labels == largest_label).astype(np.uint8)
        
        # 获取区域的边界框
        x, y, w, h, area = stats[largest_label]
        
        # 考虑扩展的比例因子
        expansion_factor = CAM_CONFIG.get("body_expansion_factor", 1.3)
        expanded_h = int(h * expansion_factor)
        
        # 创建向下扩展的区域
        expanded_mask = np.zeros_like(largest_mask)
        expanded_mask[y:min(y+expanded_h, binary.shape[0]), x:x+w] = 1
        
        # 合并原始CAM与扩展区域
        expanded_binary = np.maximum(binary, expanded_mask)
        
        # 重新应用到原始CAM
        # 保留原始值，但扩展区域应用较小的值
        result = cam_pred.copy()
        # 在扩展区域中，将0值替换为原始值的0.5倍（只在扩展区域，非原始激活区域）
        extension_mask = (expanded_mask > 0) & (binary == 0)
        
        if extension_mask.any():
            # 计算原始区域的平均值作为扩展区域的值
            orig_mean = cam_pred[binary > 0].mean() if (binary > 0).any() else 0.3
            result[extension_mask] = max(0.3, orig_mean * 0.7)  # 设置为原始区域的70%
        
        return result
    
    return cam_pred

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train_eval", action="store_true", help="Skip training set evaluation to avoid memory issues")
    parser.add_argument("--skip_cam_gen", action="store_true", help="Skip CAM generation step")
    parser.add_argument("--fast", action="store_true", help="Enable fast test mode with reduced data and training epochs")
    args = parser.parse_args()
    
    train_classifier_and_extract_cams(
        train_ratio=DATASET_CONFIG['train_ratio'],
        val_ratio=DATASET_CONFIG['val_ratio'],
        test_ratio=DATASET_CONFIG['test_ratio'],
        seed=DATASET_CONFIG['random_seed'],
        skip_train_eval=args.skip_train_eval,
        skip_cam_gen=args.skip_cam_gen,
        fast_test=args.fast
    ) 