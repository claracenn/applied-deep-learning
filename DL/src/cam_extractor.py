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
from PIL import Image, ImageFilter
from skimage.filters import threshold_otsu
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import random_split

from config import CLASSIFIER_CONFIG, CAM_DIR, CLASSIFIER_DIR, IMAGE_DIR, CAM_CONFIG
from utils import get_dataloaders, get_dataloaders_from_split, visualize_cam, save_cam, set_seed

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
        self.model_path = Path(CLASSIFIER_DIR) / "classifier_model.pth"
        
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
        
        self.scaler = torch.amp.GradScaler()
        
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
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):

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
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=self.config.get('use_mixed_precision', True)):

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
        
        # 准备保存任务队列
        from queue import Queue
        save_queue = Queue()
        
        # 启用CUDA优化
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = self.config.get('use_mixed_precision', True) and torch.cuda.is_available()
        
        # 预先创建GPU上的形态学核
        morph_kernel = torch.ones(3, 3, device=self.device)
        
        # 创建专用线程池处理保存操作
        def save_worker():
            while True:
                item = save_queue.get()
                if item is None:  # 结束信号
                    break
                    
                cam_pred, image, image_path, save_dir = item
                try:
                    image_name = Path(image_path).stem
                    cam_image_path = save_dir / f"{image_name}_cam.png"
                    cam_data_path = save_dir / f"{image_name}_cam.npy"
                    
                    # 保存PNG图像
                    visualize_cam(image.cpu(), cam_pred, save_path=cam_image_path)
                    
                    # 保存原始CAM数据
                    np.save(cam_data_path, cam_pred)
                except Exception as e:
                    print(f"Failed to save CAM for {image_path}: {e}")
                finally:
                    save_queue.task_done()
        
        # 启动保存工作线程
        import threading
        num_save_workers = min(os.cpu_count(), 4)  # 限制保存线程数
        save_threads = []
        for _ in range(num_save_workers):
            t = threading.Thread(target=save_worker, daemon=True)
            t.start()
            save_threads.append(t)
            
        # 工具函数：GPU上的形态学操作
        def gpu_morphology(tensor, kernel):
            # 膨胀操作
            dilated = torch.nn.functional.conv2d(
                tensor.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            ).squeeze() > 0
            
            # 腐蚀操作
            eroded = 1.0 - torch.nn.functional.conv2d(
                (1.0 - dilated.float()).unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            ).squeeze() > 0
            
            return eroded
            
        # 增加批量大小以提高吞吐量
        original_batch_size = data_loader.batch_size
        
        # 处理每个批次的CAM
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
            for batch_idx, batch in enumerate(data_loader):
                # 定期打印进度
                if batch_idx % 5 == 0 or batch_idx == len(data_loader) - 1:
                    print(f"  Processing batch {batch_idx+1}/{len(data_loader)} ({(batch_idx+1)/len(data_loader)*100:.1f}%)")
                
                # 高效数据加载
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                image_paths = batch['image_path']
                
                # 在单次前向传播中生成所有CAM
                outputs, cams, probas = self.model(images, return_cam=True)
                _, predicted = outputs.max(1)
                
                # 批量处理CAM (尽可能保持GPU上的计算)
                for i, (image, cam, pred, image_path) in enumerate(zip(images, cams, predicted, image_paths)):
                    try:
                        # 保持CAM在GPU上进行计算
                        cam_gpu = cam[pred]  # 选择预测类别的CAM
                        
                        # 阈值处理
                        cam_gpu = torch.nn.functional.threshold(cam_gpu, CAM_CONFIG["threshold"], 0)
                        
                        # 转为二值图
                        cam_binary = (cam_gpu > 0).float()
                        
                        # 使用GPU上的形态学操作 (闭运算：先膨胀后腐蚀)
                        if CAM_CONFIG.get("use_morphology", True):
                            cam_closed = gpu_morphology(cam_binary, morph_kernel)
                            cam_gpu = cam_gpu * cam_closed.float()
                        
                        # 调整CAM大小到原图尺寸 (仍在GPU上)
                        img_size = self.config['image_size']
                        cam_resized = torch.nn.functional.interpolate(
                            cam_gpu.unsqueeze(0).unsqueeze(0),
                            size=img_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                        
                        # 归一化 (仍在GPU上)
                        cam_min = cam_resized.min()
                        cam_max = cam_resized.max()
                        if cam_max > cam_min:
                            cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)
                        else:
                            cam_resized = torch.zeros_like(cam_resized)
                        
                        # 应用配置的阈值处理
                        if CAM_CONFIG.get("use_adaptive_threshold", False):
                            # 计算 Otsu 阈值 (需要转到CPU)
                            cam_uint8 = (cam_resized.cpu().numpy() * 255).astype(np.uint8)
                            otsu_threshold = threshold_otsu(cam_uint8)
                            threshold = 0.7 * (otsu_threshold / 255.0) + 0.3 * CAM_CONFIG["threshold"]
                            
                            # 应用阈值 (返回到GPU)
                            threshold_tensor = torch.tensor(threshold, device=self.device)
                            cam_resized = torch.where(cam_resized < threshold_tensor, 
                                                     torch.zeros_like(cam_resized), 
                                                     cam_resized)
                        else:
                            # 使用固定阈值
                            cam_resized = torch.where(cam_resized < CAM_CONFIG["threshold"], 
                                                     torch.zeros_like(cam_resized), 
                                                     cam_resized)
                        
                        # 应用gamma校正 (仍在GPU上)
                        cam_resized = torch.pow(cam_resized, CAM_CONFIG["gamma"])
                        
                        # 现在需要转到CPU进行身体区域扩展
                        cam_cpu = cam_resized.cpu().numpy()
                        
                        # 应用身体区域扩展
                        if CAM_CONFIG.get("use_morphology", True):
                            cam_cpu = _expand_head_to_body(cam_cpu)
                        
                        # 添加到结果列表
                        all_cams.append(cam_cpu)
                        all_image_paths.append(image_path)
                        
                        # 将保存任务添加到队列
                        save_queue.put((cam_cpu, image, image_path, save_dir))
                        
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                
                # 每隔几个批次清理一次GPU内存
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        # 等待所有保存任务完成
        save_queue.join()
        
        # 向保存线程发送结束信号
        for _ in range(num_save_workers):
            save_queue.put(None)
        
        # 等待所有线程完成
        for t in save_threads:
            t.join()
        
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
    image_dir, batch_size, test_ratio=0.2, seed=42, config=None
):
    """
    创建数据加载器，包含增强的数据预处理，使用官方数据集划分
    
    参数:
        image_dir: 图像目录路径
        batch_size: 批量大小
        test_ratio: 从test.txt中选取的比例作为测试集
        seed: 随机种子，用于控制测试集选择的可重现性
        config: 配置字典，包含优化参数
        
    返回:
        tuple: (train_loader, test_loader) - 只包含训练集和测试集
    """
    # 确保配置有效
    if config is None:
        config = CLASSIFIER_CONFIG.copy()
    
    print("Using official dataset split...")
    # 使用官方数据集划分
    train_loader, test_loader = get_dataloaders_from_split(
        image_dir=image_dir,
        batch_size=batch_size,
        test_ratio=test_ratio,
        seed=seed,
        config=config
    )
    
    return train_loader, test_loader

def train_classifier_and_extract_cams(seed=42, skip_train_eval=False, skip_cam_gen=False):
    """
    训练分类模型并提取CAM
    
    参数:
        seed: 随机种子
        skip_train_eval: 是否跳过训练集评估
        skip_cam_gen: 是否跳过CAM生成
        
    返回:
        CAMExtractor: 训练好的CAM提取器实例
    """
    from config import CLASSIFIER_CONFIG, IMAGE_DIR
    
    # 使用配置创建CAM提取器
    config = CLASSIFIER_CONFIG.copy()
    
    # 创建CAM提取器
    cam_extractor = CAMExtractor(config) 

    # 创建数据加载器，使用官方数据集划分
    print("Using official dataset split...")
    # test_ratio=0.2 表示从test.txt中选择20%的数据作为测试集
    train_loader, test_loader = get_dataloaders_from_split(
        image_dir=IMAGE_DIR,
        batch_size=config["batch_size"],
        test_ratio=0.2,  # 从test.txt中选择20%作为测试集
        seed=seed,
        config=config
    )
    
    # 检查是否需要训练或加载模型
    model_path = Path(CLASSIFIER_DIR) / 'classifier_model.pth'
    if model_path.exists():
        print(f"Loading existing model: {model_path}")
        cam_extractor.load_model(model_path)
    else:
        print("Training new model")
        # 从训练集中分出一小部分作为验证集来监控训练
        # 注意：这里的划分是在trainval.txt的数据上进行的，与test.txt的20%测试集无关
        train_dataset = train_loader.dataset
        val_ratio = 0.1  # 使用10%的训练数据作为验证集
        train_size = int((1 - val_ratio) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        print(f"Splitting training data: {train_size} for training, {val_size} for validation")
        train_subset, val_subset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # 创建新的数据加载器
        train_subset_loader = DataLoader(
            train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            drop_last=True
        )
        
        val_subset_loader = DataLoader(
            val_subset,
            batch_size=config["batch_size"]*2,
            shuffle=False,
            num_workers=train_loader.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        
        # 使用分割后的训练集和验证集进行训练
        cam_extractor.train(train_subset_loader, val_subset_loader)
    
    # 评估测试集（这是从test.txt中选出的20%数据）
    print("Evaluating test set performance...")
    test_acc = cam_extractor.evaluate(test_loader)
    print(f"Test set accuracy: {test_acc:.2f}%")
    
    if skip_train_eval:
        print("Skipping train set evaluation to save time...")
    else:
        try:
            # 对训练集进行评估（这是trainval.txt中的所有数据）
            print("Evaluating train set...")
            train_acc = cam_extractor.evaluate(train_loader)
            print(f"Train set accuracy: {train_acc:.2f}%")
        except Exception as e:
            print(f"Error during train set evaluation: {e}")
    
    # 是否跳过CAM生成
    if skip_cam_gen:
        print("Skipping CAM generation step...")
        return cam_extractor
    
    # 生成CAM
    print("Starting CAM generation...")
    try:
        # 只为训练集生成CAM（使用trainval.txt中的所有数据）
        print("Processing training dataset only...")
        cam_extractor.generate_cams(train_loader)
        print(f"CAMs saved to {CAM_DIR}")
    except Exception as e:
        print(f"Error during CAM generation: {e}")
    
    return cam_extractor

def visualize_cam_pil(image, cam, save_path=None, alpha=0.5, cam_config=None):
    """
    用 PIL 替代 OpenCV 实现的 CAM 可视化函数（不使用 matplotlib 或 ImageDraw）

    参数:
        image: 输入图像张量或numpy数组
        cam: 类激活图 numpy 数组 (H, W)，范围 [0,1]
        save_path: 可选保存路径
        alpha: 叠加透明度
        cam_config: 可选 CAM_CONFIG 字典
    """
    if cam_config is None:
        cam_config = {}

    if isinstance(image, torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)

    image_uint8 = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8).convert("RGB")
    w, h = pil_image.size

    # 处理 CAM，阈值过滤
    cam = np.clip(cam, 0, 1)
    cam[cam < cam_config.get("threshold", 0.05)] = 0
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h)).convert("L")

    # 构造简化的彩色热图（只使用 PIL，用红色增强高激活区域）
    cam_color = Image.new("RGB", (w, h))
    cam_pixels = cam_resized.load()
    color_pixels = cam_color.load()

    for i in range(w):
        for j in range(h):
            v = cam_pixels[i, j]
            color_pixels[i, j] = (v, 0, 0)  # 红色为主

    # 混合热图和原图
    overlay_alpha = cam_config.get("overlay_alpha", alpha)
    overlay = Image.blend(pil_image, cam_color, overlay_alpha)

    # 拼接图像：原图 | 叠加图
    result = Image.new("RGB", (w * 2, h))
    result.paste(pil_image, (0, 0))
    result.paste(overlay, (w, 0))

    # 保存或显示结果
    try:
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(save_path)
        else:
            result.show()
    except Exception as e:
        print(f"Error saving CAM visualization: {e}")

def _expand_head_to_body(cam_pred):
    """优化的CAM扩展函数，使用numpy高效矢量化操作"""
    # 创建二值掩码
    binary = (cam_pred > CAM_CONFIG["threshold"]).astype(np.uint8)
    if not np.any(binary):
        return cam_pred  # 如果没有检测到区域，直接返回
    
    # 快速膨胀操作
    kernel_size = CAM_CONFIG.get("morphology_kernel_size", 5)
    iterations = CAM_CONFIG.get("dilate_iterations", 2)
    
    # 替代custom dilation的更高效实现
    from scipy import ndimage
    dilated = binary.copy()
    for _ in range(iterations):
        dilated = ndimage.maximum_filter(dilated, size=kernel_size)
    
    # 寻找连通区域的快速方法
    labels, num_features = ndimage.label(dilated)
    if num_features == 0:
        return cam_pred
    
    # 计算所有区域的属性
    objects = ndimage.find_objects(labels)
    region_sizes = np.array([(x.stop-x.start) * (y.stop-y.start) for x, y in objects])
    
    if len(region_sizes) == 0:
        return cam_pred
    
    # 找出最大区域
    largest_idx = np.argmax(region_sizes)
    minr, minc = objects[largest_idx][0].start, objects[largest_idx][1].start
    maxr, maxc = objects[largest_idx][0].stop, objects[largest_idx][1].stop
    
    h = maxr - minr
    
    # 计算扩展后的高度
    expansion_factor = CAM_CONFIG.get("body_expansion_factor", 1.3)
    expanded_h = int(h * expansion_factor)
    end_r = min(minr + expanded_h, binary.shape[0])
    
    # 创建扩展掩码
    expanded_mask = np.zeros_like(binary)
    expanded_mask[minr:end_r, minc:maxc] = 1
    
    # 组合原始二值图和扩展掩码
    expanded_binary = np.maximum(binary, expanded_mask)
    
    # 创建结果
    result = cam_pred.copy()
    extension_mask = (expanded_mask > 0) & (binary == 0)
    
    if np.any(extension_mask):
        orig_mean = np.mean(cam_pred[binary > 0]) if np.any(binary > 0) else 0.3
        result[extension_mask] = max(0.3, orig_mean * 0.7)
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train_eval", action="store_true", help="Skip training set evaluation to avoid memory issues")
    parser.add_argument("--skip_cam_gen", action="store_true", help="Skip CAM generation step")
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    train_classifier_and_extract_cams(
        seed=42,
        skip_train_eval=args.skip_train_eval,
        skip_cam_gen=args.skip_cam_gen
    ) 