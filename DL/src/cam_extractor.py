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
from skimage.morphology import skeletonize, binary_dilation, binary_closing
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import random_split
from scipy import ndimage

from config import CLASSIFIER_CONFIG, CAM_DIR, CLASSIFIER_DIR, IMAGE_DIR, CAM_CONFIG
from utils import get_dataloaders, get_dataloaders_from_split, visualize_cam, save_cam, set_seed

class CAMModel(nn.Module):
    """
    轻量级CAM模型，使用ResNet作为骨干网络，支持混合精度训练
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(CAMModel, self).__init__()
        # 使用配置选择backbone
        backbone = CLASSIFIER_CONFIG.get("backbone", "resnet18")
        if backbone == "resnet18":
            base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 512  # ResNet18最后一层特征维度
        elif backbone == "resnet50":
            base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 2048  # ResNet50最后一层特征维度
        else:
            raise ValueError(f"不支持的backbone: {backbone}, 请使用'resnet18'或'resnet50'")
        
        print(f"使用 {backbone} 作为特征提取器，特征维度: {self.feature_dim}")
        
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
        
        # 像素到原型对比学习的特征投影层
        self.use_prototype_contrast = CAM_CONFIG.get("prototype_contrast", True)
        if self.use_prototype_contrast:
            self.projector = nn.Conv2d(self.feature_dim, 128, kernel_size=1)
            # 保存当前批次的原型
            self.prototypes = None
            
        # CAM生成和梯度跟踪相关属性
        self.cams = []
        self.target_class = []
        self.target_sizes = []
        self.gradients = {}
    
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
            # 为每个样本计算CAM
            with torch.enable_grad():  # 确保能够计算梯度
                # 直接调用_compute_gradcampp计算CAM
                cams = self._compute_gradcampp(features.detach(), logits.detach(), edge_weights.detach())
                
                # 创建CAM张量 (B, num_classes, H, W)
                batch_size = features.size(0)
                num_classes = logits.size(1)
                h, w = features.size(2), features.size(3)
                
                # 创建零张量并填充CAM
                cam_tensor = torch.zeros(batch_size, num_classes, h, w, device=features.device)
                probas = F.softmax(logits, dim=1)
                
                # 对每个样本，将CAM放入对应的类别通道
                for b in range(batch_size):
                    if b < len(self.target_class) and b < len(cams):
                        target_class = self.target_class[b]
                        cam = torch.tensor(cams[b], device=features.device)
                        if target_class < num_classes:
                            cam_tensor[b, target_class] = cam
                
                # 如果启用了像素到原型对比学习并且处于训练模式
                if self.use_prototype_contrast and self.training:
                    # 投影特征用于对比学习
                    projected_features = self.projector(features)
                    # L2标准化
                    projected_features = F.normalize(projected_features, p=2, dim=1)
                    return logits, cam_tensor, probas, projected_features
                
                return logits, cam_tensor, probas
        
        return logits
    
    def _compute_gradcampp(self, feature, logits, edge_weights=None):
        """计算Grad-CAM++，使用改进算法专注于目标轮廓，有更好的边缘感知能力"""
        B, C, H, W = feature.shape  # 获取特征图尺寸
        
        # 清空之前的CAM和目标类别列表
        self.cams = []
        self.target_class = []
        self.target_sizes = []
        
        # 确定梯度类别
        probs = F.softmax(logits, dim=1)
        class_idx = []
        for b in range(B):
            idx = torch.argmax(probs[b]).item()
            class_idx.append((0, idx))  # (layer_idx, class_idx)
            
        offset = len(class_idx)
        
        # 增加背景惩罚与前景偏好
        foreground_bias = CAM_CONFIG.get("foreground_bias", 3.0)
        
        # 边缘增强参数
        edge_strength = CAM_CONFIG.get("edge_strength", 3.0)  
        edge_sensitivity = CAM_CONFIG.get("edge_sensitivity", 1.8)
        gaussian_sigma = CAM_CONFIG.get("gaussian_sigma", 0.35)
        cam_gamma = CAM_CONFIG.get("cam_gamma", 2.0)
        activation_threshold = CAM_CONFIG.get("activation_threshold", 0.2)
        
        # 计算特征重要性，加强前景类别权重
        weights = [1.0] * len(class_idx)
        weights = [w * foreground_bias if i > 0 else w for i, w in enumerate(weights)]
        
        # 预先检查特征图大小，如果太小则使用更小的高斯核
        if H < 10 or W < 10:
            # 删除警告打印
            kernel_size = min(5, H, W)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size - 1
            if kernel_size < 3:
                # 删除警告打印
                gaussian_sigma = 0  # 禁用模糊
        else:
            kernel_size = min(15, H, W)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size - 1
        
        # 获得所有样本的CAMs，防止内存溢出，逐一处理
        for b, (idx, w) in enumerate(zip(class_idx, weights)):
            try:
                if isinstance(idx, tuple):
                    target_layer = idx[0]
                    target_class = idx[1]
                else:
                    target_layer = 0
                    target_class = idx
                
                # 准备梯度计算 - 只处理当前样本的特征
                feature_single = feature[b:b+1].clone().detach()
                feature_single.requires_grad_(True)
                
                # 前向传播，只对当前特征
                # 全局池化用于分类
                pooled = self.avgpool(feature_single)
                pooled = torch.flatten(pooled, 1)
                logits_single = self.fc(pooled)
                
                # 反向传播，计算梯度
                grad = None
                if target_class < logits_single.size(1):
                    # 清除之前的梯度
                    self.zero_grad(set_to_none=True)
                    
                    # 只关注目标类别的得分
                    score = logits_single[0, target_class]
                    
                    # 获取梯度
                    score.backward(retain_graph=True)
                    grad = feature_single.grad
                
                # 如果梯度计算失败，创建默认梯度
                if grad is None:
                    # 删除警告打印
                    grad = torch.ones_like(feature_single)
                
                # 清除梯度计算的临时状态
                feature_single.requires_grad_(False)
                feature_single.grad = None
                
                # 计算CAM
                with torch.no_grad():  # 以下操作不需要跟踪梯度
                    # 特征空间标准化 - 增强边缘感知
                    feat_b = feature[b]
                    feat_norm = feat_b / (torch.norm(feat_b, dim=0, keepdim=True) + 1e-7)
                    
                    # 使用改进的边缘增强方法
                    # 水平和垂直方向的梯度
                    feat_dx = torch.abs(F.pad(feat_b[:, :, 1:] - feat_b[:, :, :-1], (0, 1, 0, 0)))
                    feat_dy = torch.abs(F.pad(feat_b[:, 1:, :] - feat_b[:, :-1, :], (0, 0, 0, 1)))
                    
                    # 调整边缘检测敏感度
                    feat_dx = torch.pow(feat_dx, edge_sensitivity)
                    feat_dy = torch.pow(feat_dy, edge_sensitivity)
                    
                    # 梯度幅度 - 边缘强度
                    edge_weights_local = torch.sqrt(feat_dx**2 + feat_dy**2)
                    # 归一化边缘权重
                    if edge_weights_local.max() > 0:
                        edge_weights_local = edge_weights_local / edge_weights_local.max()
                    
                    # 创建综合边缘加权
                    edge_enhanced = feat_norm * (1.0 + edge_strength * edge_weights_local)
                    
                    # 计算权重系数 - Grad-CAM++算法的核心
                    grad_2 = grad.pow(2)
                    grad_3 = grad_2 * grad
                    
                    # 创建alpha系数
                    alpha_num = grad_2[0]
                    alpha_denom = 2 * grad_2[0] + (feat_b * grad_3[0]).sum(dim=(1, 2), keepdim=True)
                    alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
                    alpha = alpha_num / (alpha_denom + 1e-7)
                    
                    # 计算权重
                    weights_cam = (alpha * torch.relu(grad[0])).sum(dim=(1, 2), keepdim=True)
                    
                    # 生成初始CAM
                    cam = (weights_cam * edge_enhanced).sum(dim=0)  # 使用边缘增强的特征
                    
                    # 应用ReLU
                    cam = torch.relu(cam)
                    
                    # 伽马校正增强对比度
                    cam = cam ** cam_gamma
                    
                    # 低激活抑制
                    if activation_threshold > 0:
                        cam_max = torch.max(cam)
                        if cam_max > 0:
                            cam = torch.where(cam < activation_threshold * cam_max, 
                                            torch.zeros_like(cam), cam)
                    
                    # 归一化
                    if cam.max() > 0:
                        cam = cam / cam.max()
                    
                    # 使用更精确的高斯平滑
                    if gaussian_sigma > 0:
                        try:
                            # 使用内置的高斯模糊实现
                            cam_tensor = cam.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                            cam = gaussian_blur2d(
                                cam_tensor,
                                kernel_size=(kernel_size, kernel_size),
                                sigma=(gaussian_sigma, gaussian_sigma)
                            ).squeeze()
                        except Exception as e:
                            # 删除错误打印
                            cam = cam  # 使用原始CAM
                        
                        # 再次应用伽马校正进行锐化
                        cam = cam ** 1.5
                        
                        # 再次归一化
                        if cam.max() > 0:
                            cam = cam / cam.max()
                    
                    # 保存CAM
                    cam_np = cam.cpu().numpy()
                    self.cams.append(cam_np)
                    self.target_sizes.append(None)
                    self.target_class.append(target_class)
            except Exception as e:
                # 删除错误打印
                # 创建一个空CAM作为替代
                empty_cam = np.zeros((H, W))
                self.cams.append(empty_cam)
                self.target_sizes.append(None)
                self.target_class.append(0)  # 默认为背景类
        
        # 强制清理所有临时变量和梯度
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return self.cams
    
    def estimate_class_prototypes(self, features, cams):
        """
        从特征图和CAM中估计类别原型
        
        参数:
            features: 投影后的特征图 [B, C, H, W]
            cams: 类激活图 [B, num_classes, H, W]
            
        返回:
            prototypes: 类别原型 [num_classes, C]
        """
        batch_size, channels, height, width = features.size()
        num_classes = cams.size(1)
        prototypes = []
        
        # 将特征图调整为 [B, C, H*W]
        features_flat = features.view(batch_size, channels, -1).clone()  # 使用clone避免原位操作
        
        # 为类别选择处理创建一个分离的计算上下文
        for c in range(num_classes):
            # 在不需要梯度的操作中使用torch.no_grad()
            with torch.no_grad():
                # 获取当前类别的CAM
                class_cams = cams[:, c, :, :].view(batch_size, -1)  # [B, H*W]
                
                # 收集所有批次中的高激活像素
                all_top_features = []
                all_top_weights = []
                
                for b in range(batch_size):
                    # 获取当前批次、当前类别的CAM
                    current_cam = class_cams[b]  # [H*W]
                    
                    # 只选择有效激活的像素
                    valid_pixels = current_cam > CAM_CONFIG.get("contrast_threshold", 0.2)
                    
                    if torch.sum(valid_pixels) > 0:
                        # 选择top-k个高激活的像素
                        topk = min(CAM_CONFIG.get("prototype_topk", 100), torch.sum(valid_pixels).item())
                        if topk > 0:
                            values, indices = torch.topk(current_cam, k=topk)
                            top_features = features_flat[b, :, indices]  # [C, topk]
                            
                            all_top_features.append(top_features)
                            all_top_weights.append(values)
            
            # 如果有足够的高激活像素
            if len(all_top_features) > 0:
                all_features = torch.cat(all_top_features, dim=1)  # [C, sum(topk)]
                all_weights = torch.cat(all_top_weights, dim=0)  # [sum(topk)]
                
                # 提前计算所需的值
                weighted_features = all_features * all_weights.unsqueeze(0)
                sum_weights = torch.sum(all_weights)
                
                # 计算加权平均
                if sum_weights > 0:
                    prototype = torch.sum(weighted_features, dim=1) / sum_weights
                else:
                    prototype = torch.mean(all_features, dim=1)
            else:
                # 如果没有足够的高激活像素，创建一个随机原型
                prototype = torch.randn(channels, device=features.device)
            
            # 归一化原型
            prototype = F.normalize(prototype, p=2, dim=0)
            prototypes.append(prototype)
        
        # 将所有原型堆叠成一个张量 [num_classes, C]
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes
    
    def compute_prototype_contrast_loss(self, features, cams):
        """
        计算像素到原型的对比损失
        
        参数:
            features: 投影后的特征图 [B, C, H, W]
            cams: 类激活图 [B, num_classes, H, W]
            
        返回:
            loss: 对比损失
        """
        # 估计类别原型
        prototypes = self.estimate_class_prototypes(features, cams)
        
        # 获取维度信息
        batch_size, channels, height, width = features.size()
        num_classes = cams.size(1)
        
        # 获取伪标签 (使用CAM的argmax)
        # [B, H, W]
        with torch.no_grad():
            pseudo_labels = torch.argmax(cams, dim=1)
        
        # 临时保存原型
        self.prototypes = prototypes
        
        # 重塑特征便于计算 - 使用clone()避免修改原始张量
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, channels).clone()  # [B*H*W, C]
        labels_flat = pseudo_labels.reshape(-1)  # [B*H*W]
        
        # 计算相似度矩阵 - 每个像素与每个原型
        # [B*H*W, num_classes]
        similarity = torch.mm(features_flat, prototypes.t()) / CAM_CONFIG.get("temperature", 0.1)
        
        # 只考虑有效像素 (CAM有足够激活的像素)
        # 为每个位置找到最大激活值
        with torch.no_grad():
            max_cams = torch.max(cams, dim=1)[0].reshape(-1)  # [B*H*W]
            valid_mask = max_cams > CAM_CONFIG.get("contrast_threshold", 0.2)
        
        if torch.sum(valid_mask) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 过滤有效像素
        valid_sim = similarity[valid_mask]  # [num_valid, num_classes]
        valid_labels = labels_flat[valid_mask]  # [num_valid]
        
        # 计算InfoNCE损失
        # 构建one-hot标签
        one_hot = torch.zeros_like(valid_sim)
        one_hot.scatter_(1, valid_labels.unsqueeze(1), 1)
        
        # InfoNCE公式: -log(exp(pos_sim) / sum(exp(sim)))
        exp_sim = torch.exp(valid_sim)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)  # 显式计算总和
        log_prob = torch.log(exp_sim / (sum_exp_sim + 1e-6))
        
        # 负对数似然损失
        loss = -torch.sum(one_hot * log_prob) / one_hot.size(0)
        
        return loss

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
        
        # 简化设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.model_path = Path(CLASSIFIER_DIR) / "classifier_model.pth"
        
        # 创建CAM模型
        self.model = CAMModel(num_classes=self.config['num_classes'], 
                               pretrained=self.config.get('use_pretrained', True))
        
        # 转换模型到目标设备
        self.model.to(self.device)
        
        # 尝试使用Lion优化器，如果不可用则使用AdamW
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
        
        # 像素到原型对比学习设置
        self.use_prototype_contrast = CAM_CONFIG.get("prototype_contrast", True)
        self.prototype_weight = CAM_CONFIG.get("prototype_weight", 0.1)
    
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
        print(f"Using pixel-to-prototype contrast: {self.use_prototype_contrast}")
        
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
                    if self.use_prototype_contrast and self.model.use_prototype_contrast:
                        # 使用返回CAM的模式，同时获取投影特征用于对比学习
                        outputs, cams, probas, projected_features = self.model(images, return_cam=True)
                        
                        # 主分类损失
                        cls_loss = self.criterion(outputs, labels)
                        
                        # 计算像素到原型对比损失
                        contrast_loss = self.model.compute_prototype_contrast_loss(
                            projected_features, cams)
                        
                        # 总损失 = 分类损失 + 对比损失
                        loss = cls_loss + self.prototype_weight * contrast_loss
                        
                        if batch_idx % 20 == 0:
                            print(f"  Batch {batch_idx}: cls_loss: {cls_loss.item():.4f}, contrast_loss: {contrast_loss.item():.4f}")
                    else:
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
        
        # 记录成功和失败的计数
        success_count = 0
        error_count = 0
        
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
                    
                    # 打印保存路径，帮助调试
                    print(f"CAM saved to {cam_image_path}")
                except Exception as e:
                    print(f"Error saving CAM for {image_path}: {str(e)}")
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
                
                try:
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
                                try:
                                    cam_cpu = _expand_head_to_body(cam_cpu)
                                except Exception as e:
                                    print(f"Error in _expand_head_to_body for {image_path}: {str(e)}")
                                    # 如果扩展失败，使用原始的CAM
                                    pass
                            
                            # 添加到结果列表
                            all_cams.append(cam_cpu)
                            all_image_paths.append(image_path)
                            
                            # 将保存任务添加到队列
                            save_queue.put((cam_cpu, image, image_path, save_dir))
                            success_count += 1
                            
                        except Exception as e:
                            print(f"Error processing image {image_path}: {str(e)}")
                            error_count += 1
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                
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
        
        print(f"CAM generation completed. {len(all_cams)} CAMs generated. Success: {success_count}, Errors: {error_count}")
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

    def __call__(self, input_tensor, class_idx=None, retain_graph=False):
        """
        使用GradCAM++算法计算输入图像的类激活图
        Args:
            input_tensor (torch.Tensor): 输入模型的张量，形状为 (B, C, H, W)
            class_idx (Union[int, List[int], None]): 需要计算CAM的类别索引
                - None: 使用预测类别
                - int: 使用指定类别
                - List[int]: 对每个输入使用对应的类别
            retain_graph (bool): 是否保留梯度图，多次反向传播时需要
        Returns:
            List[np.ndarray]: 每个输入的CAM
        """
        # 重置存储
        self.cams = []
        self.target_class = []
        self.target_sizes = []
        self.gradients = {}
        
        # 判断输入是否为批次
        batch_size = input_tensor.size(0)
        if class_idx is None:
            class_idx = [None] * batch_size
        if isinstance(class_idx, int):
            class_idx = [class_idx] * batch_size

        # 注册hook以捕获梯度
        for i, m in enumerate(self.target_layers):
            # 定义梯度钩子函数
            def hook_fn(module, input_grad, output_grad, layer_idx=i):
                self.gradients[layer_idx] = output_grad[0].detach()
            
            # 注册hook
            handle = m.register_full_backward_hook(lambda m, gi, go, li=i: hook_fn(m, gi, go, li))
            self.handlers.append(handle)

        # 前向传播
        self.model.zero_grad()
        if isinstance(self.model, nn.DataParallel):
            model_output = self.model.forward(input_tensor)
        else:
            model_output = self.model(input_tensor)
        
        # 如果是logits，转为概率
        if isinstance(model_output, dict) and 'logits' in model_output:
            logits = model_output['logits']
        else:
            logits = model_output
        
        probs = F.softmax(logits, dim=1)
        
        # 收集特征
        features = {}
        def feature_hook(module, input, output, layer_idx):
            features[layer_idx] = output.detach()
        
        for i, m in enumerate(self.target_layers):
            handle = m.register_forward_hook(lambda m, inp, outp, li=i: feature_hook(m, inp, outp, li))
            self.handlers.append(handle)

        # 再次前向传播以收集特征
        if isinstance(self.model, nn.DataParallel):
            self.model.forward(input_tensor)
        else:
            self.model(input_tensor)

        weights = []
        targets = []
        
        # 准备反向传播目标
        for b, idx in enumerate(class_idx):
            if idx is None:
                # 使用预测类别
                idx = probs[b].argmax().item()
            
            # 收集目标
            targets.append(idx)
            weights.append(1.0)
            
            # 为每个样本和目标类别设置梯度
            if b == 0:  # 第一个样本
                loss = -logits[b, idx]  # 使用负号以最大化类别分数
            else:
                loss = loss - logits[b, idx]
        
        # 反向传播
        loss.backward(retain_graph=retain_graph)
        
        # 计算CAM
        for i, layer in enumerate(self.target_layers):
            feature = features[i]
            offset = self._compute_gradcampp(feature, targets, weights)
        
        # 移除钩子
        for handle in self.handlers:
            handle.remove()
        self.handlers = []
        
        # 返回CAM
        return self.cams, self.target_class

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
    用 PIL 替代 OpenCV 实现的 CAM 可视化函数（不使用 matplotlib）

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
    cam_thresh = cam_config.get("threshold", 0.05)
    cam[cam < cam_thresh] = 0
    
    # 应用身体区域扩展
    if cam_config.get("use_morphology", True):
        cam = _expand_head_to_body(cam)
    
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h)).convert("L")

    # 构造彩色热图（只使用 PIL，用红色增强高激活区域）
    cam_color = Image.new("RGB", (w, h))
    cam_pixels = cam_resized.load()
    color_pixels = cam_color.load()

    # 创建热力图颜色映射
    colormap = cam_config.get("colormap", "viridis")  # 更改为更好的viridis颜色图
    
    # 改进的颜色映射，以便更好地区分不同的激活级别
    if colormap == "viridis":
        for i in range(w):
            for j in range(h):
                v = cam_pixels[i, j]
                if v > 0:
                    # 实现viridis颜色映射的简化版本
                    v_norm = v / 255.0  # 归一化到[0,1]
                    
                    if v_norm < 0.25:
                        # 深蓝到紫色
                        r = int(68 * (v_norm / 0.25))
                        g = int(1 * (v_norm / 0.25))
                        b = int(84 + (71 * (v_norm / 0.25)))
                    elif v_norm < 0.5:
                        # 紫色到蓝绿色
                        normv = (v_norm - 0.25) / 0.25
                        r = int(68 + (49 * normv))
                        g = int(1 + (102 * normv))
                        b = int(155 + (18 * normv))
                    elif v_norm < 0.75:
                        # 蓝绿色到黄绿色
                        normv = (v_norm - 0.5) / 0.25
                        r = int(117 + (111 * normv))
                        g = int(103 + (124 * normv))
                        b = int(173 - (76 * normv))
                    else:
                        # 黄绿色到黄色
                        normv = (v_norm - 0.75) / 0.25
                        r = int(228 + (25 * normv))
                        g = int(227 + (12 * normv))
                        b = int(97 - (44 * normv))
                    
                    color_pixels[i, j] = (r, g, b)
    elif colormap == "jet":
        for i in range(w):
            for j in range(h):
                v = cam_pixels[i, j]
                if v > 0:
                    # 简化的jet颜色映射
                    r = min(255, v)
                    g = min(255, max(0, 2*v - 255) if v > 128 else 2*v)
                    b = min(255, max(0, 255 - 2*v) if v < 128 else 0)
                    color_pixels[i, j] = (r, g, b)
    else:
        # 默认使用红色热力图
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
        
    return result

def _expand_head_to_body(cam_pred):
    """优化的CAM扩展函数，改进动物完整身体检测，考虑不同的动物姿态和方向"""
    # 创建二值掩码
    threshold = CAM_CONFIG.get("threshold", 0.3)
    binary = (cam_pred > threshold).astype(np.uint8)
    if not np.any(binary):
        return cam_pred  # 如果没有检测到区域，直接返回
    
    # 快速计算边缘轮廓 - 使用梯度进行边缘检测
    sobel_x = ndimage.sobel(cam_pred, axis=1)
    sobel_y = ndimage.sobel(cam_pred, axis=0)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_sensitivity = CAM_CONFIG.get("edge_sensitivity", 1.8)
    edge_magnitude = np.power(edge_magnitude, edge_sensitivity)  # 应用边缘敏感度
    edge_mask = (edge_magnitude > 0.1).astype(np.uint8)
    
    # 形态学操作参数
    kernel_size = CAM_CONFIG.get("morphology_kernel_size", 5)  # 使用配置的核心尺寸
    iterations = CAM_CONFIG.get("dilate_iterations", 2)  # 使用配置的迭代次数
    
    # 使用更精细的形态学操作，保留细节
    dilated = binary.copy()
    for _ in range(iterations):
        dilated = ndimage.maximum_filter(dilated, size=kernel_size)
    
    # 寻找连通区域
    labels, num_features = ndimage.label(dilated)
    if num_features == 0:
        return cam_pred
    
    # 计算所有区域的属性
    objects = ndimage.find_objects(labels)
    if not objects:  # 防止空列表导致的错误
        return cam_pred
        
    region_sizes = np.array([(x.stop-x.start) * (y.stop-y.start) for x, y in objects])
    
    if len(region_sizes) == 0:
        return cam_pred
    
    # 找出最大区域
    largest_idx = np.argmax(region_sizes)
    minr, minc = objects[largest_idx][0].start, objects[largest_idx][1].start
    maxr, maxc = objects[largest_idx][0].stop, objects[largest_idx][1].stop
    
    h = maxr - minr
    w = maxc - minc
    
    # 改进的动物姿态检测逻辑
    # 1. 使用更保守的比例阈值，避免过度判断为水平
    aspect_ratio_threshold = CAM_CONFIG.get("aspect_ratio_threshold", 0.85)  # 进一步降低阈值
    
    # 创建重心分析掩码
    roi_mask = np.zeros_like(binary)
    roi_mask[minr:maxr, minc:maxc] = binary[minr:maxr, minc:maxc]
    
    # 计算质心
    if np.sum(roi_mask) > 0:
        r_indices, c_indices = np.nonzero(roi_mask)
        mass_center_r = np.mean(r_indices)
        mass_center_c = np.mean(c_indices)
        
        # 分析动物形状 - 头部通常在上方或前方
        head_region_size = 0.3  # 头部区域占比
        
        # 垂直动物 - 检查上部区域密度
        top_region = roi_mask[minr:int(minr + h*head_region_size), minc:maxc]
        top_density = np.sum(top_region) / (top_region.size + 1e-8)
        
        # 水平动物 - 检查前部区域密度（左侧或右侧，视质心位置而定）
        center_c_relative = (mass_center_c - minc) / w
        if center_c_relative < 0.5:  # 质心偏左，头部可能在右侧
            front_region = roi_mask[minr:maxr, int(maxc - w*head_region_size):maxc]
        else:  # 质心偏右，头部可能在左侧
            front_region = roi_mask[minr:maxr, minc:int(minc + w*head_region_size)]
        front_density = np.sum(front_region) / (front_region.size + 1e-8)
        
        # 分析垂直质量分布
        upper_mask = r_indices < mass_center_r
        upper_mass = np.sum(upper_mask)
        lower_mass = len(r_indices) - upper_mass
        vertical_mass_ratio = upper_mass / (lower_mass + 1e-5)
        
        # 分析左右质量分布
        left_mask = c_indices < mass_center_c
        left_mass = np.sum(left_mask)
        right_mass = len(c_indices) - left_mass
        horizontal_balance = abs(left_mass - right_mass) / (len(c_indices) + 1e-5)
    else:
        mass_center_r = (minr + maxr) / 2
        mass_center_c = (minc + maxc) / 2
        top_density = 0
        front_density = 0
        vertical_mass_ratio = 1.0
        horizontal_balance = 0.5
    
    # 增强姿态判断逻辑，采用多特征融合
    # 形状特征：高宽比
    shape_score = h / (w + 1e-5)
    
    # 质量分布特征
    mass_score = vertical_mass_ratio if shape_score > 1 else (1 - vertical_mass_ratio)
    
    # 密度特征
    density_score = top_density if shape_score > 1 else front_density
    
    # 融合特征
    vertical_score = shape_score * 0.6 + mass_score * 0.3 + density_score * 0.1
    
    # 决策边界
    is_vertical_pose = (vertical_score > 0.8) or (h > w * 1.2 and vertical_mass_ratio > 0.4)
    is_horizontal = (w > h * aspect_ratio_threshold) and not is_vertical_pose
    
    # 中间姿态判断
    is_neutral = not (is_vertical_pose or is_horizontal)
    
    # 计算中心点 - 使用更智能的方法，考虑质心
    center_r = int(mass_center_r)
    center_c = int(mass_center_c)
    
    # 计算边界到图像边缘的距离，用于限制扩展
    distance_to_left = minc
    distance_to_right = cam_pred.shape[1] - maxc
    distance_to_top = minr
    distance_to_bottom = cam_pred.shape[0] - maxr
    
    # 根据边界距离计算最大允许扩展
    # 关键改进：限制扩展不超过图像尺寸的一定比例
    max_width_ratio = CAM_CONFIG.get("max_width_ratio", 0.8)  # 最大宽度不超过图像宽度的60%
    max_height_ratio = CAM_CONFIG.get("max_height_ratio", 0.8)  # 最大高度不超过图像高度的70%
    
    max_allowed_width = int(cam_pred.shape[1] * max_width_ratio)
    max_allowed_height = int(cam_pred.shape[0] * max_height_ratio)
    
    # 根据动物方向、大小和可能的物种调整扩展因子
    base_body_expansion = CAM_CONFIG.get("body_expansion_factor", 1.4)  # 调整默认值
    base_horizontal_expansion = CAM_CONFIG.get("horizontal_expansion_factor", 1.2)  # 进一步减少默认值
    
    # 根据检测到的区域大小调整扩展比例
    area_factor = np.sqrt(h * w) / 20  # 归一化因子
    area_factor = max(0.8, min(1.3, area_factor))  # 限制在合理范围内
    
    # 应用自适应扩展因子
    if is_vertical_pose:
        # 垂直动物 - 纵向扩展更多，横向扩展更少
        vertical_expansion = base_body_expansion * 1.3 * area_factor
        horiz_expansion = base_horizontal_expansion * 0.7  # 大幅减少水平扩展
        # 重设比例
        right_ratio = 0.5  # 水平居中
        down_ratio = 0.65  # 下方扩展更多
    elif is_horizontal:
        # 水平动物 - 横向扩展适中，不要太过分
        vertical_expansion = base_body_expansion * 0.9
        horiz_expansion = base_horizontal_expansion * 1.0 * area_factor  # 减少水平扩展
        # 重设水平动物的比例
        right_ratio = 0.5  # 水平居中
        down_ratio = 0.5
    else:
        # 中间状态 - 均衡扩展
        vertical_expansion = base_body_expansion
        horiz_expansion = base_horizontal_expansion * 0.9  # 减少水平扩展
        right_ratio = 0.5
        down_ratio = 0.55
    
    # 计算新的区域范围
    expanded_h = int(min(h * vertical_expansion, max_allowed_height))
    expanded_w = int(min(w * horiz_expansion, max_allowed_width))
    
    # 调整扩展中心点 - 考虑质心偏移
    center_offset_r = (center_r - (minr + maxr) / 2) / h  # 归一化的质心偏移
    
    # 根据质心位置动态调整上下扩展比例
    if center_offset_r < 0:  # 质心偏上
        down_ratio = max(0.6, down_ratio + 0.1)  # 增加下方扩展
    else:  # 质心偏下
        down_ratio = min(0.4, down_ratio - 0.1)  # 减少下方扩展
    
    # 设置新区域的起始和结束位置，确保在图像范围内同时不超出最大允许扩展
    expanded_minr = max(0, center_r - int(expanded_h * (1-down_ratio)))
    expanded_maxr = min(cam_pred.shape[0], center_r + int(expanded_h * down_ratio))
    expanded_minc = max(0, center_c - int(expanded_w * (1-right_ratio)))
    expanded_maxc = min(cam_pred.shape[1], center_c + int(expanded_w * right_ratio))
    
    # 确保扩展区域不会到达图像边缘，保持一定边距
    margin = CAM_CONFIG.get("edge_margin", 5)  # 到图像边缘的最小边距
    if expanded_minc < margin:
        expanded_minc = margin
    if expanded_maxc > cam_pred.shape[1] - margin:
        expanded_maxc = cam_pred.shape[1] - margin
    if expanded_minr < margin:
        expanded_minr = margin
    if expanded_maxr > cam_pred.shape[0] - margin:
        expanded_maxr = cam_pred.shape[0] - margin
    
    # 创建扩展掩码
    expanded_mask = np.zeros_like(binary)
    expanded_mask[expanded_minr:expanded_maxr, expanded_minc:expanded_maxc] = 1
    
    # 生成更精确的椭圆形掩码 - 考虑动物姿态
    y, x = np.ogrid[:cam_pred.shape[0], :cam_pred.shape[1]]
    
    # 重新计算扩展后的宽高
    expanded_h = expanded_maxr - expanded_minr
    expanded_w = expanded_maxc - expanded_minc
    
    # 根据动物姿态调整椭圆参数
    if is_vertical_pose:
        # 垂直动物使用更窄长的椭圆
        a = expanded_w // 2 * 0.85  # 进一步减小宽度
        b = expanded_h // 2 * 1.1  # 增加高度
    elif is_horizontal:
        # 水平动物使用更贴合的椭圆
        a = expanded_w // 2 * 0.95  # 减小宽度，避免过度扩展
        b = expanded_h // 2 * 0.95  # 保持合理高度
    else:
        # 默认椭圆
        a = expanded_w // 2 * 0.9  # 略微减小所有椭圆
        b = expanded_h // 2 * 0.9
    
    # 确保椭圆参数有效
    a = max(3, a)  # 防止a太小
    b = max(3, b)  # 防止b太小
    
    # 创建椭圆方程 - 中心位于扩展区域中心
    center_r_ellipse = expanded_minr + expanded_h // 2
    center_c_ellipse = expanded_minc + expanded_w // 2
    ellipse_boundary = ((x - center_c_ellipse)**2 / (a**2 + 1e-8) + 
                        (y - center_r_ellipse)**2 / (b**2 + 1e-8))
    
    # 创建软边界椭圆掩码，边缘更自然过渡
    ellipse_hardness = CAM_CONFIG.get("ellipse_hardness", 3.0)  # 进一步减小硬度
    ellipse_mask = 1 / (1 + np.exp(ellipse_hardness * (ellipse_boundary - 1)))
    
    # 增加纹理感知
    texture_sigma = CAM_CONFIG.get("texture_sigma", 1.0)
    texture_mask = ndimage.gaussian_filter(binary.astype(float), sigma=texture_sigma) - \
                   ndimage.gaussian_filter(binary.astype(float), sigma=texture_sigma*3)
    texture_mask = np.clip(texture_mask, 0, 1)
    
    # 检测并保留可能的细长结构（如尾巴、耳朵等）
    # 首先使用骨架提取找到主要结构
    skeleton = skeletonize(binary)
    
    # 向外膨胀以捕获细结构
    se = np.ones((3, 3), dtype=bool)  # 结构元素
    thin_structures = skeleton.copy()
    # 手动循环实现多次膨胀
    thin_iter = CAM_CONFIG.get("thin_structure_iterations", 3)  # 增加迭代次数以更好地捕获细节
    for _ in range(thin_iter):
        thin_structures = binary_dilation(thin_structures, se)
    
    # 只保留原始区域外的新增部分
    thin_structures = thin_structures & ~binary
    
    # 加权组合掩码 - 修复类型兼容性问题
    ellipse_weight = CAM_CONFIG.get("ellipse_weight", 0.9)  # 增加椭圆权重
    # 将浮点掩码转换为布尔类型，避免类型不兼容
    ellipse_float_mask = (ellipse_mask * ellipse_weight * expanded_mask) > 0
    
    # 防止椭圆掩码超出扩展区域
    ellipse_float_mask = ellipse_float_mask & expanded_mask
    
    # 组合掩码 - 使用布尔运算
    combined_mask = np.maximum(binary, ellipse_float_mask | thin_structures)
    
    # 考虑边缘约束 - 使用边缘信息引导掩码
    edge_constraint = edge_mask & expanded_mask
    combined_mask = np.maximum(combined_mask, edge_constraint)
    
    # 确保掩码不会延伸到图像边缘
    edge_margin_mask = np.ones_like(combined_mask)
    edge_margin_mask[:margin, :] = 0  # 上边缘
    edge_margin_mask[-margin:, :] = 0  # 下边缘
    edge_margin_mask[:, :margin] = 0  # 左边缘
    edge_margin_mask[:, -margin:] = 0  # 右边缘
    combined_mask = combined_mask & edge_margin_mask
    
    # 形态学闭操作，填充空洞
    closing_size = CAM_CONFIG.get("closing_size", 5)
    se = np.ones((closing_size, closing_size), dtype=bool)
    combined_mask = binary_closing(combined_mask, se)
    
    # 创建渐变边缘（而不是硬边界）
    distance = ndimage.distance_transform_edt(1 - binary)  # 到原始区域的距离
    max_distance = np.max(distance) if np.max(distance) > 0 else 1
    
    # 使用指数衰减函数使边缘更自然
    decay_factor = CAM_CONFIG.get("distance_decay", 2.0)
    normalized_distance = np.exp(-distance * decay_factor / max_distance)
    
    # 创建结果
    result = cam_pred.copy()
    extension_mask = (combined_mask > 0) & (binary == 0)
    
    if np.any(extension_mask):
        # 计算原始区域的平均值
        orig_mean = np.mean(cam_pred[binary > 0]) if np.any(binary > 0) else 0.5
        
        # 创建渐变值 - 使用乘法因子减小扩展区域的激活值
        decay_multiplier = CAM_CONFIG.get("decay_multiplier", 0.7)
        gradient_values = decay_multiplier * orig_mean * normalized_distance[extension_mask]
        result[extension_mask] = gradient_values
    
    # 应用高斯模糊，使边缘更平滑自然
    final_blur = CAM_CONFIG.get("final_blur_sigma", 0.8)
    result = ndimage.gaussian_filter(result, sigma=final_blur)
    
    # 最后增强一下高频细节，避免过度模糊
    highpass_sigma = CAM_CONFIG.get("highpass_sigma", 3.0)
    highpass_strength = CAM_CONFIG.get("highpass_strength", 0.3)
    highpass = cam_pred - ndimage.gaussian_filter(cam_pred, sigma=highpass_sigma)
    highpass = np.clip(highpass, 0, None)  # 只保留正值
    result = np.clip(result + highpass_strength * highpass, 0, 1)  # 添加一些高频细节
    
    return result

# 添加自定义高斯模糊函数作为kornia替代
def gaussian_blur2d(input_tensor, kernel_size=(15, 15), sigma=(1.5, 1.5)):
    """自定义的2D高斯模糊函数，代替kornia的gaussian_blur2d"""
    if isinstance(sigma, (int, float)):
        sigma = (float(sigma), float(sigma))
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    # 获取输入张量的空间尺寸
    _, _, h, w = input_tensor.shape
    
    # 确保kernel_size不大于输入尺寸
    kernel_size_x = min(kernel_size[0], w)
    kernel_size_y = min(kernel_size[1], h)
    
    # 确保kernel_size是奇数
    kernel_size_x = kernel_size_x if kernel_size_x % 2 == 1 else kernel_size_x - 1
    kernel_size_y = kernel_size_y if kernel_size_y % 2 == 1 else kernel_size_y - 1
    
    # 如果尺寸太小，返回原始输入
    if kernel_size_x < 3 or kernel_size_y < 3:
        # 删除警告打印
        return input_tensor
    
    # 计算x方向的高斯核
    sigma_x = sigma[0]
    x = torch.arange(-(kernel_size_x // 2), kernel_size_x // 2 + 1, device=input_tensor.device)
    gauss_kernel_x = torch.exp(-x.pow(2.0) / (2 * sigma_x ** 2))
    gauss_kernel_x = gauss_kernel_x / gauss_kernel_x.sum()
    
    # 计算y方向的高斯核
    sigma_y = sigma[1]
    y = torch.arange(-(kernel_size_y // 2), kernel_size_y // 2 + 1, device=input_tensor.device)
    gauss_kernel_y = torch.exp(-y.pow(2.0) / (2 * sigma_y ** 2))
    gauss_kernel_y = gauss_kernel_y / gauss_kernel_y.sum()
    
    # 应用可分离卷积 (先x方向，再y方向)
    padding_x = kernel_size_x // 2
    padding_y = kernel_size_y // 2
    
    try:
        # x方向卷积
        output = F.conv2d(input_tensor, 
                        gauss_kernel_x.view(1, 1, -1, 1).repeat(input_tensor.size(1), 1, 1, 1),
                        padding=(0, padding_x), groups=input_tensor.size(1))
        
        # y方向卷积
        output = F.conv2d(output, 
                        gauss_kernel_y.view(1, 1, 1, -1).repeat(input_tensor.size(1), 1, 1, 1),
                        padding=(padding_y, 0), groups=input_tensor.size(1))
        
        return output
    except RuntimeError:
        # 删除错误打印
        # 备用方案：使用更简单的方法，单个2D卷积
        try:
            # 创建2D高斯核
            gaussian_kernel = torch.outer(gauss_kernel_y, gauss_kernel_x)
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size_y, kernel_size_x)
            gaussian_kernel = gaussian_kernel.repeat(input_tensor.size(1), 1, 1, 1)
            
            # 单次2D卷积
            output = F.conv2d(input_tensor, gaussian_kernel, 
                             padding=(padding_y, padding_x), groups=input_tensor.size(1))
            return output
        except Exception:
            # 删除错误打印
            # 最后的备用方案：返回原始张量
            return input_tensor

if __name__ == "__main__":
    import argparse
    
    # 检测可用计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
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