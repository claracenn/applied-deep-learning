"""
联合损失函数定义，包括分割损失和ScoreNet正则化损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice损失函数"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = 1e-5  # 平滑系数
    
    def forward(self, pred, target):
        # 将预测转换为one-hot编码
        pred = torch.softmax(pred, dim=1)
        
        # 二分类情况下，同时关注前景类和背景类
        if self.num_classes == 2:
            # 计算背景类的Dice
            pred_bg = pred[:, 0]
            target_bg = (target == 0).float()
            intersection_bg = (pred_bg * target_bg).sum()
            union_bg = pred_bg.sum() + target_bg.sum()
            dice_bg = (2.0 * intersection_bg + self.smooth) / (union_bg + self.smooth)
            
            # 计算前景类的Dice
            pred_fg = pred[:, 1]
            target_fg = (target == 1).float()
            intersection_fg = (pred_fg * target_fg).sum()
            union_fg = pred_fg.sum() + target_fg.sum()
            dice_fg = (2.0 * intersection_fg + self.smooth) / (union_fg + self.smooth)
            
            # 更平衡的权重
            return 1.0 - (0.4 * dice_bg + 0.6 * dice_fg)
        
        # 多分类情况
        dice = 0.0
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = (target == i).float()
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            class_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            
            # 前景类权重比背景略高
            if i == 0:  # 背景类
                dice += 0.4 * class_dice
            else:  # 前景类
                dice += 0.6 * class_dice / (self.num_classes - 1)
        
        return 1.0 - dice

class SegmentationLoss(nn.Module):
    """分割损失函数"""
    def __init__(self, num_classes=2, weights=None, use_dice=True, dice_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        
        # 将权重列表转换为PyTorch张量
        if weights is None:
            # 默认给予前景类更高权重
            weights = [0.01, 0.99]  # 99%权重分配给前景类
        
        # 确保权重是张量类型
        if isinstance(weights, list):
            weights = torch.FloatTensor(weights)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = weights.to(device)
        
        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)
        
        # Dice损失
        if use_dice:
            self.dice_loss = DiceLoss(num_classes=num_classes)
    
    def to(self, device):
        """重写to方法，确保权重移动到正确的设备"""
        super().to(device)
        if self.weights is not None:
            self.weights = self.weights.to(device)
            self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)
        return self
        
    def forward(self, pred, target):
        # 检查是否有前景像素
        has_foreground = (target == 1).float().sum() > 0
        
        # 计算交叉熵损失
        ce_loss = self.ce_loss(pred, target)
        
        if self.use_dice and has_foreground:
            # 计算Dice损失
            dice_loss_val = self.dice_loss(pred, target)
            # 使用更高的Dice损失权重
            return ce_loss * (1 - self.dice_weight) + dice_loss_val * self.dice_weight
        
        return ce_loss

class ScoreNetRegularization(nn.Module):
    """ScoreNet的正则化损失"""
    def __init__(self, smoothness_weight=0.1, sparsity_weight=0.1):
        super().__init__()
        # 直接在GPU上创建权重张量
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.smoothness_weight = torch.tensor(smoothness_weight, device=device)
        self.sparsity_weight = torch.tensor(sparsity_weight, device=device)
    
    def to(self, device):
        """重写to方法，确保权重移动到正确的设备"""
        super().to(device)
        self.smoothness_weight = self.smoothness_weight.to(device)
        self.sparsity_weight = self.sparsity_weight.to(device)
        return self
    
    def smoothness_loss(self, conf_map):
        """平滑损失：鼓励相邻像素的置信度值相近"""
        # 计算水平和垂直方向的梯度
        dx = torch.abs(conf_map[:, :, :, :-1] - conf_map[:, :, :, 1:])
        dy = torch.abs(conf_map[:, :, :-1, :] - conf_map[:, :, 1:, :])
        
        return (dx.mean() + dy.mean()) / 2
    
    def sparsity_loss(self, conf_map):
        """稀疏性损失：鼓励置信度图更加稀疏"""
        return torch.mean(torch.abs(conf_map))
    
    def forward(self, conf_map):
        smoothness = self.smoothness_loss(conf_map)
        sparsity = self.sparsity_loss(conf_map)
        
        return self.smoothness_weight * smoothness + self.sparsity_weight * sparsity

class JointLoss(nn.Module):
    """联合损失：组合分割损失和ScoreNet正则化损失"""
    def __init__(self, num_classes=2, seg_weights=None, 
                 use_dice=True, dice_weight=0.5,
                 score_smoothness_weight=0.1,
                 score_sparsity_weight=0.1,
                 score_reg_weight=0.1):
        super().__init__()
        
        # 直接在GPU上创建权重张量
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_reg_weight = torch.tensor(score_reg_weight, device=device)
        
        # 处理分割权重
        # 注意：seg_weights将在SegmentationLoss中进一步处理
        
        # 初始化子损失函数
        self.seg_loss = SegmentationLoss(
            num_classes=num_classes,
            weights=seg_weights,
            use_dice=use_dice,
            dice_weight=dice_weight
        )
        
        self.score_reg = ScoreNetRegularization(
            smoothness_weight=score_smoothness_weight,
            sparsity_weight=score_sparsity_weight
        )
    
    def to(self, device):
        """重写to方法，确保所有组件都在正确的设备上"""
        super().to(device)
        self.seg_loss = self.seg_loss.to(device)
        self.score_reg = self.score_reg.to(device)
        self.score_reg_weight = self.score_reg_weight.to(device)
        return self
    
    def forward(self, seg_pred, target, conf_map):
        """
        计算联合损失
        
        参数:
            seg_pred: 分割模型的预测结果
            target: 目标标签
            conf_map: ScoreNet生成的置信度图
            
        返回:
            total_loss: 总损失
            loss_dict: 包含各个子损失的字典
        """
        # 确保所有输入都在同一个设备上
        device = seg_pred.device
        target = target.to(device)
        conf_map = conf_map.to(device)
        
        # 计算分割损失
        seg_loss = self.seg_loss(seg_pred, target)
        
        # 计算ScoreNet正则化损失
        score_reg_loss = self.score_reg(conf_map)
        
        # 计算总损失
        total_loss = seg_loss + self.score_reg_weight * score_reg_loss
        
        # 返回总损失和损失明细
        loss_dict = {
            'total_loss': total_loss.item(),
            'seg_loss': seg_loss.item(),
            'score_reg_loss': score_reg_loss.item()
        }
        
        return total_loss, loss_dict 