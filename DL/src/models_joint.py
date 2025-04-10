"""
联合模型定义，将分割模型和ScoreNet组合在一起
"""

import torch
import torch.nn as nn
from scorenet import ScoreNet
from m3_train_segmentation import DeepLabLargeFOV

class JointSegmentationModel(nn.Module):
    """
    联合分割模型，包含分割网络和ScoreNet
    """
    def __init__(self, num_classes=2, score_channels=16, atrous_rates=(6, 12, 18, 24)):
        super(JointSegmentationModel, self).__init__()
        
        # 初始化分割模型
        self.segmentor = DeepLabLargeFOV(
            num_classes=num_classes,
            atrous_rates=atrous_rates
        )
        
        # 初始化ScoreNet
        self.scorenet = ScoreNet(
            in_channels=4,  # 3通道图像 + 1通道CAM
            base_channels=score_channels
        )
    
    def forward(self, image, cam=None, initial_mask=None):
        """
        前向传播
        
        参数:
            image: 输入图像 [B, 3, H, W]
            cam: CAM特征图 [B, 1, H, W]，可选
            initial_mask: 初始伪标签 [B, H, W]，可选
            
        返回:
            seg_pred: 分割预测
            conf_map: 置信度图（如果提供了CAM）
        """
        # 分割预测
        seg_pred = self.segmentor(image)
        
        # 如果提供了CAM，生成置信度图
        conf_map = None
        if cam is not None:
            conf_map = self.scorenet(image, cam)
        
        return seg_pred, conf_map
    
    def load_pretrained(self, segmentor_path=None, scorenet_path=None):
        """
        加载预训练模型权重
        """
        if segmentor_path:
            self.segmentor.load_state_dict(
                torch.load(segmentor_path, map_location='cpu')
            )
            print(f"Loaded pretrained segmentor from {segmentor_path}")
        
        if scorenet_path:
            self.scorenet.load_state_dict(
                torch.load(scorenet_path, map_location='cpu')
            )
            print(f"Loaded pretrained scorenet from {scorenet_path}")
    
    def get_confidence_map(self, image, cam):
        """
        单独获取置信度图
        """
        with torch.no_grad():
            return self.scorenet(image, cam)
    
    def get_segmentation(self, image):
        """
        单独获取分割预测
        """
        with torch.no_grad():
            return self.segmentor(image) 