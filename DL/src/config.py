import os
from pathlib import Path
import torch
import cv2

# 项目路径
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")))

# 数据路径
IMAGE_DIR = DATA_ROOT / "images"
ANNOTATION_DIR = DATA_ROOT / "annotations"

# 输出路径
CAM_DIR = OUTPUT_ROOT / "cams"
PSEUDO_MASK_DIR = OUTPUT_ROOT / "pseudo_masks"
SEGMENTATION_DIR = OUTPUT_ROOT / "base_pseudo"


# 模型路径
CLASSIFIER_DIR = MODEL_ROOT / "classifier"
SEGMENTOR_DIR = MODEL_ROOT / "segmentor"

# 确保目录存在
for dir_path in [
    OUTPUT_ROOT, 
    CAM_DIR,
    MODEL_ROOT, CLASSIFIER_DIR, SEGMENTOR_DIR
]:
    dir_path.mkdir(exist_ok=True, parents=True)

# CAM生成参数
CAM_CONFIG = {
    "gamma": 0.5,                      # gamma值，用于调整对比度
    "edge_enhancement": 1.5,           # 边缘增强系数
    "threshold": 0.05,                 # 阈值，用于分离前景和背景
    "apply_canny": True,               # 是否启用Canny边缘检测
    "canny_low": 30,                   # Canny低阈值
    "canny_high": 100,                 # Canny高阈值
    "use_adaptive_threshold": True,   # 是否使用自适应阈值
    "use_morphology": True,            # 是否使用形态学操作
    "colormap": cv2.COLORMAP_JET,      # 热图颜色映射
    "overlay_alpha": 0.7,              # 热图透明度
    "multi_scale_fusion": True,        # 是否使用多尺度特征融合
    "morphology_kernel_size": 5,       # 形态学核大小
    "edge_detector_kernel_size": 3,    # 边缘检测器核大小
    "foreground_bias": 1.5,            # 前景偏置
    "dilate_iterations": 2,            # 形态学膨胀迭代次数
    "body_expansion_factor": 1.3       # 身体区域扩展系数
}

# 基础掩码生成参数
BASE_MASK_CONFIG = {
    "threshold": 0.5,                  # 阈值，用于分离前景和背景
    "adaptive_threshold": True,        # 是否使用自适应阈值
    "morph_kernel_size": 5             # 形态学操作的核大小
}

# CRF后处理参数
CRF_CONFIG = {
    "pos_w": 3,                        # 标准位置权重
    "pos_xy_std": 5,                   # 位置标准差，用于更好的空间一致性
    "bi_w": 7,                         # 双边权重，用于更好的边缘附着
    "bi_xy_std": 80,                   # 更大的空间标准差，用于更好的区域连贯性
    "bi_rgb_std": 13,                  # 更高的颜色标准差，以更容忍颜色变化
    "iterations": 10                    # 迭代次数，以实现更好的收敛
}

# 分类器训练参数
CLASSIFIER_CONFIG = {
    "backbone": "resnet18",      # 使用的骨干网络
    "num_classes": 2,            # 类别数量
    "image_size": (224, 224),    # 图像大小
    "batch_size": 64,            # 批量大小
    "lr": 0.0005,                # 学习率
    "weight_decay": 1e-4,        # 权重衰减
    "epochs": 10,                # 训练轮次
    "cam_threshold": 0.4,        # CAM阈值
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "weighted_loss": True,       # 是否使用带权重的损失函数
    "use_pretrained": True,      # 是否使用预训练模型
    "early_stopping_patience": 5, # 早停耐心值
    "lr_scheduler": "cosine",    # 学习率调度器
    "warmup_epochs": 1,          # 预热轮次
    "use_mixed_precision": True, # 是否使用混合精度训练
    "num_workers": 8,            # 数据加载并行度
    "pin_memory": True,          # 是否启用内存固定
    "prefetch_factor": 2,        # 数据预取因子
    "non_blocking": True         # 是否使用非阻塞数据传输
}

FAST_TEST_CONFIG = {
    "enabled": False, 
    "max_samples": 50,  # 最大样本数量
    "epochs": 1,        # 训练轮次
    "batch_size": 16    # 批量大小
}

# 数据集划分
DATASET_CONFIG = {
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "random_seed": 44
} 