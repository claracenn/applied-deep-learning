import os
from pathlib import Path
import torch

# 项目路径
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).resolve()
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_ROOT = PROJECT_ROOT / "models"  

# 数据路径
IMAGE_DIR = DATA_ROOT / "images"
ANNOTATION_DIR = DATA_ROOT / "annotations"

# 数据集划分文件 - 只有训练集和测试集
TRAIN_FILE = ANNOTATION_DIR / "trainval.txt"  # 使用trainval.txt作为训练集
TEST_FILE = ANNOTATION_DIR / "test.txt"       # 使用test.txt作为测试集

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
    PSEUDO_MASK_DIR,
    SEGMENTATION_DIR,
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
    "colormap": "jet",                  # 热图颜色映射，移除cv2引用
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
    "adaptive_threshold": False,        # 是否使用自适应阈值
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

# 数据集划分
# 注意：此配置仅在使用随机划分时生效 (使用 --use_random_split 参数)
# 默认情况下使用官方数据集划分文件 (trainval.txt 和 test.txt)

SEGMENTATION_CONFIG = {
    "backbone": "resnet50",            # 使用的骨干网络
    "atrous_rates": (6, 12, 18, 24),  # 空洞卷积率
    "num_classes": 2,                  # 类别数量
    "batch_size": 8,                   # 批量大小
    "num_epochs": 5,                   # 训练轮次
    "learning_rate": 1e-4,             # 学习率
    "weight_decay": 1e-4,              # 权重衰减
    "train_ratio": 0.8,                # 训练集比例
    "test_ratio": 0.2,                # 测试集比例
    "image_size": (224, 224),          # 图像大小 (height, width)
    "save_every": 5,                   # 每隔多少个epoch保存一次模型
    "eval_every": 1,                   # 每隔多少个epoch评估一次模型
    "num_workers": 0,                  # 数据加载线程数
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 训练设备
    "weighted_loss": True,             # 是否使用带权重的损失函数
    "foreground_weight": 0.5,          # 前景类权重 (修改为平衡值)
    "background_weight": 0.5           # 背景类权重 (修改为平衡值)
}

# 分割数据集路径 - 改为使用基于PROJECT_ROOT的绝对路径
SEGMENTATION_PATHS = {
    "img_dir": str(IMAGE_DIR),                    # 使用已定义的IMAGE_DIR
    "mask_dir": str(SEGMENTATION_DIR),            # 使用已定义的SEGMENTATION_DIR
    "crf_mask_dir": str(PSEUDO_MASK_DIR),         # 使用已定义的PSEUDO_MASK_DIR
    "model_dir": str(SEGMENTOR_DIR),              # 使用已定义的SEGMENTOR_DIR
    "result_dir": str(OUTPUT_ROOT / "results")    # 结果目录
}

# 全监督分割训练参数
FULLY_SUP_CONFIG = {
    "backbone": "resnet50",           # 使用的骨干网络
    "atrous_rates": (6, 12, 18, 24),  # 空洞卷积率
    "num_classes": 2,                 # 类别数量
    "batch_size": 8,                  # 批量大小
    "num_epochs": 10,                 # 训练轮次
    "learning_rate": 1e-4,            # 学习率
    "weight_decay": 1e-4,             # 权重衰减
    "train_ratio": 0.8,               # 训练集比例
    "test_ratio": 0.2,               # 测试集比例
    "image_size": (224, 224),         # 图像大小
    "eval_every": 1,                  # 每隔多少个epoch评估一次
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4                  # 数据加载线程数
}

# 全监督分割路径
FULLY_SUP_PATHS = {
    "img_dir": str(IMAGE_DIR),                        # 使用已定义的IMAGE_DIR
    "mask_dir": str(DATA_ROOT / "annotations/trimaps"),
    "model_dir": str(MODEL_ROOT / "segmentor/fully_supervised"),
    "result_dir": str(OUTPUT_ROOT / "results/fully_supervised")
}