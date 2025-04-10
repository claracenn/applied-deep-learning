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

# CAM提取参数，加强对动物轮廓的贴合能力
CAM_CONFIG = {
    "gaussian_sigma": 2.5,  # 调整高斯滤波的标准差，ResNet50特征通常更精细
    "edge_enhancement": 4.0,  # 增强边缘增强系数，更好地突出动物轮廓
    "threshold": 0.22,  # 略微降低阈值，ResNet50的激活可能更分散
    "foreground_bias": 3.0,  # 增加前景类别的权重偏置以聚焦在动物上
    "use_gaussian_constraint": True,  # 使用高斯约束避免矩形形状
    "morphology_iterations": 2,  # 保持形态学操作的迭代次数
    "mask_dir": str(DATA_ROOT / "annotations/trimaps"),  # 原始标注的目录
    "colormap": "viridis",  # 使用viridis颜色映射代替jet，更好的视觉效果
    
    # 增加边缘敏感性参数
    "edge_sensitivity": 1.5,  # 略微减少边缘检测敏感度，ResNet50提取的边缘已较清晰
    "body_expansion_factor": 1.3,  # 降低身体扩展因子，随着模型能力增强扩展可以更保守
    "horizontal_expansion_factor": 1.4,  # 降低水平扩展因子
    "ellipse_weight": 0.9,  # 增加椭圆权重使掩码更自然
    "ellipse_hardness": 3.2,  # 减小椭圆边缘硬度使过渡更自然
    
    # 形态学操作参数
    "morphology_kernel_size": 5,  # 形态学操作核大小
    "dilate_iterations": 2,  # 膨胀操作迭代次数
    "thin_structure_iterations": 3,  # 增加细结构提取迭代次数，捕获更多细节
    "closing_size": 5,  # 闭操作核大小
    
    # 动物检测与扩展参数
    "aspect_ratio_threshold": 1.2,  # 水平/垂直姿态判断阈值
    "horizontal_right_ratio": 0.55,  # 水平动物右侧扩展比例
    "vertical_down_ratio": 0.6,  # 垂直动物下方扩展比例
    "vertical_right_ratio": 0.5,  # 垂直动物右侧扩展比例
    
    # 边缘和细节增强参数
    "distance_decay": 1.8,  # 减小距离衰减因子，使扩展更自然
    "decay_multiplier": 0.75,  # 增加扩展区域衰减乘数
    "texture_sigma": 0.8,  # 减小纹理提取高斯模糊参数，增强纹理细节
    "final_blur_sigma": 0.7,  # 减小最终高斯模糊参数
    "highpass_sigma": 3.0,  # 保持高通滤波器高斯模糊参数
    "highpass_strength": 0.35,  # 略微增加高通滤波器增强强度
    
    # 像素到原型对比学习参数
    "prototype_contrast": True,  # 是否使用像素到原型对比学习
    "prototype_weight": 0.15,  # 保持对比学习损失的权重
    "temperature": 0.07,  # 进一步降低对比学习温度参数使区分更明显
    "contrast_threshold": 0.25,  # 保持对比学习的像素CAM阈值
    "prototype_topk": 200,  # 增加构建原型时使用的高激活像素数量，ResNet50特征更丰富
    
    # 添加gamma参数，之前代码中使用但配置中缺失
    "gamma": 0.3,  # CAM阈值之前的指数加强
    "cam_gamma": 2.0,  # 伽马校正参数
    "activation_threshold": 0.2  # 低激活区域抑制阈值
}

# 基础掩码生成参数
BASE_MASK_CONFIG = {
    "threshold": 0.18,                 # 略微提高阈值以减少伪标签噪声
    "adaptive_threshold": True,        # 保持自适应阈值
    "morph_kernel_size": 5,            # 保持形态学核大小
    "target_foreground_percent": 30.0, # 目标前景比例与原数据集一致
    "max_foreground_percent": 42.0,    # 略微降低最大前景比例以避免过度扩展
    "uncertainty_threshold": 0.15,     # 提高不确定性阈值以减少不确定区域
    "edge_width": 5                    # 减少边缘宽度
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
    "backbone": "resnet50",      # 使用的骨干网络，从resnet18改为resnet50
    "num_classes": 2,            # 类别数量
    "image_size": (224, 224),    # 图像大小
    "batch_size": 32,            # 批量大小，适当减小以适应ResNet50
    "lr": 0.0003,                # 学习率，略微调整
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

# 分割模型训练关键参数
SEGMENTATION_CONFIG = {
    # 模型结构参数
    "backbone": "resnet50",            # 使用的骨干网络
    "atrous_rates": (6, 12, 18, 24),   # 空洞卷积率
    "num_classes": 2,                  # 类别数量
    "image_size": (224, 224),          # 图像大小 (height, width)
    
    # 训练控制参数
    "batch_size": 8,                   # 批量大小
    "num_epochs": 5,                   # 训练轮次
    "learning_rate": 1e-4,             # 学习率
    "weight_decay": 1e-4,              # 权重衰减
    "test_ratio": 0.2,                 # 测试集比例
    "save_every": 5,                   # 每隔多少个epoch保存一次模型
    "eval_every": 1,                   # 每隔多少个epoch评估一次模型
    "num_workers": 4,                  # 数据加载线程数
    "seed": 42,                        # 随机种子
    
    # 损失函数参数
    "weighted_loss": True,             # 是否使用带权重的损失函数
    "foreground_weight": 1.5,          # 前景类权重 (调高以关注前景)
    "background_weight": 0.5,          # 背景类权重
    
    # 控制训练行为的开关
    "base_only": False,                # 是否只使用基础伪标签训练
    "crf_only": False,                 # 是否只使用CRF伪标签训练
    "early_stopping": True,            # 是否启用早停
    "patience": 5                      # 早停耐心值
}

# 核心路径配置
SEGMENTATION_PATHS = {
    "img_dir": str(IMAGE_DIR),                        # 图像目录
    "base_mask_dir": str(SEGMENTATION_DIR),           # 基础伪标签目录
    "crf_mask_dir": str(PSEUDO_MASK_DIR),             # CRF处理后的伪标签目录
    "gt_mask_dir": str(ANNOTATION_DIR / "trimaps"),   # 真实标签目录
    "model_dir": str(SEGMENTOR_DIR),                  # 模型保存目录
    "result_dir": str(OUTPUT_ROOT / "results"),       # 结果保存目录
    "trainval_file": str(ANNOTATION_DIR / "trainval.txt"),  # 训练集列表文件
    "test_file": str(ANNOTATION_DIR / "test.txt")     # 测试集列表文件
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