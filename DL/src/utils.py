import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2  
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from config import FAST_TEST_CONFIG, CLASSIFIER_CONFIG

def set_seed(seed):
    """
    为所有随机数生成器设置随机种子以保证可重复性。
    
    参数:
        seed: 设置为随机种子的整数值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_transform(is_training=True, size=None):
    """
    获取用于训练或评估的图像转换管道。
    
    参数:
        is_training: 布尔值，指示转换是否用于训练
        size: 图像调整大小的(高度, 宽度)元组
        
    返回:
        transforms.Compose: 组合的转换管道
    """
    if size is None:
        size = CLASSIFIER_CONFIG['image_size']
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

class PetDataset(Dataset):
    """
    用于牛津-IIIT宠物数据集的二分类(猫/狗)数据集类。
    
    属性:
        image_dir: 包含图像文件的目录
        transform: 图像转换管道
        is_training: 布尔值，指示数据集是否用于训练
        image_paths: 图像文件路径列表
        cat_breeds: 猫品种名称列表
        dog_breeds: 狗品种名称列表
    """
    
    def __init__(self, image_dir, transform=None, is_training=True):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_training = is_training
        self.image_paths = list(self.image_dir.glob('*.jpg'))
        
        self.cat_breeds = [
            'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British', 'Egyptian', 'Maine', 
            'Persian', 'Ragdoll', 'Russian', 'Siamese', 'Sphynx'
        ]
        self.dog_breeds = [
            'american', 'basset', 'beagle', 'boxer', 'chihuahua', 'english', 'german', 
            'great', 'japanese', 'keeshond', 'leonberger', 'miniature', 'newfoundland', 
            'pomeranian', 'pug', 'saint', 'samoyed', 'scottish', 'shiba', 'staffordshire', 
            'wheaten', 'yorkshire'
        ]
        
    def __len__(self):
        """
        获取数据集中样本的总数。
        
        返回:
            int: 样本数量
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        从数据集中获取单个样本。
        
        参数:
            idx: 要检索的样本的索引
            
        返回:
            dict: 包含以下内容的字典:
                - image: 转换后的图像张量
                - label: 二值标签(0表示猫，1表示狗)
                - image_path: 原始图像文件的路径
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        filename = img_path.stem
        breed = filename.split('_')[0]
        
        if any(breed.lower().startswith(cat.lower()) for cat in self.cat_breeds):
            label = 0
        else:
            label = 1
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'image_path': str(img_path)
        }

def get_dataloaders(image_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, fast_test=None, config=None):
    """
    创建数据加载器，应用性能优化
    
    参数:
        image_dir: 图像目录路径
        batch_size: 批量大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        fast_test: 快速测试模式
        config: 配置字典，包含优化参数
    
    返回:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 确保配置有效
    if config is None:
        from config import CLASSIFIER_CONFIG
        config = CLASSIFIER_CONFIG
    
    # 提取数据加载优化参数
    num_workers = config.get('num_workers', 4)  # 默认4个工作线程
    pin_memory = config.get('pin_memory', True)  # 默认启用内存固定
    prefetch_factor = config.get('prefetch_factor', 2)  # 数据预取因子
    
    # 预处理设置
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_size = config.get('image_size', (224, 224))
    
    # 高效数据变换：减少预处理开销
    transform = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.1)),  # 稍微大一点的尺寸用于随机裁剪
        transforms.CenterCrop(image_size),  # 使用中心裁剪替代随机裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 为每个集合创建数据集
    image_dir = Path(image_dir)
    dataset = PetDataset(image_dir, transform=transform)
    
    # 如果在快速测试模式，则限制样本数量
    if fast_test:
        max_samples = fast_test.get('max_samples', 100)
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # 拆分数据集
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # 使用高效的数据加载器配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),  # 保持工作线程活跃，减少创建开销
        drop_last=True  # 丢弃最后一个不完整批次，避免不必要的计算
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证时使用更大批量，不需要反向传播
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # 测试时使用更大批量
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_loader

def visualize_cam(image, cam, save_path=None, alpha=0.5):
    """
    Visualize a Class Activation Map (CAM) over an image.
    
    Args:
        image: Image tensor or numpy array
        cam: Class activation map numpy array
        save_path: Path to save the visualization
        alpha: Transparency factor for overlay
    """
    try:
        # 如果需要，将张量转换为numpy
        if isinstance(image, torch.Tensor):
            # 确保图像在CPU上
            image = image.cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = image.permute(1, 2, 0).numpy()
            image = np.clip(image, 0, 1)
        
        # 将图像转换为uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # 处理灰度图像
        if len(image_uint8.shape) == 2:
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        elif image_uint8.shape[2] == 1:
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        
        # 确保CAM形状正确且已归一化
        h, w = image_uint8.shape[:2]
        
        # 调整CAM大小以匹配图像尺寸
        cam_resized = cv2.resize(cam, (w, h))
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        
        # 将CAM应用颜色映射
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # 检查形状兼容性
        if image_uint8.shape[:2] != heatmap.shape[:2]:
            print(f"Shape mismatch: image={image_uint8.shape}, heatmap={heatmap.shape}")
            return
        
        # 如果图像是RGB，则转换为BGR (OpenCV使用BGR)
        if image_uint8.shape[2] == 3:
            image_uint8_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        else:
            image_uint8_bgr = image_uint8
        
        # 创建叠加图
        overlay = cv2.addWeighted(image_uint8_bgr, 1-alpha, heatmap, alpha, 0)
        
        # 创建并排可视化
        result = np.zeros((h, w*2, 3), dtype=np.uint8)
        result[:, :w] = image_uint8_bgr
        result[:, w:] = overlay
        
        # 添加标签
        cv2.putText(result, 'Original Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'CAM Overlay', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 保存或显示结果
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(save_path), result)
        else:
            cv2.imshow('CAM Visualization', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in visualize_cam: {e}")
        # Try to save a grayscale version as fallback
        try:
            cv2.imwrite(str(save_path), cam_uint8)
            print(f"Saved alternative grayscale CAM to {save_path}")
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")

def save_cam(image_path, cam, save_dir, file_suffix='_cam'):
    """
    Save a CAM visualization for a given image.
    
    Args:
        image_path: Path to the input image
        cam: Class activation map
        save_dir: Directory to save the visualization
        file_suffix: Suffix to add to the file name
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    image_path = Path(image_path)
    save_path = save_dir / f"{image_path.stem}{file_suffix}.png"
    
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            h, w = cam.shape
            image = np.zeros((h, w, 3), dtype=np.uint8)
        
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        assert image.shape == heatmap.shape, f"Shape mismatch: image={image.shape}, heatmap={heatmap.shape}"
        
        overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
        
        cv2.imwrite(str(save_path), overlay)
        
        return save_path
    except Exception as e:
        print(f"Failed to save CAM visualization: {e}")
        cam_uint8 = (cam * 255).astype(np.uint8)
        cv2.imwrite(str(save_path), cam_uint8)
        return save_path 

def get_cam_dataloaders(dataset, batch_size=8):
    """创建专用于CAM生成的DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,  # 使用较小的批次大小避免内存问题
        shuffle=False,  # 不需要打乱顺序
        num_workers=min(os.cpu_count(), 4) if os.cpu_count() else 2,
        pin_memory=True,
        drop_last=False  # 确保处理所有图像
    ) 