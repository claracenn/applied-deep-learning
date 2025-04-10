import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
# import cv2  
from PIL import Image, ImageFilter, ImageDraw
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

# from config import FAST_TEST_CONFIG, CLASSIFIER_CONFIG
from config import CLASSIFIER_CONFIG, TRAIN_FILE, TEST_FILE

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
        labels_dict: 字典，将图像ID映射到标签(0表示猫，1表示狗)
    """
    
    def __init__(self, image_dir, transform=None, is_training=True, labels_dict=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_training = is_training
        self.labels_dict = labels_dict or {}  # 如果没有提供标签字典，则使用空字典
        
        # 获取所有图像文件路径
        self.image_paths = list(self.image_dir.glob('*.jpg'))
        
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
        
        # 获取文件名（不带扩展名）
        filename = img_path.stem
        
        # 如果有标签字典，则使用它
        if filename in self.labels_dict:
            # 标签值 1=猫, 2=狗，我们转换为 0=猫, 1=狗
            species = self.labels_dict[filename]
            label = species - 1  # 将1,2转换为0,1
        else:
            # 没有标签字典，则通过文件名判断
            # 大写字母开头的是猫，小写字母开头的是狗
            if filename[0].isupper():
                label = 0  # 猫
            else:
                label = 1  # 狗
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'image_path': str(img_path)
        }

def get_dataloaders(image_dir, batch_size=32, test_ratio=0.2, seed=42, config=None):
    """
    创建数据加载器，使用官方数据集划分，从test.txt中选择一部分作为测试集
    
    参数:
        image_dir: 图像目录路径
        batch_size: 批量大小
        test_ratio: 从test.txt中选取的比例作为测试集
        seed: 随机种子，用于控制测试集选择的可重现性
        config: 配置字典，包含优化参数
    
    返回:
        tuple: (train_loader, test_loader)
    """
    # 确保配置有效
    if config is None:
        from config import CLASSIFIER_CONFIG
        config = CLASSIFIER_CONFIG
    
    # 使用官方数据集划分
    train_loader, test_loader = get_dataloaders_from_split(
        image_dir=image_dir,
        batch_size=batch_size,
        test_ratio=test_ratio,
        seed=seed,
        config=config
    )
    
    return train_loader, test_loader

def visualize_cam(image, cam, save_path=None, alpha=0.5):
    try:
        if isinstance(image, torch.Tensor):
            image = image.cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = image.permute(1, 2, 0).numpy()
            image = np.clip(image, 0, 1)

        # Ensure cam is a valid numpy array with values in [0, 1]
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()
        cam = np.clip(cam, 0, 1)

        image_uint8 = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8).convert("RGB")
        w, h = pil_image.size

        cam_uint8 = (cam * 255).astype(np.uint8)
        cam_resized = Image.fromarray(cam_uint8).resize((w, h)).convert("L")

        # 将CAM映射为热图（简单替代色）
        heatmap = Image.new("RGB", (w, h))
        cam_array = np.array(cam_resized)
        heat_array = np.zeros((h, w, 3), dtype=np.uint8)
        heat_array[..., 0] = cam_array  # R
        heat_array[..., 1] = cam_array // 2  # G
        heatmap = Image.fromarray(heat_array)

        # 叠加
        overlay = Image.blend(pil_image, heatmap, alpha)

        # 并排显示
        total_width = w * 2
        result = Image.new('RGB', (total_width, h))
        result.paste(pil_image, (0, 0))
        result.paste(overlay, (w, 0))

        # 添加文字
        try:
            draw = ImageDraw.Draw(result)
            draw.text((10, 10), "Original Image", fill=(255, 255, 255))
            draw.text((w + 10, 10), "CAM Overlay", fill=(255, 255, 255))
        except Exception as text_error:
            print(f"Could not add text to image: {text_error}")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(save_path)
            print(f"Saved visualization to {save_path}")
            return True
        else:
            result.show()
            return True

    except Exception as e:
        print(f"Error in visualize_cam: {e}")
        try:
            if save_path:
                cam_uint8 = (cam * 255).astype(np.uint8)
                Image.fromarray(cam_uint8).save(save_path)
                print(f"Saved fallback image to {save_path}")
                return True
        except Exception as e2:
            print(f"Fallback save also failed: {e2}")
        return False

def save_cam(image_path, cam, save_dir, file_suffix='_cam'):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    image_path = Path(image_path)
    save_path = save_dir / f"{image_path.stem}{file_suffix}.png"

    try:
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            h, w = cam.shape
            image = Image.new("RGB", (w, h), color=(0, 0, 0))

        cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size).convert("L")

        cam_array = np.array(cam_resized)
        heat_array = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
        heat_array[..., 0] = cam_array
        heat_array[..., 1] = cam_array // 2
        heatmap = Image.fromarray(heat_array)

        overlay = Image.blend(image, heatmap, alpha=0.5)
        overlay.save(save_path)

        return save_path

    except Exception as e:
        print(f"Failed to save CAM visualization: {e}")
        cam_uint8 = (cam * 255).astype(np.uint8)
        Image.fromarray(cam_uint8).save(save_path)
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

def load_dataset_split():
    """
    加载官方的数据集划分，只有训练集和测试集
    解析每行格式为: "Image_name CLASS-ID SPECIES BREED_ID"
    
    返回:
        dict: 包含train_ids和test_ids的字典，以及对应的标签
    """
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        print(f"Warning: Dataset split files not found at {TRAIN_FILE} or {TEST_FILE}")
        return None
    
    # 加载trainval.txt作为训练集
    train_ids = []
    train_labels = {}
    with open(TRAIN_FILE, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  # 跳过注释和空行
                
            parts = line.strip().split()
            if len(parts) >= 3:
                img_id = parts[0]  # 图像名称
                species = int(parts[2])  # SPECIES: 1=猫, 2=狗
                train_ids.append(img_id)
                train_labels[img_id] = species
    
    # 加载test.txt作为测试集
    test_ids = []
    test_labels = {}
    with open(TEST_FILE, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  # 跳过注释和空行
                
            parts = line.strip().split()
            if len(parts) >= 3:
                img_id = parts[0]  # 图像名称
                species = int(parts[2])  # SPECIES: 1=猫, 2=狗
                test_ids.append(img_id)
                test_labels[img_id] = species
    
    print(f"Dataset split: {len(train_ids)} training images, {len(test_ids)} test images")
    
    return {
        'train_ids': train_ids,
        'test_ids': test_ids,
        'train_labels': train_labels,
        'test_labels': test_labels
    }

def get_dataloaders_from_split(image_dir, batch_size=32, test_ratio=0.2, seed=42, config=None):
    """
    根据官方数据集划分创建数据加载器，从test.txt中选择一部分作为测试集
    
    参数:
        image_dir: 图像目录路径
        batch_size: 批量大小
        test_ratio: 从test.txt中选取的比例作为测试集
        seed: 随机种子，用于控制测试集选择的可重现性
        config: 配置字典，包含优化参数
    
    返回:
        tuple: (train_loader, test_loader)
    """
    # 确保配置有效
    if config is None:
        from config import CLASSIFIER_CONFIG
        config = CLASSIFIER_CONFIG
    
    # 获取数据集划分
    splits = load_dataset_split()
    
    # 提取数据加载优化参数
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)
    prefetch_factor = config.get('prefetch_factor', 2)
    
    # 预处理设置
    transform = get_transform(is_training=True, size=config.get('image_size', (224, 224)))
    eval_transform = get_transform(is_training=False, size=config.get('image_size', (224, 224)))
    
    # 创建图像文件名到文件路径的映射
    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob('*.jpg'))
    image_dict = {path.stem: path for path in image_paths}
    
    # 创建训练集和测试集的图像路径列表
    train_paths = [image_dict[img_id] for img_id in splits['train_ids'] if img_id in image_dict]
    
    # 从test.txt中随机选择test_ratio比例的样本作为测试集
    random.seed(seed)  # 设置随机种子以确保可重现性
    test_ids = splits['test_ids']
    test_size = int(len(test_ids) * test_ratio)
    selected_test_ids = random.sample(test_ids, test_size)
    test_paths = [image_dict[img_id] for img_id in selected_test_ids if img_id in image_dict]
    
    # 创建训练集和测试集
    train_dataset = PetDataset(image_dir, transform=transform, labels_dict=splits['train_labels'])
    train_dataset.image_paths = train_paths
    
    test_dataset = PetDataset(image_dir, transform=eval_transform, is_training=False, labels_dict=splits['test_labels'])
    test_dataset.image_paths = test_paths
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0)
    )
    
    print(f"DataLoaders created: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_loader, test_loader 