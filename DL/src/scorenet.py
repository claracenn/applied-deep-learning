import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class ScoreNet(nn.Module):
    """
    ScoreNet用于生成像素级置信度图。
    
    输入: 
        image: [B, 3, H, W] 原始图像（归一化后）
        cam:   [B, 1, H, W] 由CAM提取器获得的类激活图
        
    输出:
        confidence_map: [B, 1, H, W] 像素级置信度图，值在 [0,1] 之间
    """
    def __init__(self, in_channels=4, base_channels=16):
        super(ScoreNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 2)
        
        self.conv4 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_channels)
        
        self.conv5 = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def forward(self, image, cam):
        x = torch.cat([image, cam], dim=1)  # 拼接成4通道输入
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return torch.sigmoid(x)

class ScoreNetDataset(Dataset):
    def __init__(self, image_dir, cam_dir, transform=None):
        """
        image_dir: 原始图像所在目录，如 "data/images"
        cam_dir: CAM文件存放目录，如 "outputs/cams"
        transform: 图像预处理（建议 resize 到 ScoreNet 需要的尺寸，例如 (224,224)）
        """
        self.image_dir = image_dir
        self.cam_dir = cam_dir
        self.transform = transform

        # 列出 image_dir 下所有图像文件（支持 jpg/png/jpeg）
        self.image_paths = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self.image_paths = sorted(self.image_paths)

        # 构建一个 CAM 文件字典，key 为小写的图像基本名（不含 _cam），value 为完整的 CAM 文件路径
        self.cam_files = {}
        for file in glob.glob(os.path.join(cam_dir, "*.npy")):
            stem = os.path.splitext(os.path.basename(file))[0]
            stem_lower = stem.lower()
            if stem_lower.endswith("_cam"):
                key = stem_lower[:-4]  # 去掉 "_cam"
            else:
                key = stem_lower
            self.cam_files[key] = file

        # 过滤掉那些没有对应 CAM 文件的图像
        valid_images = []
        for image_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0].lower()
            if base_name in self.cam_files:
                valid_images.append(image_path)
            else:
                print(f"Warning: No CAM file found for {image_path}, skipping.")
        self.image_paths = valid_images

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        # 根据图像文件名推断对应的 CAM 文件（大小写不敏感）
        base_name = os.path.splitext(os.path.basename(image_path))[0].lower()
        cam_file = self.cam_files[base_name]
        cam_array = np.load(cam_file)
        cam_tensor = torch.from_numpy(cam_array).float().unsqueeze(0)
        # 使用 cam > 0.5 得到二值伪监督目标
        target_conf = (cam_tensor > 0.5).float()
        return {"image": image_tensor, "cam": cam_tensor, "target_conf": target_conf}



def train_scorenet(train_loader, epochs=10, device="cuda", lr=1e-3, save_path="models/scorenet/scorenet.pth"):
    device = torch.device(device)
    model = ScoreNet(in_channels=4, base_channels=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)    # [B, 3, H, W]
            cams = batch["cam"].to(device)          # [B, 1, H, W]
            targets = batch["target_conf"].to(device) # [B, 1, H, W]
            
            optimizer.zero_grad()
            outputs = model(images, cams)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"ScoreNet model saved to {save_path}")


if __name__ == "__main__":
    # 参数设置
    EPOCHS = 10
    LR = 0.001
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = "models/scorenet/scorenet.pth"
    
    # 数据路径设置（根据 README，图像在 data/images，CAM 在 outputs/cams）
    IMAGE_DIR = "data/images"
    CAM_DIR = "outputs/cams"
    
    # 图像预处理（假设统一调整到 224x224，并归一化）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    dataset = ScoreNetDataset(image_dir=IMAGE_DIR, cam_dir=CAM_DIR, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 训练 ScoreNet
    train_scorenet(train_loader, epochs=EPOCHS, device=DEVICE, lr=LR, save_path=SAVE_PATH)
