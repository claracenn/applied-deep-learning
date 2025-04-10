import os
import glob
import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image

from scorenet import ScoreNet


CAM_DIR = "outputs/cams"
IMAGE_DIR = "data/images"
CONF_DIR = "outputs/confidence_maps"

# 与CAM一致的图像大小（与CAM生成阶段相同）
IMAGE_SIZE = (224, 224)

# 预处理：与cam_extractor中对图像的预处理保持一致
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    """加载原图并做同样的预处理"""
    img = Image.open(image_path).convert('RGB')
    return transform(img)  # [3, H, W]

def generate_confidence_maps(device="cuda"):
    """
    批量生成置信度图:
      1. 遍历CAM文件 (.npy)
      2. 找到对应的原图
      3. 拼接输入ScoreNet
      4. 保存置信度图到 outputs/confidence_maps
    """
    os.makedirs(CONF_DIR, exist_ok=True)
    
    # 初始化ScoreNet
    score_net = ScoreNet(in_channels=4, base_channels=16).to(device)
    score_net.eval()
    
    # 加载ScoreNet模型
    weight_path = "models/scorenet/scorenet.pth"
    score_net.load_state_dict(torch.load(weight_path, map_location=device))
    
    # 遍历CAM文件（.npy）
    cam_files = sorted(glob.glob(os.path.join(CAM_DIR, "*_cam.npy")))
    if not cam_files:
        print(f"No .npy CAM files found in {CAM_DIR}")
        return
    
    print(f"Found {len(cam_files)} CAM files. Generating confidence maps...")
    
    for cam_file in cam_files:
        cam_name = Path(cam_file).stem  
        base_name = cam_name.replace("_cam", "")  
        
        # 在IMAGE_DIR中寻找匹配的图像
        possible_extensions = [".jpg", ".png", ".jpeg"]
        image_path = None
        for ext in possible_extensions:
            test_path = os.path.join(IMAGE_DIR, base_name + ext)
            print(f"Checking for image: {test_path}")
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        if image_path is None:
            print(f"Warning: No matching image found for CAM file {cam_file}")
            continue
        
        # 加载原图和CAM
        image_tensor = load_image(image_path).unsqueeze(0).to(device)  # [1,3,H,W]
        cam_array = np.load(cam_file)  # shape: [H, W] (224x224)
        
        # 确保CAM也在[0,1]范围内 (CAM 可能已经归一化，但可再次clip)
        cam_array = np.clip(cam_array, 0.0, 1.0)
        
        # 转成tensor
        cam_tensor = torch.from_numpy(cam_array).float().unsqueeze(0).unsqueeze(0).to(device)
        # cam_tensor shape: [1,1,H,W]
        
        # 前向推理得到confidence map
        with torch.no_grad():
            conf_map = score_net(image_tensor, cam_tensor)  # [1,1,H,W]
        
        # 转回CPU保存
        conf_map_np = conf_map.squeeze().cpu().numpy()  # [H, W]
        
        # 保存到 outputs/confidence_maps
        conf_name = base_name + "_conf.npy"
        conf_path_npy = os.path.join(CONF_DIR, conf_name)
        np.save(conf_path_npy, conf_map_np)
        
        # 同时保存可视化图像 
        conf_vis_name = base_name + "_conf.png"
        conf_path_png = os.path.join(CONF_DIR, conf_vis_name)
        
        # 将[0,1]映射到[0,255]再可视化
        conf_vis = (conf_map_np * 255).astype(np.uint8)
        cv2.imwrite(conf_path_png, conf_vis)
        
        print(f"Saved confidence map: {conf_path_npy} | {conf_path_png}")
    
    print("All confidence maps generated.")

if __name__ == "__main__":
    generate_confidence_maps()
