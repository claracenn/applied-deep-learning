import os
import glob
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image

from scorenet import ScoreNet
from config import SCORNET_CONFIG

# Set parameters from configuration
CAM_DIR = SCORNET_CONFIG.get("cam_dir", "outputs/cams")
IMAGE_DIR = SCORNET_CONFIG.get("image_dir", "data/images")
CONF_DIR = SCORNET_CONFIG.get("confidence_dir", "outputs/confidence_maps")
IMAGE_SIZE = SCORNET_CONFIG.get("input_size", (224, 224))

# Define image preprocessing to be consistent with the CAM extractor
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    """
    Load the original image and apply the same preprocessing.
    
    Args:
        image_path (str): Path to the original image.
        
    Returns:
        Tensor: Preprocessed image tensor of shape [3, H, W].
    """
    img = Image.open(image_path).convert('RGB')
    return transform(img)

def generate_confidence_maps(device="cuda"):
    """
    Generate confidence maps in batch:
      1. Iterate over CAM files (.npy).
      2. Find the corresponding original image.
      3. Concatenate inputs and run ScoreNet.
      4. Save the generated confidence maps to the confidence directory.
    
    Args:
        device (str): Device to run inference (e.g., "cuda" or "cpu").
    """
    os.makedirs(CONF_DIR, exist_ok=True)
    
    # Initialize ScoreNet
    score_net = ScoreNet(in_channels=4, base_channels=16).to(device)
    score_net.eval()
    
    # Load the pretrained ScoreNet model weights
    weight_path = SCORNET_CONFIG.get("save_path", "models/scorenet/scorenet.pth")
    score_net.load_state_dict(torch.load(weight_path, map_location=device))
    
    # Iterate over CAM files
    cam_files = sorted(glob.glob(os.path.join(CAM_DIR, "*_cam.npy")))
    if not cam_files:
        print(f"No .npy CAM files found in {CAM_DIR}")
        return
    
    print(f"Found {len(cam_files)} CAM files. Generating confidence maps...")
    
    for cam_file in cam_files:
        cam_name = Path(cam_file).stem  
        base_name = cam_name.replace("_cam", "")  
        
        # Search for the matching image
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
        
        # Load the original image and the corresponding CAM
        image_tensor = load_image(image_path).unsqueeze(0).to(device)  
        cam_array = np.load(cam_file)  
        
        # Ensure CAM values are within [0,1] and convert to tensor
        cam_array = np.clip(cam_array, 0.0, 1.0)
        cam_tensor = torch.from_numpy(cam_array).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Forward pass through ScoreNet 
        with torch.no_grad():
            conf_map = score_net(image_tensor, cam_tensor) 
        
        # Convert the confidence map to a numpy array
        conf_map_np = conf_map.squeeze().cpu().numpy()
        
        # Save the confidence map
        conf_name = base_name + "_conf.npy"
        conf_path_npy = os.path.join(CONF_DIR, conf_name)
        np.save(conf_path_npy, conf_map_np)
        
        # Save visualization images
        conf_vis_name = base_name + "_conf.png"
        conf_path_png = os.path.join(CONF_DIR, conf_vis_name)
        conf_vis = (conf_map_np * 255).astype(np.uint8)
        im = Image.fromarray(conf_vis)
        im.save(conf_path_png)
        
        print(f"Saved confidence map: {conf_path_npy} | {conf_path_png}")
    
    print("All confidence maps generated.")

if __name__ == "__main__":
    generate_confidence_maps()
