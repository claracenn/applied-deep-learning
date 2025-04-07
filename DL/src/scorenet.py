import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

from config import SCORNET_CONFIG


# ScoreNet Model Definition
class ScoreNet(nn.Module):
    """
    ScoreNet generates pixel-level confidence maps.
    
    Inputs: 
        image: [B, 3, H, W] normalized original image.
        cam:   [B, 1, H, W] CAM obtained from the CAM extractor.
        
    Output:
        confidence_map: [B, 1, H, W] with values in the range [0,1].
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
        # Concatenate image and CAM to form a 4-channel input.
        x = torch.cat([image, cam], dim=1) 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return torch.sigmoid(x)


# Dataset Definition
class ScoreNetDataset(Dataset):
    def __init__(self, image_dir, cam_dir, transform=None):
        """
        image_dir: Directory of original images (e.g., "data/images").
        cam_dir: Directory where CAM files are stored (e.g., "outputs/cams").
        transform: Preprocessing for images (recommended: resize to ScoreNet input size).
        """
        self.image_dir = image_dir
        self.cam_dir = cam_dir
        self.transform = transform

        # List all image files in image_dir 
        self.image_paths = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self.image_paths = sorted(self.image_paths)

        # Build a dictionary for CAM files
        self.cam_files = {}
        for file in glob.glob(os.path.join(cam_dir, "*.npy")):
            stem = os.path.splitext(os.path.basename(file))[0]
            stem_lower = stem.lower()
            if stem_lower.endswith("_cam"):
                key = stem_lower[:-4]  
            else:
                key = stem_lower
            self.cam_files[key] = file

        # Filter out images that do not have a corresponding CAM file
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
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        # Determine corresponding CAM file case-insensitive
        base_name = os.path.splitext(os.path.basename(image_path))[0].lower()
        cam_file = self.cam_files[base_name]
        cam_array = np.load(cam_file)
        cam_tensor = torch.from_numpy(cam_array).float().unsqueeze(0)
        # Generate binary pseudo-target from cam > 0.5
        target_conf = (cam_tensor > 0.5).float()
        return {"image": image_tensor, "cam": cam_tensor, "target_conf": target_conf}


# Evaluation Function for ScoreNet: Computes BCE loss and pixel accuracy
def evaluate_scorenet(model, dataloader, device):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)    
            cams = batch["cam"].to(device)         
            targets = batch["target_conf"].to(device) 
            outputs = model(images, cams)         
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            correct_pixels += (preds == targets).sum().item()
            total_pixels += targets.numel()
    avg_loss = total_loss / len(dataloader.dataset)
    pixel_accuracy = correct_pixels / total_pixels * 100
    return avg_loss, pixel_accuracy


# Train ScoreNet Function with Validation, Best-Model Saving, and Early Stopping
def train_scorenet(train_loader, val_loader, config):
    device = torch.device(config["device"])
    model = ScoreNet(in_channels=4, base_channels=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    epochs = config["epochs"]
    patience = config["patience"]
    save_path = config["save_path"]
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            cams = batch["cam"].to(device)
            targets = batch["target_conf"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, cams)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        val_loss, val_accuracy = evaluate_scorenet(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}] Validation Loss: {val_loss:.4f}, Pixel Accuracy: {val_accuracy:.2f}%")
        
        # Check for improvement and apply early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print(f"Validation loss improved, saving model to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return model


if __name__ == "__main__":
    # Load config
    config = SCORNET_CONFIG
    device = config["device"]
    input_size = config["input_size"]
    
    # Define image preprocessing: resize to input_size and normalize
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and split it into training and validation sets
    dataset = ScoreNetDataset(image_dir=config["image_dir"], cam_dir=config["cam_dir"], transform=transform)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    
    # Train ScoreNet with validation and early stopping
    model = train_scorenet(train_loader, val_loader, config)
    
    # Final evaluation on the validation set
    final_loss, final_accuracy = evaluate_scorenet(model, val_loader, device)
    print(f"Final Validation Loss: {final_loss:.4f}, Pixel Accuracy: {final_accuracy:.2f}%")
