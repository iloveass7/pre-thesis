# train_path_predictor.py
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import PathPredictorDB_CNN

class PathPredictionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.meta_files = glob.glob(os.path.join(root_dir, "meta", "*.npz"))
        
    def __len__(self):
        return len(self.meta_files)
    
    def draw_blob(self, canvas, r, c, size=5):
        h, w = canvas.shape
        r_min, r_max = max(0, r-size), min(h, r+size+1)
        c_min, c_max = max(0, c-size), min(w, c+size+1)
        canvas[r_min:r_max, c_min:c_max] = 1.0
    
    def create_value_map(self, path, target, size=128):
        """Create value map where path positions have high values"""
        value_map = np.zeros((size, size), dtype=np.float32)
        
        # Target has highest value
        self.draw_blob(value_map, target[0], target[1], size=3)
        value_map[target[0], target[1]] = 1.0
        
        # Path positions have decreasing values based on distance to target
        for i, (r, c) in enumerate(path):
            distance = len(path) - i
            value = 0.8 * (distance / len(path))
            value_map[r, c] = max(value_map[r, c], value)
        
        # Smooth the value map
        value_map = cv2.GaussianBlur(value_map, (5, 5), 1.0)
        
        return value_map
    
    def __getitem__(self, idx):
        meta_path = self.meta_files[idx]
        file_id = os.path.basename(meta_path).replace(".npz", "")
        data = np.load(meta_path)
        
        path = data['path']
        target = data['target']
        start = data['start']
        
        # Load image
        img_path = os.path.join(self.root_dir, "images", f"{file_id}.png")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # Create channels
        c0 = image  # Map
        
        c1 = np.zeros_like(image)  # Target
        self.draw_blob(c1, target[0], target[1])
        
        c2 = np.zeros_like(image)  # Start
        self.draw_blob(c2, start[0], start[1])
        
        # Create value map target
        value_map = self.create_value_map(path, target)
        
        # Stack input
        input_tensor = np.stack([c0, c1, c2], axis=0)
        
        return (torch.tensor(input_tensor, dtype=torch.float32),
                torch.tensor(value_map, dtype=torch.float32).unsqueeze(0))

def train_path_predictor():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")
    
    # Use your curved dataset
    DATASET_DIR = "dataset_curved"
    dataset = PathPredictionDataset(DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    model = PathPredictorDB_CNN().to(DEVICE)
    criterion = nn.MSELoss()  # Mean squared error for value map
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        total_loss = 0
        model.train()
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch {epoch}, Batch {i}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"path_predictor_epoch_{epoch+1}.pth")
    
    print("Training complete!")

if __name__ == "__main__":
    train_path_predictor()