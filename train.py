import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import DBCNN

# --- CONFIGURATION ---
DATASET_DIR = "dataset_v3"
BATCH_SIZE = 64  # Increased batch size for stable gradients
LEARNING_RATE = 0.001
EPOCHS = 100      # Fewer epochs needed with augmentation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move mapping
MOVES_TO_LABEL = {
    (0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3,
    (1, 1): 4, (-1, 1): 5, (1, -1): 6, (-1, -1): 7
}

class MarsRoverDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.meta_files = glob.glob(os.path.join(root_dir, "meta", "*.npz"))
        if len(self.meta_files) == 0:
            raise RuntimeError(f"No data found in {root_dir}")

    def __len__(self):
        return len(self.meta_files)

    def draw_blob(self, canvas, r, c):
        # Draws a 5x5 square blob so the CNN can see the target easily
        h, w = canvas.shape
        r_min, r_max = max(0, r-2), min(h, r+3)
        c_min, c_max = max(0, c-2), min(w, c+3)
        canvas[r_min:r_max, c_min:c_max] = 1.0

    def rotate_state(self, image, target, current, move_delta):
        # Randomly rotate 0, 90, 180, or 270 degrees
        k = np.random.randint(0, 4)
        if k == 0: return image, target, current, move_delta
        
        # Rotate maps (np.rot90 rotates Counter-Clockwise)
        image = np.rot90(image, k)
        target = np.rot90(target, k)
        current = np.rot90(current, k)
        
        # Rotate the move vector (delta) to match Counter-Clockwise rotation
        dr, dc = move_delta
        for _ in range(k):
            # Counter-Clockwise Formula: (r, c) -> (-c, r)
            # Old Clockwise Formula was: (dc, -dr) <-- THIS WAS THE BUG
            dr, dc = -dc, dr
            
        return image, target, current, (dr, dc)
        # Randomly rotate 0, 90, 180, or 270 degrees
        k = np.random.randint(0, 4)
        if k == 0: return image, target, current, move_delta
        
        # Rotate maps
        image = np.rot90(image, k)
        target = np.rot90(target, k)
        current = np.rot90(current, k)
        
        # Rotate the move vector (delta)
        dr, dc = move_delta
        for _ in range(k):
            # Rotate vector 90 degrees clockwise: (r, c) -> (c, -r)
            # But in image coords (row, col), 90 deg rotation is (col, H-1-row).
            # For a relative vector: (dr, dc) -> (dc, -dr)
            dr, dc = dc, -dr
            
        return image, target, current, (dr, dc)

    def __getitem__(self, idx):
        try:
            meta_path = self.meta_files[idx]
            file_id = os.path.basename(meta_path).replace(".npz", "")
            data = np.load(meta_path)
            path = data['path']
            
            # Load Image
            img_path = os.path.join(self.root_dir, "images", f"{file_id}.png")
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: raise ValueError("Image not found")

            # Pick random step
            if len(path) < 2: return self.__getitem__((idx + 1) % len(self))
            t = np.random.randint(0, len(path) - 1)
            curr_pos = path[t]
            next_pos = path[t+1]

            # Prepare Channels
            c0 = image.astype(np.float32) / 255.0
            
            # Draw BIG blobs for Target and Rover
            c1 = np.zeros_like(c0); self.draw_blob(c1, data['target'][0], data['target'][1])
            c2 = np.zeros_like(c0); self.draw_blob(c2, curr_pos[0], curr_pos[1])

            # Calculate Move Delta
            delta = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])

            # Apply Rotation Augmentation (Crucial for fixing bias)
            c0, c1, c2, delta = self.rotate_state(c0, c1, c2, delta)

            # Get Label
            label = MOVES_TO_LABEL.get(delta, 0)

            # Stack
            state = np.stack([c0, c1, c2], axis=0)
            return torch.tensor(state, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

def train():
    print(f"Training on device: {DEVICE} with AUGMENTATION")
    
    dataset = MarsRoverDataset(DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} paths.")

    model = DBCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (states, labels) in enumerate(dataloader):
            states, labels = states.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f} | Acc: {accuracy:.2f}%")
        
        # Save every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), "rover_model_latest.pth")

    print("Training Complete.")

if __name__ == "__main__":
    train()