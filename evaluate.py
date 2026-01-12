import torch
import cv2
import numpy as np
import os
import glob
from model import DBCNN

# --- CONFIG ---
MODEL_PATH = "rover_model_latest.pth"
DATA_DIR = "dataset_v1/images"
OUTPUT_DIR = "results_visualized"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move mapping
MOVES = {
    0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0),
    4: (1, 1), 5: (-1, 1), 6: (1, -1), 7: (-1, -1)
}

def draw_blob(canvas, r, c):
    # This must MATCH the training script exactly!
    h, w = canvas.shape
    r_min, r_max = max(0, r-2), min(h, r+3)
    c_min, c_max = max(0, c-2), min(w, c+3)
    canvas[r_min:r_max, c_min:c_max] = 1.0

def predict_move(model, image, current_pos, target_pos):
    # Prepare Input State (3 Channels)
    c0 = image.astype(np.float32) / 255.0
    
    # Channel 1: Target (Now with Blob!)
    c1 = np.zeros_like(c0)
    draw_blob(c1, target_pos[0], target_pos[1])
    
    # Channel 2: Rover (Now with Blob!)
    c2 = np.zeros_like(c0)
    draw_blob(c2, current_pos[0], current_pos[1])
    
    state = np.stack([c0, c1, c2], axis=0)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(state_tensor)
        # We can also print probabilities to debug
        probs = torch.softmax(output, dim=1)
        action_idx = torch.argmax(output, dim=1).item()
        
    return action_idx

def run_simulation():
    if not os.path.exists(MODEL_PATH):
        print("Error: Train the model first!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = DBCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    image_files = glob.glob(os.path.join(DATA_DIR, "*.png"))
    if not image_files:
        print("No images found.")
        return

    print(f"Testing on 5 maps...")

    for i in range(min(5, len(image_files))):
        img_path = image_files[i]
        
        # Load Map
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        color_map = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        
        # Force a Diagonal Scenario (Top-Left to Bottom-Right)
        # This tests if it can actually "aim" or just goes East
        start = (20, 20) 
        target = (100, 100)
        
        current = start
        path = [current]
        
        # Draw Start (Green) and Target (Blue)
        cv2.circle(color_map, (start[1], start[0]), 3, (0, 255, 0), -1)
        cv2.circle(color_map, (target[1], target[0]), 5, (255, 0, 0), -1)

        success = False
        for step in range(128): # Max steps
            action = predict_move(model, raw_img, current, target)
            delta = MOVES[action]
            
            # Update
            next_pos = (current[0] + delta[0], current[1] + delta[1])
            
            # Wall Check
            if not (0 <= next_pos[0] < 128 and 0 <= next_pos[1] < 128):
                break
                
            current = next_pos
            path.append(current)
            
            # Target Check (Distance < 5)
            if np.sqrt((current[0]-target[0])**2 + (current[1]-target[1])**2) < 5:
                success = True
                break

        # Draw Path
        for j in range(len(path) - 1):
            pt1 = (path[j][1], path[j][0])
            pt2 = (path[j+1][1], path[j+1][0])
            cv2.line(color_map, pt1, pt2, (0, 0, 255), 1)

        status = "SUCCESS" if success else "FAIL"
        save_path = os.path.join(OUTPUT_DIR, f"result_{i}_{status}.png")
        cv2.imwrite(save_path, color_map)
        print(f"Simulation {i}: {status}")

if __name__ == "__main__":
    run_simulation()