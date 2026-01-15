import torch
import cv2
import numpy as np
import os
import glob
from model import DBCNN

# --- CONFIG ---
MODEL_PATH = "rover_model_latest.pth"
DATA_DIR = "dataset_v4/images"
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

def get_obstacle_map(image):
    # Precise multi-feature detection for 'Shortest' and 'Easiest' planning
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 1. Canny for sharp hazards
    edges = cv2.Canny(denoised, 30, 90)
    
    # 2. Local thresholding with a larger block size to reduce noise in flat areas
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 5)
    
    combined = cv2.bitwise_or(thresh, edges)
    
    # Cleaning
    kernel = np.ones((2,2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Thin buffer (Matches rover size better for shorter paths)
    dilate_kernel = np.ones((2,2), np.uint8)
    combined = cv2.dilate(combined, dilate_kernel, iterations=1)
    
    return combined

def predict_move(model, image, current_pos, target_pos, obstacle_map, dist_map, history=None, stuck_mode=False):
    # Channel Preparation
    c0 = image.astype(np.float32) / 255.0
    c1 = np.zeros_like(c0); draw_blob(c1, target_pos[0], target_pos[1])
    c2 = np.zeros_like(c0); draw_blob(c2, current_pos[0], current_pos[1])
    
    state = np.stack([c0, c1, c2], axis=0)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(state_tensor)
        logits = output.squeeze(0).cpu().numpy()
        
    # Standardize Model Guidance
    logits = (logits - np.mean(logits)) / (np.std(logits) + 1e-6)
    
    scores = logits.copy()
    current_dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
    
    for action_idx, delta in MOVES.items():
        next_p = (current_pos[0] + delta[0], current_pos[1] + delta[1])
        
        # 1. Collision & Boundary Hard Filter
        if not (0 <= next_p[0] < 128 and 0 <= next_p[1] < 128):
            scores[action_idx] = -1000; continue
        if obstacle_map[next_p[0], next_p[1]] > 0:
            scores[action_idx] = -1000; continue
            
        # 2. Shortest Path Cost (Prefer closing distance)
        next_dist = np.linalg.norm(np.array(next_p) - np.array(target_pos))
        delta_dist = current_dist - next_dist
        # High weight on progress to target
        scores[action_idx] += delta_dist * 5.0 
        
        # 3. Easy Path Cost (Safety/Smoothness)
        # Stay clear of obstacles. Max reward is ~3.0 for being 10px+ away.
        safety = dist_map[next_p[0], next_p[1]]
        scores[action_idx] += min(safety, 5.0) * 0.4 
        
        # 4. Diagonal Cost (Prefer straight lines if distance gain is same)
        is_diagonal = abs(delta[0]) + abs(delta[1]) == 2
        if is_diagonal:
            scores[action_idx] -= 0.1
            
        # 5. Anti-Oscillation Penalty
        if history is not None:
            # Check last 30 steps for loops
            visits = history[-30:].count(next_p)
            if visits > 0:
                scores[action_idx] -= visits * 10.0
        
        # 6. Stuck Mode: Extra push to target
        if stuck_mode:
            scores[action_idx] += delta_dist * 15.0

    return np.argmax(scores)

def run_simulation():
    if not os.path.exists(MODEL_PATH):
        print("Error: Train the model first!"); return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = DBCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    image_files = glob.glob(os.path.join(DATA_DIR, "*.png"))
    if not image_files: return

    print(f"Goal: 'Shortest & Easiest' Path via Efficiency-Weighted Planning...")

    for i in range(min(25, len(image_files))):
        img_path = image_files[i]
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None: continue
        
        obs_map = get_obstacle_map(raw_img)
        dist_map = cv2.distanceTransform(255 - obs_map, cv2.DIST_L2, 5)
        
        # Mission Spawning
        safe_ground = (obs_map == 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(safe_ground)
        if num_labels < 2: continue
        label_counts = np.bincount(labels.flatten()); label_counts[0] = 0 
        largest_island = np.argmax(label_counts)
        valid_coords = np.argwhere(labels == largest_island)
        if len(valid_coords) < 100: continue
        
        start = tuple(valid_coords[np.random.choice(len(valid_coords))])
        target = tuple(valid_coords[np.random.choice(len(valid_coords))])
        for _ in range(100):
            if np.linalg.norm(np.array(start)-np.array(target)) > 80: break
            target = tuple(valid_coords[np.random.choice(len(valid_coords))])
            
        current = start; path = [current]
        dist_direct = np.linalg.norm(np.array(start)-np.array(target))
        
        color_map = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        # We no longer tint the results; drawing directly on the gray image as requested.
        cv2.circle(color_map, (start[1], start[0]), 3, (0, 255, 0), -1)
        cv2.circle(color_map, (target[1], target[0]), 3, (255, 0, 0), -1)

        success = False; collision = False; stuck_mode = False
        
        for step in range(1000):
            # Dynamic Stuck Detection
            if len(path) > 10:
                prog = np.linalg.norm(np.array(path[-10])-np.array(target)) - np.linalg.norm(np.array(current)-np.array(target))
                stuck_mode = prog < 1.0 
            
            action = predict_move(model, raw_img, current, target, obs_map, dist_map, path, stuck_mode)
            delta = MOVES[action]
            next_pos = (current[0] + delta[0], current[1] + delta[1])
            
            if not (0 <= next_pos[0] < 128 and 0 <= next_pos[1] < 128): break
            if obs_map[next_pos[0], next_pos[1]] > 0:
                collision = True; break
                
            current = next_pos; path.append(current)
            if np.linalg.norm(np.array(current) - np.array(target)) < 5:
                success = True; break

        # Drawing the path
        for j in range(len(path) - 1):
            cv2.line(color_map, (path[j][1], path[j][0]), (path[j+1][1], path[j+1][0]), (0, 0, 255), 1)

        # Calculate Efficiency (Shortest Path Efficiency)
        eff = dist_direct / len(path) if len(path) > 0 else 0
        status = "SUCCESS" if success else ("COLLISION" if collision else "TIMEOUT")
        save_path = os.path.join(OUTPUT_DIR, f"result_{i}_{status}.png")
        cv2.imwrite(save_path, color_map)
        print(f"Sim {i:2d}: {status:9s} | Eff: {eff:4.2f} | Dist: {dist_direct:5.1f} | Steps: {len(path):4d} {'[STUCK_REC]' if stuck_mode else ''}")

if __name__ == "__main__":
    run_simulation()