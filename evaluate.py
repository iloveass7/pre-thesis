import torch
import cv2
import numpy as np
import os
import glob
import heapq
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

def astar(obstacle_map, start, target):
    """A* pathfinding algorithm."""
    open_list = []
    # (f_cost, position)
    h_cost = np.linalg.norm(np.array(start) - np.array(target))
    heapq.heappush(open_list, (h_cost, start))
    
    came_from = {start: None}
    g_score = {start: 0}

    while open_list:
        _, current_pos = heapq.heappop(open_list)

        if np.linalg.norm(np.array(current_pos) - np.array(target)) < 1.5: # Close enough
            path = []
            while current_pos is not None:
                path.append(current_pos)
                current_pos = came_from.get(current_pos)
            return path[::-1]

        for action_idx, delta in MOVES.items():
            neighbor_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])

            if not (0 <= neighbor_pos[0] < 128 and 0 <= neighbor_pos[1] < 128):
                continue
            if obstacle_map[neighbor_pos[0], neighbor_pos[1]] > 0:
                continue

            move_cost = 1 if action_idx < 4 else np.sqrt(2)
            tentative_g_score = g_score.get(current_pos, float('inf')) + move_cost

            if tentative_g_score < g_score.get(neighbor_pos, float('inf')):
                came_from[neighbor_pos] = current_pos
                g_score[neighbor_pos] = tentative_g_score
                h_cost = np.linalg.norm(np.array(neighbor_pos) - np.array(target))
                f_cost = tentative_g_score + h_cost
                heapq.heappush(open_list, (f_cost, neighbor_pos))

    return None # No path found

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
        
    # Standardize model output, but don't over-amplify it.
    scores = (logits - np.mean(logits)) / (np.std(logits) + 1e-6)
    
    if stuck_mode:
        # Stuck mode: The main goal is to find a new path. We heavily prioritize
        # moving into open space, while maintaining a slight pull to the target.
        dist_weight = 1.0
        safety_weight = 2.5
        oscillation_penalty = 25.0
    else:
        # Normal mode: A robust balance with a strong pull to the target to ensure convergence.
        dist_weight = 3.5
        safety_weight = 0.6
        oscillation_penalty = 15.0

    current_dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
    
    for action_idx, delta in MOVES.items():
        next_p = (current_pos[0] + delta[0], current_pos[1] + delta[1])
        
        # 1. Hard Filters
        if not (0 <= next_p[0] < 128 and 0 <= next_p[1] < 128):
            scores[action_idx] = -1000; continue
        if obstacle_map[next_p[0], next_p[1]] > 0:
            scores[action_idx] = -1000; continue
            
        # 2. Shortest Path Cost (Progress towards target)
        next_dist = np.linalg.norm(np.array(next_p) - np.array(target_pos))
        delta_dist = current_dist - next_dist
        scores[action_idx] += delta_dist * dist_weight
        
        # 3. Easy Path Cost (Obstacle avoidance)
        # A moderate reward for staying in open areas.
        safety = dist_map[next_p[0], next_p[1]]
        scores[action_idx] += min(safety, 8.0) * safety_weight
            
        # 4. Anti-Oscillation Penalty
        if history is not None:
            visits = history[-50:].count(next_p)
            if visits > 0:
                scores[action_idx] -= visits * oscillation_penalty

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
    
    total_path_optimality = 0
    total_successful_sims = 0

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

        # --- A* Pathfinding ---
        astar_path = astar(obs_map, start, target)
        if astar_path is None:
            print(f"Sim {i:2d}: SKIPPED - A* could not find a path.")
            continue
            
        current = start; path = [current]
        dist_direct = np.linalg.norm(np.array(start)-np.array(target))
        
        color_map = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        cv2.circle(color_map, (start[1], start[0]), 3, (0, 255, 0), -1) # Start: Green
        cv2.circle(color_map, (target[1], target[0]), 3, (255, 0, 0), -1) # Target: Blue

        success = False; collision = False; stuck_mode = False
        
        for step in range(1000):
            # Dynamic Stuck Detection: If we haven't made at least 2px of progress
            # towards the target in the last 15 steps, activate stuck mode to explore.
            if len(path) > 15:
                progress = np.linalg.norm(np.array(path[-15]) - np.array(target)) - np.linalg.norm(np.array(current) - np.array(target))
                stuck_mode = progress < 2.0 
            
            action = predict_move(model, raw_img, current, target, obs_map, dist_map, path, stuck_mode)
            delta = MOVES[action]
            next_pos = (current[0] + delta[0], current[1] + delta[1])
            
            if not (0 <= next_pos[0] < 128 and 0 <= next_pos[1] < 128): break
            if obs_map[next_pos[0], next_pos[1]] > 0:
                collision = True; break
                
            current = next_pos; path.append(current)
            if np.linalg.norm(np.array(current) - np.array(target)) < 5:
                success = True; break

        # Drawing paths
        # A* path (Ground Truth): Yellow
        if astar_path:
            for j in range(len(astar_path) - 1):
                cv2.line(color_map, (astar_path[j][1], astar_path[j][0]), (astar_path[j+1][1], astar_path[j+1][0]), (0, 255, 255), 1)
        # Model's path: Red
        for j in range(len(path) - 1):
            cv2.line(color_map, (path[j][1], path[j][0]), (path[j+1][1], path[j+1][0]), (0, 0, 255), 1)

        # Calculate Metrics
        eff = dist_direct / len(path) if len(path) > 0 else 0
        path_optimality = len(astar_path) / len(path) if success and len(path) > 0 and astar_path is not None else 0.0
        if success:
            total_path_optimality += path_optimality
            total_successful_sims += 1

        status = "SUCCESS" if success else ("COLLISION" if collision else "TIMEOUT")
        save_path = os.path.join(OUTPUT_DIR, f"result_{i}_{status}.png")
        cv2.imwrite(save_path, color_map)
        
        astar_steps = len(astar_path) if astar_path is not None else 0
        print(f"Sim {i:2d}: {status:9s} | Eff: {eff:4.2f} | Steps: {len(path):4d} | A* Steps: {astar_steps:4d} | Optimality: {path_optimality:4.2f} {'[STUCK_REC]' if stuck_mode else ''}")

    if total_successful_sims > 0:
        avg_optimality = total_path_optimality / total_successful_sims
        print("\n--- Evaluation Summary ---")
        print(f"Average Path Optimality (vs A*): {avg_optimality:.3f} ({total_successful_sims} successful sims)")
        print("--------------------------")

if __name__ == "__main__":
    run_simulation()
