# data_generator_curved.py
import os
import cv2
import numpy as np
import heapq
import glob
from pathlib import Path

# --- CONFIGURATION ---
PATCH_SIZE = 128
NUM_SAMPLES_PER_IMAGE = 50  # Reduced for quality
OUTPUT_DIR = "dataset_curved"  # New folder for curved paths
RAW_IMAGES_DIR = r"C:\Users\Mahadir\pre-thesis\dataset_v3\images"

class CurvedPathGenerator:
    def __init__(self, patch_size=128):
        self.patch_size = patch_size
        self.moves = {
            (0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3,
            (1, 1): 4, (-1, 1): 5, (1, -1): 6, (-1, -1): 7
        }
    
    def generate_curved_astar_path(self, grid, start, end, randomness=0.3):
        """Generate A* path with some curvature/randomness"""
        open_list = []
        closed_set = set()
        heapq.heappush(open_list, (0, start, None))
        came_from = {}
        g_score = {start: 0}
        
        max_iter = 2000
        iter_count = 0
        
        while open_list:
            iter_count += 1
            if iter_count > max_iter:
                break
                
            _, current, parent = heapq.heappop(open_list)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            came_from[current] = parent
            
            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            # Add some randomness to the heuristic
            for (dr, dc), _ in self.moves.items():
                nr, nc = current[0] + dr, current[1] + dc
                
                if not (0 <= nr < self.patch_size and 0 <= nc < self.patch_size):
                    continue
                    
                if grid[nr, nc] != 0:
                    continue
                    
                new_pos = (nr, nc)
                
                # Add small random cost to encourage non-straight paths
                random_cost = np.random.uniform(0, randomness)
                tentative_g = g_score[current] + np.sqrt(dr**2 + dc**2) + random_cost
                
                if new_pos not in g_score or tentative_g < g_score[new_pos]:
                    g_score[new_pos] = tentative_g
                    
                    # Heuristic with small random component
                    h = np.sqrt((end[0] - nr)**2 + (end[1] - nc)**2) * (1 + np.random.uniform(-0.1, 0.1))
                    f = tentative_g + h
                    
                    heapq.heappush(open_list, (f, new_pos, current))
        
        return None
    
    def smooth_path(self, path, smoothness=0.5):
        """Apply simple path smoothing"""
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]
        for i in range(1, len(path)-1):
            prev = np.array(path[i-1])
            curr = np.array(path[i])
            next_pos = np.array(path[i+1])
            
            # Weighted average
            smoothed_point = (prev + curr + next_pos) / 3.0
            smoothed_point = tuple(np.round(smoothed_point).astype(int))
            smoothed.append(smoothed_point)
            
        smoothed.append(path[-1])
        return smoothed

def process_patch_curved(patch_img, patch_id, save_dir, generator):
    # 1. Create obstacle map
    blurred = cv2.GaussianBlur(patch_img, (5, 5), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    _, obstacle_map = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    obstacle_map = cv2.morphologyEx(obstacle_map, cv2.MORPH_OPEN, kernel)
    grid = (obstacle_map > 0).astype(int)
    
    # 2. Find safe areas
    safe_ground_mask = (grid == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(safe_ground_mask)
    
    if num_labels < 2:
        return False
        
    label_counts = np.bincount(labels.flatten())
    label_counts[0] = 0
    largest_island_label = np.argmax(label_counts)
    valid_coords = np.argwhere(labels == largest_island_label)
    
    if len(valid_coords) < 100:
        return False
    
    # 3. Generate start and end
    start_idx = np.random.choice(len(valid_coords))
    end_idx = np.random.choice(len(valid_coords))
    start = tuple(valid_coords[start_idx])
    end = tuple(valid_coords[end_idx])
    
    if np.linalg.norm(np.array(start) - np.array(end)) < 40:
        return False
    
    # 4. Generate curved path
    path = generator.generate_curved_astar_path(grid, start, end, randomness=0.2)
    
    if path is None or len(path) < 10:
        return False
    
    # 5. Smooth the path
    smoothed_path = generator.smooth_path(path, smoothness=0.3)
    
    # 6. Visualize and save
    debug_vis = cv2.cvtColor(patch_img, cv2.COLOR_GRAY2BGR)
    
    # Draw path
    for i in range(len(smoothed_path) - 1):
        cv2.line(debug_vis, 
                (smoothed_path[i][1], smoothed_path[i][0]),
                (smoothed_path[i+1][1], smoothed_path[i+1][0]),
                (0, 0, 255), 1)
    
    # Draw start and end
    cv2.circle(debug_vis, (start[1], start[0]), 3, (0, 255, 0), -1)
    cv2.circle(debug_vis, (end[1], end[0]), 3, (255, 0, 0), -1)
    
    cv2.imwrite(os.path.join(save_dir, 'debug', f'{patch_id}_curved.png'), debug_vis)
    
    # 7. Save data
    save_path_img = os.path.join(save_dir, 'images', f'{patch_id}.png')
    save_path_meta = os.path.join(save_dir, 'meta', f'{patch_id}.npz')
    
    cv2.imwrite(save_path_img, patch_img)
    np.savez(save_path_meta, 
             path=np.array(smoothed_path), 
             target=np.array(end), 
             start=np.array(start))
    
    return True

def main_curved():
    generator = CurvedPathGenerator(PATCH_SIZE)
    
    Path(os.path.join(OUTPUT_DIR, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUTPUT_DIR, 'meta')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUTPUT_DIR, 'debug')).mkdir(parents=True, exist_ok=True)
    
    image_files = []
    for root, _, files in os.walk(RAW_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.tif')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print("ERROR: No images found!")
        return
    
    print(f"Found {len(image_files)} images. Generating curved dataset...")
    
    global_counter = 0
    for img_path in image_files:
        full_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if full_img is None:
            continue
            
        h, w = full_img.shape
        success_count = 0
        
        for r in range(0, h - PATCH_SIZE, PATCH_SIZE):
            for c in range(0, w - PATCH_SIZE, PATCH_SIZE):
                if success_count >= NUM_SAMPLES_PER_IMAGE:
                    break
                    
                patch = full_img[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                if np.mean(patch) < 5:
                    continue
                    
                if process_patch_curved(patch, global_counter, OUTPUT_DIR, generator):
                    global_counter += 1
                    success_count += 1
                    print(f"Generated {global_counter} curved samples...", end='\r')
    
    print(f"\nDone! Generated {global_counter} curved paths.")

if __name__ == "__main__":
    main_curved()