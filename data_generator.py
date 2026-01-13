import os
import cv2
import numpy as np
import heapq
import glob
from pathlib import Path

# --- CONFIGURATION ---
PATCH_SIZE = 128
NUM_SAMPLES_PER_IMAGE = 100
OUTPUT_DIR = "dataset_v3"  # New folder
RAW_IMAGES_DIR = r"D:\Thesis\hirise-map-proj-v3\map-proj-v3" # Verify this path!

MOVES = {
    (0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3,
    (1, 1): 4, (-1, 1): 5, (1, -1): 6, (-1, -1): 7
}

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0; self.h = 0; self.f = 0
    def __lt__(self, other): return self.f < other.f

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(grid, start, end):
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, Node(start, None))
    max_iter = 5000
    iter_count = 0

    while open_list:
        iter_count += 1
        if iter_count > max_iter: return None
        current = heapq.heappop(open_list)
        closed_set.add(current.position)

        if current.position == end:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        (r, c) = current.position
        for (dr, dc), _ in MOVES.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < PATCH_SIZE and 0 <= nc < PATCH_SIZE): continue
            if grid[nr][nc] != 0: continue
            if (nr, nc) in closed_set: continue

            new_node = Node((nr, nc), current)
            new_node.g = current.g + np.sqrt(dr**2 + dc**2)
            new_node.h = heuristic((nr, nc), end)
            new_node.f = new_node.g + new_node.h
            heapq.heappush(open_list, new_node)
    return None

def process_patch(patch_img, patch_id, save_dir):
    # 1. Blur and Threshold (Teacher Vision)
    blurred = cv2.GaussianBlur(patch_img, (5, 5), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    _, obstacle_map = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Clean up noise
    kernel = np.ones((3,3), np.uint8)
    obstacle_map = cv2.morphologyEx(obstacle_map, cv2.MORPH_OPEN, kernel)
    grid = (obstacle_map > 0).astype(int)

    # 3. Find Largest Safe Island
    safe_ground_mask = (grid == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(safe_ground_mask)
    if num_labels < 2: return False
    
    label_counts = np.bincount(labels.flatten())
    label_counts[0] = 0 
    largest_island_label = np.argmax(label_counts)
    valid_coords = np.argwhere(labels == largest_island_label)
    
    if len(valid_coords) < 100: return False

    # 4. Spawn Mission
    start = tuple(valid_coords[np.random.choice(len(valid_coords))])
    end = tuple(valid_coords[np.random.choice(len(valid_coords))])
    if np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2) < 40: return False

    # 5. Solve with A*
    path = astar(grid, start, end)
    if path is None: return False

    # --- NEW: VISUAL DEBUGGING ---
    # Create a color version of the obstacle map to draw on
    # Ground will be grey, obstacles will be white
    debug_vis = cv2.cvtColor(obstacle_map, cv2.COLOR_GRAY2BGR)
    
    # Draw the path (Red Line)
    for i in range(len(path) - 1):
        cv2.line(debug_vis, (path[i][1], path[i][0]), (path[i+1][1], path[i+1][0]), (0, 0, 255), 1)
    
    # Draw Start (Green) and Target (Blue)
    cv2.circle(debug_vis, (start[1], start[0]), 3, (0, 255, 0), -1)
    cv2.circle(debug_vis, (end[1], end[0]), 3, (255, 0, 0), -1)

    # Save this to the debug folder
    cv2.imwrite(os.path.join(save_dir, 'debug', f'{patch_id}_path_check.png'), debug_vis)

    # 6. Save Training Data
    save_path_img = os.path.join(save_dir, 'images', f'{patch_id}.png')
    save_path_meta = os.path.join(save_dir, 'meta', f'{patch_id}.npz')
    cv2.imwrite(save_path_img, patch_img)
    np.savez(save_path_meta, path=np.array(path), target=np.array(end), start=np.array(start))
    return True
    # 1. Smooth the image to ignore tiny noise/dust
    blurred = cv2.GaussianBlur(patch_img, (5, 5), 0)
    
    # 2. Normalize
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    
    # 3. Auto-Threshold with Otsu
    # We remove BINARY_INV here to see if your images were "inverted" 
    # (Check your debug folder: Obstacles should be WHITE, Ground should be BLACK)
    _, obstacle_map = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Clean up: Remove tiny "speck" obstacles that are smaller than a rover
    kernel = np.ones((3,3), np.uint8)
    obstacle_map = cv2.morphologyEx(obstacle_map, cv2.MORPH_OPEN, kernel)
    
    grid = (obstacle_map > 0).astype(int)

    # --- THE SMART SPAWN LOGIC ---
    # Find all "islands" of safe ground (0s)
    safe_ground_mask = (grid == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(safe_ground_mask)
    
    if num_labels < 2: # No safe ground at all
        return False

    # Find the largest "island" of safe ground
    label_counts = np.bincount(labels.flatten())
    label_counts[0] = 0  # Ignore the background/obstacles
    largest_island_label = np.argmax(label_counts)
    
    # Get all coordinates in that largest island
    valid_coords = np.argwhere(labels == largest_island_label)
    
    if len(valid_coords) < 100: # Island too small for a mission
        return False

    # Pick Start and Target from the SAME island (Guarantees a path exists!)
    start = tuple(valid_coords[np.random.choice(len(valid_coords))])
    end = tuple(valid_coords[np.random.choice(len(valid_coords))])

    if heuristic(start, end) < 40: return False

    # Solve
    path = astar(grid, start, end)
    
    if path is None:
        return False

    # Save Data
    save_path_img = os.path.join(save_dir, 'images', f'{patch_id}.png')
    save_path_meta = os.path.join(save_dir, 'meta', f'{patch_id}.npz')
    
    # Debug map
    cv2.imwrite(os.path.join(save_dir, 'debug', f'{patch_id}_map.png'), obstacle_map)

    cv2.imwrite(save_path_img, patch_img)
    np.savez(save_path_meta, path=np.array(path), target=np.array(end), start=np.array(start))
    return True
def main():
    # Create directories
    Path(os.path.join(OUTPUT_DIR, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUTPUT_DIR, 'meta')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUTPUT_DIR, 'debug')).mkdir(parents=True, exist_ok=True)

    print(f"Searching in: {RAW_IMAGES_DIR}")
    image_files = []
    for root, _, files in os.walk(RAW_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.tif')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("ERROR: No images found! Check path.")
        return

    print(f"Found {len(image_files)} images. Generating dataset_v3...")
    
    global_counter = 0
    for img_path in image_files:
        full_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if full_img is None: continue
        
        h, w = full_img.shape
        success_count = 0
        
        for r in range(0, h - PATCH_SIZE, PATCH_SIZE):
            for c in range(0, w - PATCH_SIZE, PATCH_SIZE):
                if success_count >= NUM_SAMPLES_PER_IMAGE: break
                
                patch = full_img[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                if np.mean(patch) < 5: continue
                
                if process_patch(patch, global_counter, OUTPUT_DIR):
                    global_counter += 1
                    success_count += 1
                    print(f"Generated {global_counter} samples...", end='\r')

    print(f"\nDone! Check '{OUTPUT_DIR}/debug' to see the obstacle maps.")

if __name__ == "__main__":
    main()