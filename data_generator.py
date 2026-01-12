import os
import cv2
import numpy as np
import heapq
import glob
from pathlib import Path

# --- CONFIGURATION ---
PATCH_SIZE = 128
NUM_SAMPLES_PER_IMAGE = 500
OUTPUT_DIR = "dataset_v2" # New folder for better data
RAW_IMAGES_DIR = r"D:\Thesis\hirise-map-proj-v3\map-proj-v3" # Check your path

# Move mapping
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
    max_iter = 10000 # Increased for complex mazes
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
    # 1. CONTRAST ENHANCEMENT (Crucial for Mars)
    # This makes faint craters distinct before edge detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(patch_img)
    
    # 2. SENSITIVE EDGE DETECTION
    # Lower thresholds to pick up soft crater rims
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Dilate to make obstacles "scary" (thick)
    kernel = np.ones((3,3), np.uint8)
    obstacle_map = cv2.dilate(edges, kernel, iterations=2)
    grid = (obstacle_map > 0).astype(int)

    free_spaces = np.argwhere(grid == 0)
    if len(free_spaces) < 100: return False # Too crowded

    start = tuple(free_spaces[np.random.choice(len(free_spaces))])
    end = tuple(free_spaces[np.random.choice(len(free_spaces))])

    # Ensure start/end are far apart
    dist = heuristic(start, end)
    if dist < 40: return False

    path = astar(grid, start, end)
    if path is None: return False
    
    # 3. THE "BORING PATH" FILTER
    # A straight line length is equal to distance.
    # If path_length is almost equal to distance, the path was a straight line.
    # We only save paths that are at least 10% longer than the straight line
    # (meaning they had to curve around something).
    path_len = len(path)
    if path_len < (dist * 1.1):
        return False # Skip straight lines

    # Save
    save_path_img = os.path.join(save_dir, 'images', f'{patch_id}.png')
    save_path_meta = os.path.join(save_dir, 'meta', f'{patch_id}.npz')
    
    # Debug: Save the obstacle map occasionally to check validity
    if patch_id % 100 == 0:
        cv2.imwrite(os.path.join(save_dir, 'debug', f'{patch_id}_edges.png'), obstacle_map * 255)

    cv2.imwrite(save_path_img, patch_img)
    np.savez(save_path_meta, path=np.array(path), target=np.array(end), start=np.array(start))
    return True

def main():
    Path(os.path.join(OUTPUT_DIR, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUTPUT_DIR, 'meta')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUTPUT_DIR, 'debug')).mkdir(parents=True, exist_ok=True)

    print(f"Searching in: {RAW_IMAGES_DIR}")
    image_files = []
    for root, _, files in os.walk(RAW_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.tif')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} images. Generating HARD examples...")
    
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
                    print(f"Generated {global_counter} curved paths...", end='\r')
                    
    print(f"\nDone! Generated {global_counter} high-quality samples in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()