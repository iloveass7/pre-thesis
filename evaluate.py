# evaluate_simple.py
import torch
import numpy as np
import os
import glob
import cv2
from model import PathPredictorDB_CNN

MODEL_PATH = "path_predictor_epoch_100.pth"
DATA_DIR = "dataset_curved"
OUTPUT_DIR = "successful_samples"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smart_extract_path(value_map, start_pos, target_pos):
    """SMART path extraction that won't get stuck"""
    value_norm = (value_map - value_map.min()) / (value_map.max() - value_map.min() + 1e-8)
    
    path = [tuple(start_pos)]
    current = np.array(start_pos, dtype=float)
    target_arr = np.array(target_pos)
    visited = set([tuple(start_pos)])
    
    # Add target attraction force
    for step in range(120):
        r, c = int(np.round(current[0])), int(np.round(current[1]))
        
        # Check if reached target
        if np.linalg.norm(current - target_arr) < 5:
            break
        
        # Get ALL possible moves with scores
        moves = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                    
                nr, nc = r + dr, c + dc
                if 0 <= nr < 128 and 0 <= nc < 128 and (nr, nc) not in visited:
                    # Value from map
                    val_score = value_norm[nr, nc]
                    
                    # STRONG target attraction
                    old_dist = np.linalg.norm(np.array([r, c]) - target_arr)
                    new_dist = np.linalg.norm(np.array([nr, nc]) - target_arr)
                    
                    if new_dist < old_dist:
                        # Bonus for moving toward target
                        target_bonus = 0.5
                    else:
                        # Small penalty for moving away
                        target_bonus = -0.1
                    
                    # Total score
                    total_score = val_score + target_bonus
                    moves.append((total_score, (dr, dc), (nr, nc)))
        
        if not moves:
            # If stuck, backtrack
            if len(path) > 2:
                path.pop()  # Remove last position
                current = np.array(path[-1])
                continue
            else:
                break
        
        # Sort by score and pick best
        moves.sort(reverse=True, key=lambda x: x[0])
        
        # Try multiple options if best move seems poor
        best_score, best_move, best_pos = moves[0]
        
        # If score is too low, try second best
        if best_score < 0.1 and len(moves) > 1:
            best_score2, best_move2, best_pos2 = moves[1]
            if best_score2 > best_score:
                best_move, best_pos = best_move2, best_pos2
        
        # Apply move
        new_r, new_c = best_pos
        path.append((new_r, new_c))
        visited.add((new_r, new_c))
        current = np.array([new_r, new_c])
        
        # Stop if making no progress
        if step > 20 and len(path) > 10:
            # Check progress toward target
            positions = np.array(path)
            recent_progress = np.linalg.norm(positions[-5:] - target_arr, axis=1)
            if np.std(recent_progress) < 0.5:  # Stuck in same area
                break
    
    return path

def save_visualization(image, pred_path, true_path, start, target, save_path):
    """Save visualization of predicted vs ground truth path"""
    # Create color image
    vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw ground truth path (green)
    if len(true_path) > 1:
        for i in range(len(true_path) - 1):
            pt1 = (int(true_path[i][1]), int(true_path[i][0]))
            pt2 = (int(true_path[i+1][1]), int(true_path[i+1][0]))
            cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
    
    # Draw predicted path (red)
    if len(pred_path) > 1:
        for i in range(len(pred_path) - 1):
            pt1 = (int(pred_path[i][1]), int(pred_path[i][0]))
            pt2 = (int(pred_path[i+1][1]), int(pred_path[i+1][0]))
            cv2.line(vis_img, pt1, pt2, (0, 0, 255), 2)
    
    # Draw start (cyan) and target (magenta)
    cv2.circle(vis_img, (int(start[1]), int(start[0])), 4, (255, 255, 0), -1)
    cv2.circle(vis_img, (int(target[1]), int(target[0])), 6, (255, 0, 255), -1)
    
    # Save image
    cv2.imwrite(save_path, vis_img)

def evaluate():
    """Simple evaluation - just print the metrics and save successful samples"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model = PathPredictorDB_CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Get samples
    meta_files = glob.glob(os.path.join(DATA_DIR, "meta", "*.npz"))
    num_samples = min(100, len(meta_files))
    
    acc, prec, rec, f1 = [], [], [], []
    successful_samples = []  # Store successful samples for saving
    
    print(f"Evaluating {num_samples} samples...")
    
    for i in range(num_samples):
        # Load data
        data = np.load(meta_files[i])
        file_id = os.path.basename(meta_files[i]).replace(".npz", "")
        img_path = os.path.join(DATA_DIR, "images", f"{file_id}.png")
        
        img = cv2.imread(img_path, 0)
        if img is None:
            continue
        
        true_path = [tuple(p) for p in data['path']]
        target, start = tuple(data['target']), tuple(data['start'])
        
        # Prepare input
        c0 = img.astype(np.float32) / 255.0
        c1, c2 = np.zeros_like(c0), np.zeros_like(c0)
        
        # Target
        tr, tc = target
        c1[max(0, tr-2):min(128, tr+3), max(0, tc-2):min(128, tc+3)] = 1.0
        
        # Start
        sr, sc = start
        c2[max(0, sr-2):min(128, sr+3), max(0, sc-2):min(128, sc+3)] = 1.0
        
        input_tensor = torch.tensor(np.stack([c0, c1, c2], axis=0), 
                                   dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            value_map = model(input_tensor)[0, 0].cpu().numpy()
        
        # Extract path using SMART method
        path = smart_extract_path(value_map, start, target)
        
        # Calculate metrics
        if path and true_path:
            threshold = 5
            
            # Accuracy - check if reached target
            final = path[-1]
            target_dist = np.sqrt((final[0]-target[0])**2 + (final[1]-target[1])**2)
            is_successful = target_dist < 50  # 50 pixel threshold for success
            acc.append(1.0 if is_successful else 0.0)
            
            # Precision
            correct = 0
            for p in path:
                min_dist = min(np.sqrt((p[0]-t[0])**2 + (p[1]-t[1])**2) for t in true_path)
                if min_dist < threshold:
                    correct += 1
            prec.append(correct / len(path) if len(path) > 0 else 0)
            
            # Recall
            covered = 0
            for t in true_path:
                min_dist = min(np.sqrt((t[0]-p[0])**2 + (t[1]-p[1])**2) for p in path)
                if min_dist < threshold:
                    covered += 1
            rec.append(covered / len(true_path) if len(true_path) > 0 else 0)
            
            # F1
            p, r = prec[-1], rec[-1]
            if p + r > 0:
                f1.append(2 * p * r / (p + r))
            else:
                f1.append(0)
            
            # Store successful samples for saving
            if is_successful and len(successful_samples) < 10:
                successful_samples.append({
                    'image': img,
                    'pred_path': path,
                    'true_path': true_path,
                    'start': start,
                    'target': target,
                    'file_id': file_id,
                    'accuracy': target_dist
                })
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {np.mean(acc)*100:.1f}%")
    print(f"Precision: {np.mean(prec)*100:.1f}%")
    print(f"Recall:    {np.mean(rec)*100:.1f}%")
    print(f"F1:        {np.mean(f1)*100:.1f}%")
    
    # Success statistics
    success_rate = np.mean(acc) * 100
    success_count = sum(1 for a in acc if a > 0)
    print(f"\nSuccess Rate: {success_rate:.1f}% ({success_count}/{len(acc)})")
    
    # Save successful samples
    if successful_samples:
        print(f"\nSaving {len(successful_samples)} successful samples...")
        
        for idx, sample in enumerate(successful_samples):
            save_path = os.path.join(OUTPUT_DIR, f"success_{idx+1}_{sample['file_id']}.png")
            save_visualization(
                sample['image'],
                sample['pred_path'],
                sample['true_path'],
                sample['start'],
                sample['target'],
                save_path
            )
            print(f"  Saved: {save_path}")
        
        # Also save a summary image with all successful samples
        if len(successful_samples) >= 4:
            create_summary_grid(successful_samples)
        
        print(f"\nAll successful samples saved to: {OUTPUT_DIR}/")
    else:
        print("\nNo successful samples found to save.")
    
    print("="*50)

def create_summary_grid(samples):
    """Create a summary grid of successful samples"""
    # Create a 2x2 or 3x3 grid based on number of samples
    num_samples = min(len(samples), 9)  # Max 3x3 grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create grid image
    grid_img = np.zeros((grid_size * 128, grid_size * 128, 3), dtype=np.uint8)
    
    for idx in range(num_samples):
        sample = samples[idx]
        
        # Create visualization for this sample
        vis_img = cv2.cvtColor(sample['image'], cv2.COLOR_GRAY2BGR)
        
        # Draw ground truth path (green)
        if len(sample['true_path']) > 1:
            for i in range(len(sample['true_path']) - 1):
                pt1 = (int(sample['true_path'][i][1]), int(sample['true_path'][i][0]))
                pt2 = (int(sample['true_path'][i+1][1]), int(sample['true_path'][i+1][0]))
                cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        # Draw predicted path (red)
        if len(sample['pred_path']) > 1:
            for i in range(len(sample['pred_path']) - 1):
                pt1 = (int(sample['pred_path'][i][1]), int(sample['pred_path'][i][0]))
                pt2 = (int(sample['pred_path'][i+1][1]), int(sample['pred_path'][i+1][0]))
                cv2.line(vis_img, pt1, pt2, (0, 0, 255), 1)
        
        # Draw start and target
        cv2.circle(vis_img, (int(sample['start'][1]), int(sample['start'][0])), 2, (255, 255, 0), -1)
        cv2.circle(vis_img, (int(sample['target'][1]), int(sample['target'][0])), 3, (255, 0, 255), -1)
        
        # Add sample number
        cv2.putText(vis_img, f"#{idx+1}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Place in grid
        row = idx // grid_size
        col = idx % grid_size
        grid_img[row*128:(row+1)*128, col*128:(col+1)*128] = vis_img
    
    # Save grid
    grid_path = os.path.join(OUTPUT_DIR, "successful_samples_grid.png")
    cv2.imwrite(grid_path, grid_img)
    print(f"  Summary grid saved to: {grid_path}")

if __name__ == "__main__":
    evaluate()