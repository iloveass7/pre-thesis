# evaluate_path_predictor.py
import torch
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
from model import PathPredictorDB_CNN

# --- CONFIG ---
MODEL_PATH = "path_predictor_epoch_100.pth"
DATA_DIR = "dataset_curved"
OUTPUT_DIR = "evaluation_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_input_tensor(image, start_pos, target_pos):
    """Create 3-channel input tensor matching training format"""
    # Channel 0: Map
    c0 = image.astype(np.float32) / 255.0
    
    # Channel 1: Target position (with blob)
    c1 = np.zeros_like(c0)
    draw_blob(c1, target_pos[0], target_pos[1])
    
    # Channel 2: Start/Rover position (with blob)
    c2 = np.zeros_like(c0)
    draw_blob(c2, start_pos[0], start_pos[1])
    
    # Stack channels
    input_tensor = np.stack([c0, c1, c2], axis=0)
    return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)

def draw_blob(canvas, r, c, size=3):
    """Draw a blob at position (r,c)"""
    h, w = canvas.shape
    r_min, r_max = max(0, r-size), min(h, r+size+1)
    c_min, c_max = max(0, c-size), min(w, c+size+1)
    canvas[r_min:r_max, c_min:c_max] = 1.0

def extract_ground_truth_path(meta_path):
    """Extract ground truth path from npz file"""
    data = np.load(meta_path)
    path = data['path']
    target = data['target']
    start = data['start']
    return path, start, target

def create_path_mask(path, size=128, thickness=2):
    """Create binary mask of the path"""
    mask = np.zeros((size, size), dtype=np.uint8)
    for i in range(len(path) - 1):
        pt1 = (int(path[i][1]), int(path[i][0]))
        pt2 = (int(path[i+1][1]), int(path[i+1][0]))
        cv2.line(mask, pt1, pt2, 255, thickness)
    return mask

def plan_path_from_value_map(value_map, start_pos, target_pos, max_steps=200):
    """Plan path using the predicted value map (greedy approach)"""
    value_grid = value_map[0, 0].cpu().numpy()
    value_grid = (value_grid - value_grid.min()) / (value_grid.max() - value_grid.min() + 1e-8)
    
    path = [start_pos]
    current = start_pos
    visited = set([tuple(start_pos)])
    
    for step in range(max_steps):
        r, c = current[0], current[1]
        
        # Check if reached target (within 3 pixels)
        if np.linalg.norm(np.array(current) - np.array(target_pos)) < 3:
            break
        
        # Get 8-neighborhood values
        best_move = None
        best_value = -float('inf')
        moves = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                nr, nc = r + dr, c + dc
                if 0 <= nr < 128 and 0 <= nc < 128:
                    move_value = value_grid[nr, nc]
                    
                    # Encourage moving towards target
                    dist_to_target = np.linalg.norm(np.array([nr, nc]) - np.array(target_pos))
                    move_value = move_value * 0.7 + (1.0 / (dist_to_target + 1)) * 0.3
                    
                    if (nr, nc) not in visited and move_value > best_value:
                        best_value = move_value
                        best_move = (nr, nc)
                    moves.append(((nr, nc), move_value))
        
        if best_move is None:
            # If all neighbors visited, pick the highest value move
            if moves:
                moves.sort(key=lambda x: x[1], reverse=True)
                for (nr, nc), val in moves:
                    if (nr, nc) not in visited:
                        best_move = (nr, nc)
                        break
        
        if best_move is None:
            break
        
        path.append(best_move)
        visited.add(best_move)
        current = best_move
    
    return np.array(path)

def calculate_path_metrics(pred_path, gt_path, gt_mask):
    """Calculate various path evaluation metrics"""
    # Create predicted path mask
    pred_mask = create_path_mask(pred_path, thickness=2)
    
    # Flatten masks for metric calculation
    gt_flat = (gt_mask.flatten() > 0).astype(int)
    pred_flat = (pred_mask.flatten() > 0).astype(int)
    
    # Calculate pixel-level metrics
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    # Calculate path similarity (Hausdorff distance approximation)
    if len(pred_path) > 1 and len(gt_path) > 1:
        # Average distance from predicted path to ground truth
        min_dists = []
        for pred_pt in pred_path:
            dists = np.linalg.norm(gt_path - pred_pt, axis=1)
            min_dists.append(np.min(dists))
        
        avg_distance = np.mean(min_dists) if min_dists else float('inf')
        
        # Path length ratio
        pred_length = len(pred_path)
        gt_length = len(gt_path)
        length_ratio = pred_length / gt_length if gt_length > 0 else float('inf')
        
        # Success rate (reached within 5 pixels of target)
        final_dist = np.linalg.norm(pred_path[-1] - gt_path[-1])
        success = final_dist < 5
    else:
        avg_distance = float('inf')
        length_ratio = float('inf')
        success = False
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'avg_distance': avg_distance,
        'length_ratio': length_ratio,
        'success': success,
        'pred_length': len(pred_path),
        'gt_length': len(gt_path)
    }

def visualize_paths(image, pred_path, gt_path, start_pos, target_pos, save_path):
    """Create clean visualization comparing predicted and ground truth paths"""
    # Create visualization image
    vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw ground truth path (green)
    for i in range(len(gt_path) - 1):
        pt1 = (int(gt_path[i][1]), int(gt_path[i][0]))
        pt2 = (int(gt_path[i+1][1]), int(gt_path[i+1][0]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)  # Green
    
    # Draw predicted path (red)
    for i in range(len(pred_path) - 1):
        pt1 = (int(pred_path[i][1]), int(pred_path[i][0]))
        pt2 = (int(pred_path[i+1][1]), int(pred_path[i+1][0]))
        cv2.line(vis_img, pt1, pt2, (0, 0, 255), 2)  # Red
    
    # Draw start (cyan) and target (magenta)
    cv2.circle(vis_img, (int(start_pos[1]), int(start_pos[0])), 4, (255, 255, 0), -1)  # Cyan
    cv2.circle(vis_img, (int(target_pos[1]), int(target_pos[0])), 6, (255, 0, 255), -1)  # Magenta
    
    # Save clean visualization (no text)
    cv2.imwrite(save_path, vis_img)
    
    return vis_img

def create_metric_visualization(metrics_dict, save_path):
    """Create a separate visualization for metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Path Prediction Evaluation Metrics', fontsize=16)
    
    metric_names = ['precision', 'recall', 'f1', 'accuracy', 'avg_distance', 'length_ratio']
    titles = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Avg Distance', 'Length Ratio']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for idx, (metric, title, color) in enumerate(zip(metric_names, titles, colors)):
        row = idx // 3
        col = idx % 3
        
        if metric in metrics_dict and len(metrics_dict[metric]) > 0:
            values = metrics_dict[metric]
            ax = axes[row, col]
            
            # Plot histogram
            ax.hist(values, bins=20, alpha=0.7, color=color, edgecolor='black')
            
            # Add vertical line for mean
            mean_val = np.mean(values)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            
            # Add statistics text
            stats_text = f'Mean: {mean_val:.3f}\nStd: {np.std(values):.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(title)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[row, col].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            axes[row, col].set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_model():
    """Main evaluation function"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "value_maps"), exist_ok=True)
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first or specify correct path.")
        return
    
    model = PathPredictorDB_CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Get all samples
    meta_files = glob.glob(os.path.join(DATA_DIR, "meta", "*.npz"))
    if not meta_files:
        print(f"Error: No data found in {DATA_DIR}")
        return
    
    print(f"Found {len(meta_files)} samples for evaluation")
    
    # Initialize metrics accumulators
    all_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': [],
        'avg_distance': [],
        'length_ratio': [],
        'success': []
    }
    
    # Store detailed results for each sample
    sample_results = []
    
    # Evaluate each sample
    for i, meta_path in enumerate(meta_files[:100]):  # Evaluate first 50 samples
        try:
            # Load ground truth data
            file_id = os.path.basename(meta_path).replace(".npz", "")
            gt_path, start_pos, target_pos = extract_ground_truth_path(meta_path)
            
            # Load image
            img_path = os.path.join(DATA_DIR, "images", f"{file_id}.png")
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Create input tensor
            input_tensor = create_input_tensor(image, start_pos, target_pos)
            
            # Get value map prediction
            with torch.no_grad():
                value_map = model(input_tensor)
            
            # Plan path from value map
            pred_path = plan_path_from_value_map(value_map, start_pos, target_pos)
            
            # Create ground truth mask
            gt_mask = create_path_mask(gt_path)
            
            # Calculate metrics
            metrics = calculate_path_metrics(pred_path, gt_path, gt_mask)
            
            # Accumulate metrics
            for key in all_metrics.keys():
                if key in metrics:
                    if key == 'success':
                        all_metrics[key].append(metrics[key])
                    elif not np.isinf(metrics[key]):
                        all_metrics[key].append(metrics[key])
            
            # Store sample results
            sample_results.append({
                'file_id': file_id,
                'metrics': metrics,
                'pred_path_length': len(pred_path),
                'gt_path_length': len(gt_path)
            })
            
            # Create and save clean visualization (no text)
            vis_save_path = os.path.join(OUTPUT_DIR, "visualizations", f"{file_id}_comparison.png")
            visualize_paths(image, pred_path, gt_path, start_pos, target_pos, vis_save_path)
            
            # Save value map visualization
            value_map_np = value_map[0, 0].cpu().numpy()
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Input Map')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(value_map_np, cmap='hot')
            plt.colorbar()
            plt.title('Predicted Value Map')
            plt.axis('off')
            
            plt.savefig(os.path.join(OUTPUT_DIR, "value_maps", f"{file_id}_value_map.png"))
            plt.close()
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{min(50, len(meta_files))} samples")
                
        except Exception as e:
            print(f"Error processing sample {meta_path}: {e}")
            continue
    
    # Calculate average metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values and len(values) > 0:
            avg_metrics[key] = np.mean(values)
            print(f"{key.capitalize()}: {avg_metrics[key]:.4f}")
        else:
            avg_metrics[key] = 0
            print(f"{key.capitalize()}: No valid values")
    
    # Calculate success rate
    success_rate = np.mean(all_metrics['success']) if all_metrics['success'] else 0
    print(f"Success Rate: {success_rate:.4f}")
    
    # Create metrics visualization
    metric_vis_path = os.path.join(OUTPUT_DIR, "metrics_distribution.png")
    create_metric_visualization(all_metrics, metric_vis_path)
    
    # Create results summary table
    print("\n" + "="*50)
    print("TOP 5 PERFORMING SAMPLES")
    print("="*50)
    
    # Sort by F1-score
    sample_results_sorted = sorted(sample_results, key=lambda x: x['metrics']['f1'], reverse=True)
    
    for idx, result in enumerate(sample_results_sorted[:5]):
        print(f"\nSample {idx+1}: {result['file_id']}")
        print(f"  F1-Score: {result['metrics']['f1']:.4f}")
        print(f"  Precision: {result['metrics']['precision']:.4f}")
        print(f"  Recall: {result['metrics']['recall']:.4f}")
        print(f"  Success: {result['metrics']['success']}")
        print(f"  Pred Length: {result['pred_path_length']}, GT Length: {result['gt_path_length']}")
    
    # Save detailed results to JSON
    results_dict = {
        'summary': {
            'num_samples_evaluated': len(sample_results),
            'avg_precision': avg_metrics.get('precision', 0),
            'avg_recall': avg_metrics.get('recall', 0),
            'avg_f1': avg_metrics.get('f1', 0),
            'avg_accuracy': avg_metrics.get('accuracy', 0),
            'avg_distance': avg_metrics.get('avg_distance', 0),
            'avg_length_ratio': avg_metrics.get('length_ratio', 0),
            'success_rate': success_rate
        },
        'sample_details': sample_results,
        'all_metrics': all_metrics
    }
    
    with open(os.path.join(OUTPUT_DIR, "detailed_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    # Save concise results to text file
    with open(os.path.join(OUTPUT_DIR, "results_summary.txt"), 'w') as f:
        f.write("="*60 + "\n")
        f.write("PATH PREDICTION EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Number of samples evaluated: {len(sample_results)}\n")
        f.write(f"Model used: {MODEL_PATH}\n")
        f.write(f"Dataset: {DATA_DIR}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Average Precision:  {avg_metrics.get('precision', 0):.4f}\n")
        f.write(f"Average Recall:     {avg_metrics.get('recall', 0):.4f}\n")
        f.write(f"Average F1-Score:   {avg_metrics.get('f1', 0):.4f}\n")
        f.write(f"Average Accuracy:   {avg_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Average Distance:   {avg_metrics.get('avg_distance', 0):.2f} pixels\n")
        f.write(f"Average Length Ratio: {avg_metrics.get('length_ratio', 0):.3f}\n")
        f.write(f"Success Rate:       {success_rate:.4f} ({int(success_rate*len(sample_results))}/{len(sample_results)})\n\n")
        
        f.write("-"*60 + "\n")
        f.write("TOP 5 SAMPLES BY F1-SCORE\n")
        f.write("-"*60 + "\n")
        for idx, result in enumerate(sample_results_sorted[:5]):
            f.write(f"\n{idx+1}. {result['file_id']}:\n")
            f.write(f"   F1: {result['metrics']['f1']:.4f}, Precision: {result['metrics']['precision']:.4f}, "
                   f"Recall: {result['metrics']['recall']:.4f}\n")
            f.write(f"   Success: {result['metrics']['success']}, "
                   f"Path Length: {result['pred_path_length']}/{result['gt_path_length']}\n")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Visualizations: {os.path.join(OUTPUT_DIR, 'visualizations')}")
    print(f"Value maps: {os.path.join(OUTPUT_DIR, 'value_maps')}")
    print(f"Detailed results: {os.path.join(OUTPUT_DIR, 'detailed_results.json')}")
    print(f"Summary: {os.path.join(OUTPUT_DIR, 'results_summary.txt')}")
    print(f"Metrics visualization: {metric_vis_path}")

if __name__ == "__main__":
    evaluate_model()