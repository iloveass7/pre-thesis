# evaluate_paths.py
import torch
import cv2
import numpy as np
import os
import glob
from model import PathPredictorDB_CNN, NonLinearPathPlanner

def visualize_paths():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = PathPredictorDB_CNN().to(DEVICE)
    model.load_state_dict(torch.load("path_predictor_epoch_20.pth", map_location=DEVICE))
    model.eval()
    
    planner = NonLinearPathPlanner()
    planner.db_cnn = model
    
    # Test on sample images
    test_images = glob.glob("dataset_curved/images/*.png")[:5]
    
    for i, img_path in enumerate(test_images):
        # Load and prepare input
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # Create random start and target
        h, w = image.shape
        start = (np.random.randint(20, h-20), np.random.randint(20, w-20))
        target = (np.random.randint(20, h-20), np.random.randint(20, w-20))
        
        # Create input channels
        c0 = image
        c1 = np.zeros_like(image)
        c1[target[0]-2:target[0]+3, target[1]-2:target[1]+3] = 1.0
        c2 = np.zeros_like(image)
        c2[start[0]-2:start[0]+3, start[1]-2:start[1]+3] = 1.0
        
        input_tensor = torch.tensor(np.stack([c0, c1, c2], axis=0), 
                                   dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Get value map
        with torch.no_grad():
            value_map = model(input_tensor)
        
        # Plan path
        start_tensor = torch.tensor([start], dtype=torch.float32).to(DEVICE)
        path = planner.plan_path(value_map, start_tensor)[0]
        
        # Visualize
        vis = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Draw path
        for j in range(len(path) - 1):
            pt1 = (int(path[j][1]), int(path[j][0]))
            pt2 = (int(path[j+1][1]), int(path[j+1][0]))
            cv2.line(vis, pt1, pt2, (0, 0, 255), 2)
        
        # Draw start and target
        cv2.circle(vis, (int(start[1]), int(start[0])), 5, (0, 255, 0), -1)
        cv2.circle(vis, (int(target[1]), int(target[0])), 5, (255, 0, 0), -1)
        
        # Save visualization
        cv2.imwrite(f"nonlinear_path_{i}.png", vis)
        print(f"Saved path visualization {i}")

if __name__ == "__main__":
    visualize_paths()