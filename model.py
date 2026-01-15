# model_path_predictor.py
import torch
import numpy as np  
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class PathPredictorDB_CNN(nn.Module):
    """
    Enhanced DB-CNN that predicts VALUE MAP for entire path planning
    instead of just the next move
    """
    def __init__(self):
        super(PathPredictorDB_CNN, self).__init__()
        
        input_channels = 3  # map, start, target
        
        # --- SHARED FEATURE EXTRACTOR ---
        self.shared_features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # --- BRANCH 1: GLOBAL CONTEXT (Value Map) ---
        self.global_branch = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # --- BRANCH 2: LOCAL GUIDANCE ---
        self.local_branch = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # --- FUSION AND VALUE MAP PREDICTION ---
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32+32=64 channels
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Upsample back to original size
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),  # Single channel value map
        )
        
        # --- PATH DECODER (Optional: Predict waypoints) ---
        self.path_decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 50 * 2),  # Predict 50 waypoints (x,y)
        )
    
    def forward(self, x, predict_path=False):
        """
        Args:
            x: [B, 3, 128, 128] - map, start, target channels
            predict_path: If True, also predict waypoints
        Returns:
            value_map: [B, 1, 128, 128] - Value function for all positions
            waypoints: [B, 50, 2] - Optional predicted waypoints
        """
        # Extract shared features
        features = self.shared_features(x)  # [B, 32, 32, 32]
        
        # Global context
        global_feat = self.global_branch(features)  # [B, 32, 32, 32]
        
        # Local features
        local_feat = self.local_branch(features)  # [B, 32, 32, 32]
        
        # Fuse features
        fused = torch.cat([global_feat, local_feat], dim=1)  # [B, 64, 32, 32]
        fused = self.fusion(fused)  # [B, 16, 32, 32]
        
        # Predict value map (heatmap of good positions)
        value_map = self.upsample(fused)  # [B, 1, 128, 128]
        
        if predict_path:
            # Predict waypoints
            waypoints = self.path_decoder(fused)  # [B, 100]
            waypoints = waypoints.view(-1, 50, 2)  # [B, 50, 2]
            return value_map, waypoints
        
        return value_map

class NonLinearPathPlanner(nn.Module):
    """
    Wrapper model that uses value map to plan non-linear paths
    """
    def __init__(self):
        super(NonLinearPathPlanner, self).__init__()
        self.db_cnn = PathPredictorDB_CNN()
        
        # LSTM for sequential path generation
        self.lstm = nn.LSTM(input_size=128*128, hidden_size=256, num_layers=2, batch_first=True)
        self.position_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Predict next (row, col)
        )
    
    def plan_path(self, value_map, start_pos, max_steps=100):
        """
        Use value map to plan a complete path
        """
        batch_size = value_map.size(0)
        paths = []
        
        for b in range(batch_size):
            path = [start_pos[b].cpu().numpy()]
            current = start_pos[b].cpu().numpy()
            
            value_grid = value_map[b, 0].cpu().numpy()  # [128, 128]
            
            # Normalize value grid
            value_grid = (value_grid - value_grid.min()) / (value_grid.max() - value_grid.min() + 1e-8)
            
            for step in range(max_steps):
                # Get current position value
                r, c = int(current[0]), int(current[1])
                
                # Look at neighbors
                best_move = None
                best_value = -float('inf')
                
                # Check 8 directions
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                            
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 128 and 0 <= nc < 128:
                            neighbor_value = value_grid[nr, nc]
                            
                            # Penalize moving away from target direction
                            # (Simple heuristic - can be improved)
                            if neighbor_value > best_value:
                                best_value = neighbor_value
                                best_move = (dr, dc)
                
                if best_move is None:
                    break
                    
                # Move with some randomness for non-linearity
                if np.random.random() < 0.1:  # 10% chance to take suboptimal move
                    moves = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
                    best_move = moves[np.random.randint(len(moves))]
                
                new_pos = (r + best_move[0], c + best_move[1])
                path.append(new_pos)
                current = new_pos
                
                # Check if we've reached high-value region
                if value_grid[new_pos[0], new_pos[1]] > 0.8:
                    break
            
            paths.append(np.array(path))
        
        return paths