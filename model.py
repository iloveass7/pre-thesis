import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Implementation of the Residual Block shown in Figure 5[cite: 187].
    Structure: Conv -> ReLU -> Conv -> Sum(Input) -> ReLU
    """
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
        out += residual # The "Skip Connection"
        out = F.relu(out)
        return out

class DBCNN(nn.Module):
    def __init__(self):
        super(DBCNN, self).__init__()
        
        # --- INPUT DEFINITION ---
        # We use 3 channels: 
        # 1. The Mars Map (Gray)
        # 2. The Target Position (One-hot map)
        # 3. The Rover Position (One-hot map)
        input_channels = 3 

        # --- REPROCESSING LAYERS (Shared) [cite: 139, 190] ---
        # Conv-00: 6 filters, 5x5, stride 1
        self.conv00 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        # Pool-00: 3x3, stride 2
        self.pool00 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Conv-01: 12 filters, 4x4, stride 1
        self.conv01 = nn.Conv2d(6, 12, kernel_size=4, padding=1) # padding adjusted for 4x4
        # Pool-01: 3x3, stride 2
        self.pool01 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # After these layers, a 128x128 image becomes approx 32x32

        # --- BRANCH ONE (Global) [cite: 170, 190] ---
        # Conv-10: 20 filters, 5x5
        self.b1_conv10 = nn.Conv2d(12, 20, kernel_size=5, padding=2)
        self.b1_pool10 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Res-11
        self.b1_res11 = ResidualBlock(20, 20)
        self.b1_pool11 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Res-12
        self.b1_res12 = ResidualBlock(20, 20)
        self.b1_pool12 = nn.MaxPool2d(3, stride=1, padding=1) # Stride 1 per table
        
        # Res-13
        self.b1_res13 = ResidualBlock(20, 20)
        self.b1_pool13 = nn.MaxPool2d(3, stride=1, padding=1)

        # Fully Connected Layers for Branch 1
        # Calculation: 32x32 -> pool(2) -> 16 -> pool(2) -> 8 -> pool(1) -> 8 -> pool(1) -> 8
        # Final map size is roughly 8x8 with 20 channels = 1280 inputs
        self.b1_fc1 = nn.Linear(20 * 8 * 8, 192)
        self.b1_fc2 = nn.Linear(192, 10)

        # --- BRANCH TWO (Local) [cite: 174, 190] ---
        # Conv-20: 20 filters, 5x5
        self.b2_conv20 = nn.Conv2d(12, 20, kernel_size=5, padding=2)
        
        # 4 Residual Blocks in a row
        self.b2_res21 = ResidualBlock(20, 20)
        self.b2_res22 = ResidualBlock(20, 20)
        self.b2_res23 = ResidualBlock(20, 20)
        self.b2_res24 = ResidualBlock(20, 20)
        
        # Conv-21: 10 filters, 3x3
        self.b2_conv21 = nn.Conv2d(20, 10, kernel_size=3, padding=1)
        
        # Branch 2 Output Flattening
        # Map size here is still approx 32x32 (no pooling in Branch 2)
        # 10 channels * 32 * 32 = 10240
        self.b2_fc3_input_size = 10 * 32 * 32 

        # --- FUSION & OUTPUT [cite: 156, 190] ---
        # Takes output of B1 (10) + Output of B2 (10240)
        # Note: In the paper, they likely crop B2 at the rover location, 
        # but flattening is a safer/easier implementation for a thesis baseline.
        self.fc3 = nn.Linear(10 + self.b2_fc3_input_size, 8) 

    def forward(self, x):
        # x shape: [Batch, 3, 128, 128]
        
        # --- Shared Reprocessing ---
        x = F.relu(self.conv00(x))
        x = self.pool00(x)
        x = F.relu(self.conv01(x))
        x = self.pool01(x)
        # x is now the "Deep Feature Map" (approx 32x32)

        # --- Branch 1 (Global) ---
        out1 = F.relu(self.b1_conv10(x))
        out1 = self.b1_pool10(out1)
        out1 = self.b1_res11(out1)
        out1 = self.b1_pool11(out1)
        out1 = self.b1_res12(out1)
        out1 = self.b1_pool12(out1)
        out1 = self.b1_res13(out1)
        out1 = self.b1_pool13(out1)
        
        out1 = out1.view(out1.size(0), -1) # Flatten
        out1 = F.relu(self.b1_fc1(out1))
        out1 = F.relu(self.b1_fc2(out1)) # Vector of size 10

        # --- Branch 2 (Local) ---
        out2 = F.relu(self.b2_conv20(x))
        out2 = self.b2_res21(out2)
        out2 = self.b2_res22(out2)
        out2 = self.b2_res23(out2)
        out2 = self.b2_res24(out2)
        out2 = F.relu(self.b2_conv21(out2))
        
        out2 = out2.view(out2.size(0), -1) # Flatten

        # --- Fusion ---
        # Combine the Global info (out1) and Local info (out2)
        combined = torch.cat((out1, out2), dim=1)
        
        # Final prediction (8 directions)
        prediction = self.fc3(combined)
        #print(prediction.shape)
        return prediction # Returns raw scores (logits), use CrossEntropyLoss later