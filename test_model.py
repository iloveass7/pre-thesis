import torch
from model import PathPredictorDB_CNN

# Create a random fake input (Batch Size 1, 3 Channels, 128x128)
fake_input = torch.randn(1, 3, 128, 128)

# Initialize model
model = PathPredictorDB_CNN()

# Try a forward pass
output = model(fake_input)

print("Model output shape:", output.shape)
# Should print: torch.Size([1, 8])