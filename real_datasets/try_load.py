# script2_load.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Assuming SimpleModel is defined in this script as well
# Dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # Dummy init values
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # Dummy init values

# Load model state
model= torch.load('./model_state_dict.pth')

# Load optimizer and scheduler states
checkpoint = torch.load('./checkpoint.pth')
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

print("Model, optimizer, and scheduler states loaded successfully.")
print("Loaded states successfully.")
# Optionally, print or inspect the loaded states to verify
print(model.state_dict())
print(optimizer.state_dict())
print(scheduler.state_dict())