# script1_train_save.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def dummy_dataset(size, batch_size=1):
    for _ in range(size):
        yield torch.randn(batch_size, 10), torch.randint(0, 2, (batch_size,)).long()


# Dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Initialize model, optimizer, scheduler
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

for inputs, targets in dummy_dataset(5, batch_size=1):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = nn.functional.cross_entropy(outputs, targets)  # Now targets should be correctly shaped
    loss.backward()
    optimizer.step()

# Save model state, optimizer state, and scheduler state
torch.save(model, './model_state_dict.pth')
torch.save({
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}, './checkpoint.pth')
print("Training completed and states saved.")
print(model.state_dict())
print(optimizer.state_dict())
print(scheduler.state_dict())
