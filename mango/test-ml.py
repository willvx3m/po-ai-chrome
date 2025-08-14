import torch
import torch.nn as nn
import torch.optim as optim

# Simple neural network
model = nn.Linear(100, 10).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Dummy data
x = torch.randn(64, 100).cuda()
y = torch.randn(64, 10).cuda()

# Training step
model.train()
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
print("Training step completed, Loss:", loss.item())