import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset
class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Define the regression model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return self.w * x + self.b

# Prepare the data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

# Create dataset and dataloader
dataset = RegressionDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize model, loss function, and optimizer
model = RegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.001)

loss_list = []
for epoch in range(100):
    epoch_loss = 0.0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()  # Reset gradients
        y_pred = model(x_batch)  # Forward pass
        loss = criterion(y_pred, y_batch)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    loss_list.append(epoch_loss)
    print(f"Epoch {epoch+1}: Parameters w = {model.w.item()}, b = {model.b.item()}, loss = {epoch_loss}")

# Plot the loss over epochs
plt.plot(loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()