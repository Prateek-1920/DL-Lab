import numpy as np
import torch

# PyTorch setup
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

u = w * x
v = u + b
a = torch.sigmoid(v)
a.backward()

# Manual calculation
binp = 1.0
xinp = 2.0
winp = 3.0

vinp = binp + winp * xinp

# Compute the sigmoid and its derivative
sigmoid_vinp = 1 / (1 + np.exp(-vinp))
dadw = sigmoid_vinp * (1 - sigmoid_vinp) * xinp

# Print results
print("Torch gradient: ", w.grad)
print("Manual gradient: ", dadw)
