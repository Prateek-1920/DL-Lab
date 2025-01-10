import torch
import numpy as np

x = torch.tensor(1.0,requires_grad=True)
a = torch.exp(-x**2 - 2*x - torch.sin(x))
a.backward()
b = 1.0

ainp = np.exp(-b**2 - 2*b - np.sin(b)) * (-2*b - 2 - np.cos(b))

print("Torch gradient: ",x.grad)
print("Manual gradient: ",ainp)
