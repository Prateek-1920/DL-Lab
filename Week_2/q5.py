import torch
import numpy as np

def fp(x):
    return 32 * x**3 + 9 * x**2 + 14*x + 6

x = torch.tensor(1.0,requires_grad=True)
b = 8 * x**4 + 3 * x**3 + 7 * x**2 + 6*x + 3

b.backward()
xinp = 1.0
print("Torch Gradient: ",x.grad)
print("Maunal Gradient: ",fp(xinp))