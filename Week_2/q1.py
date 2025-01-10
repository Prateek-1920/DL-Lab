import torch
import numpy as np

a = torch.tensor(1.0,requires_grad=True)
b = torch.tensor(2.0,requires_grad=True)

x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

z.backward()

ainp = 1.0
binp = 2.0
dzda = 4 + 30 * ainp



print("Torch gradient : ",a.grad)
print("Analytical gradient : ",dzda)

