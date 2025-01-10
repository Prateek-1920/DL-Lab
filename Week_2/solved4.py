import torch
import numpy as np

def sigma(x):
    a = -x
    b = np.exp(a)
    c = 1+b
    d = 1.0/c

    dsdc = -1.0/c**2
    dsdb = dsdc * 1
    dsda = dsdb * np.exp(a)
    dsx = dsda * (-1)

    return dsx

def sigmatorch(x):
    y = 1.0 / (1.0 + torch.exp(-x))
    return y

input_x = 2.0
x = torch.tensor(input_x,requires_grad=True)
y = sigmatorch(x)
y.backward()

print("Manual sigmoid = ",sigma(input_x))
print(("Torch sigmoid= ",x.grad))

