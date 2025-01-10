import torch
import numpy
from torch.nn.functional import relu_

def relu(x):
	return max(0.0, x)


b = torch.tensor(1.0,requires_grad=True)
x = torch.tensor(2.0,requires_grad=True)
w = torch.tensor(3.0,requires_grad=True)

u = w * x
v = u+b
a = torch.relu_(v)
a.backward()

binp = 1.0
xinp = 2.0
winp = 3.0

dadw = 1.0 * 2.0

print("Torch gradient: ",w.grad)
print(("Manual gradient: ",dadw))


