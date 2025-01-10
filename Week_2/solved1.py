import torch

x = torch.tensor(3.5,requires_grad=True)
y = x*x
z = 2*y+3

print("x: ",x)
print(("y = x*x: ",y))

z.backward()
print("Gradients dz/dx")
print(("Gradient at x=3.5: ",x.grad))



