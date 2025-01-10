import torch

def f(x):
    return (x-2) ** 2

def fp(x):
    return 2*(x-2)

x = torch.tensor([1.0],requires_grad=True)
y = f(x)
y.backward()

print("Analytical f`(x): ",fp(x))
print("Pytorch f`(x)",x.grad)