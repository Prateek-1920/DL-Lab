import torch

x = torch.tensor([2.0, 4.0])
y = torch.tensor([20.0, 40.0])

w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
alpha = 0.001

yval = []
for epoch in range(2):
    for j in range(len(x)):
        ypred = w * x[j] + b
        yval.append((ypred))
        loss = (ypred - y[j]) ** 2
        loss.backward()
        w.data -= alpha * w.grad
        b.data -= alpha * b.grad
        w.grad.zero_()
        b.grad.zero_()

    print(f"Epoch {epoch}: w = {w.item()}, b = {b.item()}")
print(yval)

