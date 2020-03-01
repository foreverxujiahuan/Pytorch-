import torch
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
a.requires_grad_(True)
b = (a * a).sum()
out.backward()
out2 = x.sum()
out2.backward()
out3 = x.sum()
x.grad.data.zero_()
out3.backward()
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
v = torch.tensor([[1, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
y3.backward()
print(y3)
x.backward()
print(x)
print(x.grad)
