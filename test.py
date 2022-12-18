import numpy as np
import torch
def f1(x, t):
  return 1 / (1 + torch.exp(-x + t))

def f2(x):
  return 1 - (0.5 - torch.mean(x,dim=-1))**2

import matplotlib.pyplot as plt

# Define the range of values to evaluate f2 at
t = torch.linspace(-5, 5, 100)
ndim = 3
nthresh = 4
x = torch.randn(ndim)
# Evaluate f2 at each value in the range
y = f2(f1(x[None,:], t[:,None]))

# Plot f2 as a function of t
plt.plot(t, y)
plt.show()

def observe(masks):
    assert masks.ndim == 2,'batch_size,ndim'
    masks = masks.detach().requires_grad_(True)
    y = f2(masks)
    grads = y.sum().backward()

    return y,grads

ndim = 3
nthresh = 4
x = torch.randn(ndim)
# Evaluate f2 at each value in the range
masks = f1(x[None,:], t[:,None])
# masks = torch.rand(nthresh,ndim)
y,grads = observe(masks)
# Plot f2 as a function of t
plt.plot(y.detach().numpy())
plt.show()