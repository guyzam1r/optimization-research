import numpy as np
import torch
from torch import nn

def sphere(x):
    return torch.sum(x**2)

def ellipsoid(x):
    n = x.size(0)
    elli_coeffs = torch.pow(10, torch.arange(n)/(n-1))
    return torch.sum(elli_coeffs * x**2)

def tablet(x):
    return 10**6 * x[0]**2 + torch.sum(x[1:]**2)

def parabolic_ridge(x):
    return -x[0] + 100*torch.sum(x[1:]**2)

def sharp_ridge(x):
    return -x[0] + 100*torch.norm(x[1:])

def diffpow(x):
    n = x.size(0)
    out = 0
    for i in range(n):
        out += torch.abs(x[i])**(2 + 10*i/(n-1))
    return out

def rosenbrock(x):
    n = x.size(0)
    out = 0
    for i in range(n-1):
        out += 100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return out

#NOTE: Only takes x in R^2
def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def multi_layer_init(num_samples, in_dim, out_dim, w_dims, seed=12345):
    #NOTE: w_dims[0] must equal in_dim and w_dims[-1] must equal out_dim
    torch.manual_seed(seed)
    
    X = torch.rand(in_dim, num_samples, requires_grad=True)
    Y = torch.rand(out_dim, num_samples, requires_grad=True)
    
    return X, Y

sigma = nn.LeakyReLU(0.1)
sigma_inv = nn.LeakyReLU(10)
softmax = nn.Softmax(dim=1)

def forward_pass(w, w_dims, data):
    h = data
    start = 0
    for i in range(len(w_dims)-2):
        end = start + w_dims[i+1]*w_dims[i+2]
        cur_w = w[start:end].reshape(w_dims[i+2],w_dims[i+1])
        h = sigma(torch.matmul(cur_w, h))
        start = end
    h = softmax(torch.t(h))
    return h

crossentropy = nn.CrossEntropyLoss()

def multi_layer(w, X, Y):
    h = forward_pass(w, X)
    return crossentropy(h, Y)

