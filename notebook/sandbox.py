import torch
from torch import nn, Tensor
import numpy as np


def main():
    model = nn.Tanh().to("cpu")

    x = nn.Parameter(data=Tensor([[1., 2., 3.], [2., 3., 4.]]))

    pred = model(x)
    pred.backward(torch.ones_like(x))

    grad = x.grad

    grad_analytic = grad_activation(x)

    return 0


def grad_activation(z):
    return 1.0 / torch.cosh(z) ** 2


if __name__ == '__main__':
    main()
