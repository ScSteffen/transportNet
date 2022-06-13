import torch
from torch import nn, Tensor
import math

from src.layers import LinearLayer


class NewtinImplictNet(nn.Module):
    def __init__(self, units, input_dim, output_dim, num_layers):
        super(NewtinImplictNet, self).__init__()
        self.num_layers = num_layers
        self.linearInput = LinearLayer(input_dim, units)
        self.block1 = TanhNewtonImplicitLayer(out_features=units, tol=1e-4, max_iter=50)
        self.block2 = TanhNewtonImplicitLayer(out_features=units, tol=1e-4, max_iter=50)
        self.block3 = TanhNewtonImplicitLayer(out_features=units, tol=1e-4, max_iter=50)
        self.block4 = TanhNewtonImplicitLayer(out_features=units, tol=1e-4, max_iter=50)

        self.linearOutput = LinearLayer(units, output_dim)

    def forward(self, x):
        z = self.linearInput(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        z = self.linearOutput(z)
        logits = nn.Softmax(dim=1)(z)
        return logits

    def print_grads(self) -> None:
        # t = self.linearInput.weight.grad
        # print(self.linearInput.weight.grad)

        # for i in range(784):
        #    print(t[i, :])
        # print(self.block1.weight.grad)
        # print(self.block2.weight.grad)
        print(self.block1.linear.weight.grad)
        # print(self.block4.weight.grad)
        # print(self.linearOutput.weight.grad)

    def print_weights(self) -> None:
        # print(self.linearInput.weight)

        # print(self.block1.weight)
        # print(self.block2.weight)
        print(self.block1.linear.weight)
        # print(self.block4.weight)
        # print(self.linearOutput.weight)


class TanhNewtonImplicitLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

        self.weight = nn.Parameter(torch.empty((out_features, out_features), dtype=torch.float))  # W^T
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float))
        self.activation = torch.tanh

    def forward(self, x):
        # Run Newton's method outside of the autograd framework
        with torch.no_grad():
            z = x
            self.iterations = 0
            while self.iterations < self.max_iter:
                # definition of g
                g = torch.matmul(z, self.weight) - x - self.activation(x) - self.bias
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break

                # newton step
                J = torch.transpose(self.weight, 0, 1).repeat(x.shape[0], 1, 1)  # assemble broadcastet matrix of system

                z = z - torch.solve(g[:, :, None], J)[0][:, :, 0]

                self.iterations += 1

        # reengage autograd and add the gradient hook
        # t = z - torch.tanh(self.linear(z) + x)
        z = z - (torch.matmul(z, self.weight) - x - self.activation(x) - self.bias)
        if z.requires_grad:
            z.register_hook(lambda grad: torch.solve(grad[:, :, None], J.transpose(1, 2))[0][:, :, 0])
        return z


class TanhNewtonImplicitLayerOrig(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        # Run Newton's method outside of the autograd framework
        with torch.no_grad():
            z = torch.tanh(x)
            self.iterations = 0
            while self.iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break

                # newton step
                J = torch.eye(z.shape[1])[None, :, :] - (1 / torch.cosh(z_linear) ** 2)[:, :,
                                                        None] * self.linear.weight[None, :, :]
                z = z - torch.solve(g[:, :, None], J)[0][:, :, 0]

                self.iterations += 1

        # reengage autograd and add the gradient hook
        # t = z - torch.tanh(self.linear(z) + x)
        z = torch.tanh(self.linear(z) + x)
        if z.requires_grad:
            z.register_hook(lambda grad: torch.solve(grad[:, :, None], J.transpose(1, 2))[0][:, :, 0])
        return z
