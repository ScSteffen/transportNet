import torch
from torch import nn, Tensor
import math

from src.layers import LinearLayer


class NewtonImplicitNet(nn.Module):
    def __init__(self, units, input_dim, output_dim, num_layers, device):
        super(NewtonImplicitNet, self).__init__()
        self.num_layers = num_layers
        self.linearInput = LinearLayer(input_dim, units)

        self.blocks = torch.nn.ModuleList(
            [TanhNewtonImplicitLayer(out_features=units, tol=1e-4, max_iter=50, device=device) for _ in
             range(num_layers)])

        self.linearOutput = LinearLayer(units, output_dim)

    def forward(self, x):
        z = self.linearInput(x)
        for block in self.blocks:
            z = block(z)

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
    def __init__(self, out_features, tol=1e-4, max_iter=50, device="cuda"):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter
        self.device = device

    def forward(self, x):
        # Run Newton's method outside of the autograd framework
        with torch.no_grad():
            z = torch.tanh(x)
            self.iterations = 0
            while self.iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                self.err = torch.norm(g)

                # newton step
                J = torch.eye(z.shape[1], device=self.device)[None, :, :] - (1 / torch.cosh(z_linear) ** 2)[:, :,
                                                                            None] * self.linear.weight[None, :, :]
                if self.err < self.tol:
                    break

                z = z - torch.solve(g[:, :, None], J)[0][:, :, 0]

                self.iterations += 1

        # reengage autograd and add the gradient hook
        # t = z - torch.tanh(self.linear(z) + x)
        z = torch.tanh(self.linear(z) + x)
        
        if z.requires_grad:
            z.register_hook(lambda grad: torch.solve(grad[:, :, None], J.transpose(1, 2))[0][:, :, 0])
        return z
