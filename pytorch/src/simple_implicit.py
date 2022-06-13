import torch
from torch import nn, Tensor
import math

from src.layers import LinearLayer


class ImplicitNet(nn.Module):
    def __init__(self, units, input_dim, output_dim, num_layers):
        super(ImplicitNet, self).__init__()
        self.num_layers = num_layers
        self.linearInput = LinearLayer(input_dim, units)
        self.block1 = ImplicitLayer(units, units)
        self.block2 = ImplicitLayer(units, units)
        self.block3 = ImplicitLayer(units, units)
        self.block4 = ImplicitLayer(units, units)

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
        print(self.block3.weight.grad)
        # print(self.block4.weight.grad)
        # print(self.linearOutput.weight.grad)

    def print_weights(self) -> None:
        # print(self.linearInput.weight)

        # print(self.block1.weight)
        # print(self.block2.weight)
        print(self.block3.weight)
        # print(self.block4.weight)
        # print(self.linearOutput.weight)


class ImplicitLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_features, out_features), dtype=torch.double))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.double))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.activation = nn.ReLU()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        :param x: Layer input
        :return: Output y of implicit Layer Ay = x + f(x) + b
        """
        # 1)  assemble right hand side
        rhs = x + self.activation(x) + self.bias

        # 2) Solve system (detached from gradient tape)
        with torch.no_grad():
            # Due to the first dimension being the batch dimenstion,  and pytorchs matmul convetion of zA
            # we need to transpose the linear system before solving
            y = torch.linalg.solve(torch.transpose(self.weight, 0, 1), torch.transpose(rhs, 0, 1))
            y = torch.transpose(y, 0, 1)

        # 3)  reengage autograd and add the gradient hook
        y = y - (torch.matmul(y, self.weight) - x - self.activation(x) - self.bias)

        # 4) Use implicit function theorem (or adjoint equation of the KKT system to compute the real gradient)
        #   Let g = Ay - (x + f(x) + b) (layer in fixed point notation). Then grad = dg/dx.
        #   We need gradient dy/dx, using dg/dy*dy/dx =dg/dy with A=dg/dy
        if y.requires_grad:
            y.register_hook(lambda grad: torch.transpose(
                torch.linalg.solve(self.weight, torch.transpose(grad, 0, 1)), 0, 1))
        return y
