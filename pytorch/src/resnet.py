import torch
from torch import nn, Tensor
import math
from src.layers import LinearLayer


class ResNet(nn.Module):
    def __init__(self, units, input_dim, output_dim, num_layers, device):
        super(ResNet, self).__init__()

        self.num_layers = num_layers
        self.linearInput = LinearLayer(input_dim, units)

        self.blocks = torch.nn.ModuleList(
            [ResNetLayer(units, units, device=device) for _ in range(num_layers)])

        self.linearOutput = LinearLayer(units, output_dim)

    def forward(self, x):
        z = self.linearInput(x)
        for block in self.blocks:
            z = block(z)

        z = self.linearOutput(z)
        logits = nn.Softmax(dim=1)(z)
        return logits

    def print_grads(self) -> None:
        print(self.linearInput.weight.grad)

        print(self.resNetBlocks1.weight.grad)
        print(self.resNetBlocks2.weight.grad)
        print(self.resNetBlocks3.weight.grad)
        print(self.resNetBlocks4.weight.grad)
        print(self.linearOutput.weight.grad)


class ResNetLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
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

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Layer input
        :return: Output y of implicit Layer Ay = x + f(x) + b
        """
        y = x + torch.matmul(self.activation(x), self.weight) + self.bias

        return y
