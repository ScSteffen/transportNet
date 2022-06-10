import torch
from torch import nn


class ResNet(nn.Module):
    def __init__(self, units, input_dim, output_dim):
        super(ResNet, self).__init__()
        self.implict_stack = nn.Sequential(
            nn.Linear(input_dim, units),
            ResNetLayer(units, units),
            ResNetLayer(units, units),
            ResNetLayer(units, units),
            ResNetLayer(units, units),
            ResNetLayer(units, units),
            nn.Linear(units, output_dim),
            nn.Softmax(dim=1)  # output activation
        )

    def forward(self, x):
        logits = self.implict_stack(x)
        return logits


class ResNetLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, size_out):
        super().__init__()

        self.size_in = size_in
        self.size_out = size_out

        self.A = nn.Parameter(torch.randn(size_out, size_in, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(size_out, dtype=torch.float))

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        :param x: Layer input
        :return: Output y of implicit Layer Ay = x + f(x) + b
        """
        y = x + torch.matmul(self.activation(x), self.A) + self.b

        return y
