import torch
from torch import nn


class ImplicitNet(nn.Module):
    def __init__(self, units, input_dim, output_dim):
        super(ImplicitNet, self).__init__()
        self.implict_stack = nn.Sequential(
            nn.Linear(input_dim, units),
            ImplicitLayer(units, units),
            ImplicitLayer(units, units),
            ImplicitLayer(units, units),
            ImplicitLayer(units, units),
            ImplicitLayer(units, units),
            nn.Linear(units, output_dim),
            nn.Softmax(dim=1)  # output activation
        )

    def forward(self, x):
        logits = self.implict_stack(x)
        return logits


class ImplicitLayer(nn.Module):
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
        # 1)  assemble right hand side
        rhs = x + self.activation(x) + self.b

        y = nn.Identity(x)
        # 2) Solve system (detached from gradient tape)
        with torch.no_grad():
            # Due to the first dimension being the batch dimenstion,  and pytorchs matmul convetion of zA
            # we need to transpose the linear system before solving
            y = torch.linalg.solve(torch.transpose(self.A, 0, 1), torch.transpose(rhs, 0, 1))
            y = torch.transpose(y, 0, 1)

        # 3)  reengage autograd and add the gradient hook
        y = y - (torch.matmul(y, self.A) - x - self.activation(x) - self.b)

        # 4) Use implicit function theorem (or adjoint equation of the KKT system to compute the real gradient)
        #   Let g = Ay - (x + f(x) + b) (layer in fixed point notation). Then grad = dg/dx.
        #   We need gradient dy/dx, using dg/dy*dy/dx =dg/dy with A=dg/dy
        if y.requires_grad:
            y.register_hook(lambda grad: torch.transpose(
                torch.linalg.solve(self.A, torch.transpose(grad, 0, 1)), 0, 1))
        return y
