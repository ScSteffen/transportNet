import torch
from torch import nn, Tensor
import math

from src.layers import LinearLayer


class TransNetSplit2(nn.Module):
    def __init__(self, units, input_dim, output_dim, num_layers, epsilon, dt, device):
        super(TransNetSplit2, self).__init__()
        self.num_layers = num_layers
        self.units = units
        self.linearInput = LinearLayer(input_dim, units)
        self.block1 = TransNetLayerSplit2(units, units, epsilon=epsilon, dt=dt, device=device)
        self.block2 = TransNetLayerSplit2(units, units, epsilon=epsilon, dt=dt, device=device)
        self.block3 = TransNetLayerSplit2(units, units, epsilon=epsilon, dt=dt, device=device)
        self.block4 = TransNetLayerSplit2(units, units, epsilon=epsilon, dt=dt, device=device)

        self.linearOutput = LinearLayer(units, output_dim)

    def forward(self, x):
        z = self.linearInput(x)
        z = torch.cat((z, self.block1.activation(z)), 1)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        z = z[:, :self.units]
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


class TransNetLayerSplit2(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    # A:Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, epsilon=0.01, dt=0.1, device="cpu",
                 dtype=None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = epsilon * torch.ones(1).to(device)
        self.dt = dt * torch.ones(1).to(device)
        self.device = device

        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float))  # W^T
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.activation = nn.Tanh()

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

        # y = x
        u_in = x[:, :self.out_features]
        v_in = x[:, self.out_features:]

        with torch.no_grad():
            # 1)  assemble right hand side (relaxation step)
            v_star = 1.0 / (1 + self.dt / self.epsilon) * (v_in + self.dt / self.epsilon * self.activation(u_in))

            rhs = torch.cat((u_in + self.dt * self.bias, v_star), 1)
            rhs = rhs[:, :, None]  # assemble broadcasted rhs of system

            # 2) Solve system (detached from gradient tape)
            A = torch.eye(2 * self.out_features).to(self.device)  # .double()
            A[:self.out_features, self.out_features:] = self.dt * self.weight
            A[self.out_features:, :self.out_features] = - self.dt * torch.transpose(self.weight, 0, 1)

            A = A.repeat(x.shape[0], 1, 1).to(self.device)  # assemble broadcastet matrix of system on device
            y = torch.solve(rhs, A)[0][:, :, 0]

            w = self.weight.clone().detach().requires_grad_(
                False)  # copy weights for forward pass, such that only the parameters of the first relaxation equation is used for gradient updates
        # 3)  reengage autograd and add the gradient hook
        u = y[:, :self.out_features]
        v = y[:, self.out_features:]
        # a) u part
        u_out = self.dt * torch.matmul(v, self.weight.T) - u_in - self.dt * self.bias
        # b) v part
        v_out = - self.dt * torch.matmul(u, w) + \
                self.dt / self.epsilon * v - v_in - self.dt / self.epsilon * self.activation(u_in)

        # Assemble layer solution vector
        y = torch.cat((u_out, v_out), 1)

        # 4) Use implicit function theorem (or adjoint equation of the KKT system to compute the real gradient)
        #   Let g = Ay - (x + f(x) + b) (layer in fixed point notation). Then grad = dg/dx.
        #   We need gradient dy/dx, using dg/dy*dy/dx =dg/dy with A=dg/dy
        if y.requires_grad:
            y.register_hook(lambda grad: torch.solve(grad[:, :, None], A.transpose(1, 2))[0][:, :, 0])
        return y
