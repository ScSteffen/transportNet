import torch
from torch import nn, Tensor
import math

from src.layers import LinearLayer


class TransNetSweeping(nn.Module):
    def __init__(self, units, input_dim, output_dim, num_layers, epsilon, dt, device):
        super(TransNetSweeping, self).__init__()
        self.num_layers = num_layers
        self.units = units
        self.epsilon = epsilon
        self.dt = dt
        self.tol = 1e-4

        self.device = device
        self.linearInput = LinearLayer(input_dim, units)
        self.block1 = TransNetLayerSweeping(units, units, epsilon=epsilon, dt=dt, device=device)
        self.block2 = TransNetLayerSweeping(units, units, epsilon=epsilon, dt=dt, device=device)
        self.block3 = TransNetLayerSweeping(units, units, epsilon=epsilon, dt=dt, device=device)
        self.block4 = TransNetLayerSweeping(units, units, epsilon=epsilon, dt=dt, device=device)

        self.linearOutput = LinearLayer(units, output_dim)

    def forward(self, x):
        x = self.linearInput(x)
        z_in = torch.cat((x, self.block1.activation(x)), 1)

        # Source_Iteration
        with torch.no_grad():
            # 1) initialize model
            self.block1.initialize_model(x)
            self.block2.initialize_model(x)
            self.block3.initialize_model(x)
            self.block4.initialize_model(x)

            # 2) Setup system matrix
            self.block1.setup_system_mat()
            self.block2.setup_system_mat()
            self.block3.setup_system_mat()
            self.block4.setup_system_mat()

            step = 0
            total_err = 10
            while step < self.steps or total_err < self.tol:
                # 1) relax
                self.block1.relax()
                self.block2.relax()
                self.block3.relax()
                self.block4.relax()

                # 2) sweep
                z, err1 = self.block1.sweep(z_in)
                z, err2 = self.block2.sweep(z)
                z, err3 = self.block3.sweep(z)
                z, err4 = self.block4.sweep(z)
                total_err = 1. / 4. * (err1 + err2 + err3 + err4)

        # Forward iteration for gradient tape
        z = self.block1.implicit_forward(z_in)
        z = self.block2.implicit_forward(z)
        z = self.block3.implicit_forward(z)
        z = self.block4.implicit_forward(z)

        #  linear output
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


class TransNetLayerSweeping(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    A: Tensor
    rhs: Tensor
    z_k: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, epsilon=0.01, dt=0.1, device="CPU",
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
        self.A = torch.eye(2 * self.out_features).to(self.device)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, sweep):
        """
        :param x: Layer input
        :return: Output y of implicit Layer Ay = x + f(x) + b
        """

        # y = x
        with torch.no_grad():
            # 1)  assemble right hand side (relaxation step)

            u_in = x[:, :self.out_features]
            v_in = x[:, self.out_features:]
            rhs = torch.cat((u_in + self.dt * self.bias, v_in + self.dt / self.epsilon * self.activation(u_in)),
                            1) - sweep

            rhs = rhs[:, :, None]  # assemble broadcasted rhs of system

            # 2) Solve system (detached from gradient tape)
            A = torch.eye(2 * self.out_features).to(self.device)
            A[:self.out_features, self.out_features:] = self.dt * self.weight
            A[self.out_features:, :self.out_features] = - self.dt * torch.transpose(self.weight, 0, 1)
            A[self.out_features:, self.out_features:] = A[self.out_features:,
                                                        self.out_features:] + self.dt / self.epsilon * torch.eye(
                self.out_features, device=self.device)

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
        v_out = - self.dt * torch.matmul(u,
                                         w) + self.dt / self.epsilon * v - v_in - self.dt / self.epsilon * self.activation(
            u_in)

        # Assemble layer solution vector
        y = torch.cat((u_out, v_out), 1)

        # 4) Assemble sweep for next layer
        sweep = sweep + torch.matmul(y, torch.transpose(A[0], 0, 1))

        # 5) Use implicit function theorem (or adjoint equation of the KKT system to compute the real gradient)
        #   Let g = Ay - (x + f(x) + b) (layer in fixed point notation). Then grad = dg/dx.
        #   We need gradient dy/dx, using dg/dy*dy/dx =dg/dy with A=dg/dy
        if y.requires_grad:
            y.register_hook(lambda grad: torch.solve(grad[:, :, None], A.transpose(1, 2))[0][:, :, 0])
        return y, sweep

    def initialize_model(self, input_x):
        """
        :param input_x: input data
        :return:
        """
        self.z_l = torch.cat((input_x, self.activation(input_x)), 1)

        return self.z_l

    def setup_system_mat(self):
        """
        :param z_in: [u,v]
        :return: constructs the current System matrix.
                DONT USE WITH GRADIENT TAPE ACTIVE
        """
        # 1)  assemble system matrix
        self.A[:self.out_features, :self.out_features] = torch.eye(self.out_features, device=self.device)
        self.A[:self.out_features, self.out_features:] = self.dt * self.weight
        self.A[self.out_features:, :self.out_features] = - self.dt * torch.transpose(self.weight, 0, 1)
        self.A[self.out_features:, self.out_features:] = (1 + self.dt / self.epsilon) * torch.eye(self.out_features,
                                                                                                  device=self.device)
        return 0

    def relax(self):
        """
        :param z_in: [u,v]
        :return: constructs the rhs of the relaxation system.
                DONT USE WITH GRADIENT TAPE ACTIVE
        """
        # 1)  assemble right hand side
        self.rhs = torch.cat(
            (self.dt * self.bias, self.dt / self.epsilon * self.activation(self.z_l[:, :self.out_features])), 1)

        return 0

    def sweep(self, z_lp1_i):
        """
            :param z_lp1_i: sweep solve of previous layer
            :return: Output y of implicit Layer Ay = x + f(x) + b
                    DONT USE WITH GRADIENT TAPE ACTIVE
        """
        A = self.A.repeat(self.z_l.shape[0], 1, 1).to(self.device)
        rhs = self.rhs + z_lp1_i

        rhs = rhs[:, :, None]

        y = torch.solve(rhs, A)[0][:, :, 0]
        error = torch.mean(torch.linalg.norm(self.z_l - y, dim=1))
        self.z_l = y
        return self.z_l, error

    def implicit_forward(self, z_in):

        return z_out
