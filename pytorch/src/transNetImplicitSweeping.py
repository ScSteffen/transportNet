import torch
from torch import nn, Tensor
import math

from src.layers import LinearLayer


class TransNetSweeping(nn.Module):
    def __init__(self, units, input_dim, output_dim, num_layers, epsilon, dt, device, steps=100):
        super(TransNetSweeping, self).__init__()
        self.num_layers = num_layers
        self.units = units
        self.epsilon = epsilon
        self.dt = dt
        self.tol = 1e-4
        self.steps = steps
        self.device = device
        self.linearInput = LinearLayer(input_dim, units)
        self.blocks = torch.nn.ModuleList(
            [TransNetLayerSweeping(units, units, epsilon=epsilon, dt=dt, device=device) for _ in range(num_layers)])

        self.linearOutput = LinearLayer(units, output_dim)
        self.activation = nn.Tanh()
        self.batch_size = 0

    def forward(self, x):
        self.batch_size = x.size()[0]
        self.set_batch_size()

        x = self.linearInput(x)
        z_in = torch.cat((x, self.activation(x)), 1)

        # Source_Iteration
        with torch.no_grad():
            # 1) initialize state variables
            self.initialize_model(x)

            # 2) Setup system matrix
            self.setup_system_mats()

            step = 0
            total_err = 10
            while step < self.steps and total_err > self.tol:
                # 1) relax
                self.relax()

                # 2) sweep
                total_err = self.sweep(z_in)
                step += 1

            # print(total_err)
            # print(step)
            # print("-----")

        # Forward iteration for gradient tape
        z = self.implicit_forward(z_in)

        #  linear output
        z = z[:, :self.units]
        z = self.linearOutput(z)
        logits = nn.Softmax(dim=1)(z)
        return logits

    def initialize_model(self, x):
        for block in self.blocks:
            block.initialize_model(x)
        return 0

    def setup_system_mats(self):
        for block in self.blocks:
            block.setup_system_mat()
        return 0

    def relax(self):
        for block in self.blocks:
            block.relax()
        return 0

    def sweep(self, z_in):
        err = [0.0] * self.num_layers
        z = z_in
        i = 0
        for block in self.blocks:
            z, err[i] = block.sweep(z)
            i += 1

        return sum(err) / len(err)

    def implicit_forward(self, z_in):
        z = z_in
        for block in self.blocks:
            z = block.implicit_forward(z_in)
        return z

    def set_batch_size(self):
        for block in self.blocks:
            block.batch_size = self.batch_size
        return 0


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
        self.z_l = torch.eye(1)
        self.batch_size = 0

        self.w_t = torch.eye(self.out_features).to(self.device)

    @staticmethod
    def grad_activation(z):
        return 1.0 / torch.cosh(z) ** 2

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def initialize_model(self, input_x):
        """
        :param input_x: input data
        :return:
        """

        if self.z_l.size()[0] != input_x.size()[0]:
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

        self.w = self.weight.T
        return 0

    def relax(self):
        """
        :param z_in: [u,v]
        :return: constructs the rhs of the relaxation system.
                DONT USE WITH GRADIENT TAPE ACTIVE
        """
        # 1)  assemble right hand side
        zeros = torch.zeros(size=(self.z_l.size()[0], self.out_features), device=self.device)
        self.rhs = torch.cat(
            (zeros + self.dt * self.bias, self.dt / self.epsilon * self.activation(self.z_l[:, :self.out_features])), 1)

        return 0

    def sweep(self, z_lp1_i):
        """
            :param z_lp1_i: sweep solve of previous layer
            :return: Output y of implicit Layer Ay = x + f(x) + b
                    DONT USE WITH GRADIENT TAPE ACTIVE
        """
        A = self.A.repeat(self.z_l.shape[0], 1, 1).to(self.device)
        rhs = self.rhs[:self.batch_size, :] + z_lp1_i

        rhs = rhs[:, :, None]

        y = torch.solve(rhs, A)[0][:, :, 0]
        error = torch.mean(torch.linalg.norm(self.z_l - y, dim=1))
        self.z_l = y
        return self.z_l, error

    def implicit_forward(self, z_in):

        u_in = z_in[:, :self.out_features]
        v_in = z_in[:, self.out_features:]

        # a) u part
        u_out = - self.dt * torch.matmul(self.z_l[:, self.out_features:],
                                         self.weight.T) + u_in + self.dt * self.bias
        # b) v part
        v_out = self.dt * torch.matmul(self.z_l[:, :self.out_features],
                                       self.w.T) + v_in - self.dt / self.epsilon * (
                        self.z_l[:, self.out_features:] - self.activation(self.z_l[:, :self.out_features]))

        # Assemble layer solution vector
        z_out = torch.cat((u_out, v_out), 1)

        # 4) Use implicit function theorem (or adjoint equation of the KKT system to compute the real gradient)
        #   Let g = Ay - (x + f(x) + b) (layer in fixed point notation). Then grad = dg/dx.
        #   We need gradient dy/dx, using dg/dy*dy/dx =dg/dy

        # assemble Jacobian, i.e. dg/dy
        with torch.no_grad():
            b_prime = self.grad_activation(self.z_l[:, :self.out_features])[:, :,
                      None] * self.dt / self.epsilon * torch.eye(self.out_features, device=self.device)[None, :,
                                                       :]  # db/du
            J = self.A.repeat(self.z_l.shape[0], 1, 1)
            J[:, self.out_features:, :self.out_features] -= b_prime

        # register backward hook
        if z_out.requires_grad:
            z_out.register_hook(
                lambda grad: torch.solve(grad[:, :, None], J.transpose(1, 2))[0][:, :, 0])
        return z_out
