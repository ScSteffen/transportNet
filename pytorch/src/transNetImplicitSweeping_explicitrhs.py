import torch
from torch import nn, Tensor
import math

from src.layers import LinearLayer


class TransNetSweepingExplRhs(nn.Module):
    def __init__(self, units, input_dim, output_dim, num_layers, epsilon, dt, device, steps=40):
        super(TransNetSweepingExplRhs, self).__init__()
        self.num_layers = num_layers
        self.units = units
        self.epsilon = epsilon
        self.dt = dt
        self.tol = 1e-4
        self.steps = steps
        self.device = device
        self.linearInput = LinearLayer(input_dim, units)
        self.block1 = TransNetLayerSweepingExplRhs(units, units, epsilon=epsilon, dt=dt, device=device)
        self.block2 = TransNetLayerSweepingExplRhs(units, units, epsilon=epsilon, dt=dt, device=device)
        self.block3 = TransNetLayerSweepingExplRhs(units, units, epsilon=epsilon, dt=dt, device=device)
        self.block4 = TransNetLayerSweepingExplRhs(units, units, epsilon=epsilon, dt=dt, device=device)

        self.linearOutput = LinearLayer(units, output_dim)
        self.batch_size = 0

    def forward(self, x):
        self.batch_size = x.size()[0]
        self.set_batch_size()

        x = self.linearInput(x)
        z_in = torch.cat((x, self.block1.activation(x)), 1)

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
                self.relax(z_in)

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
        # x = self.linearInput(x)

        self.block1.initialize_model(x)
        self.block2.initialize_model(x)
        # self.block3.initialize_model(x)
        # self.block4.initialize_model(x)
        return 0

    def setup_system_mats(self):
        self.block1.setup_system_mat()
        self.block2.setup_system_mat()
        # self.block3.setup_system_mat()
        # self.block4.setup_system_mat()
        return 0

    def relax(self, z_in):
        self.block1.relax(z_in)
        self.block2.relax(z_in)
        # self.block3.relax()
        # self.block4.relax()
        return 0

    def sweep(self, z_in):
        z, err1 = self.block1.sweep(z_in)
        z, err2 = self.block2.sweep(z)
        # z, err3 = self.block3.sweep(z)
        # z, err4 = self.block4.sweep(z)
        total_err = 1. / 2. * (err1 + err2)  # + err3 + err4)
        return total_err

    def implicit_forward(self, z_in):
        z = self.block1.implicit_forward(z_in)
        z = self.block2.implicit_forward(z)
        # z = self.block3.implicit_forward(z)
        # z = self.block4.implicit_forward(z)
        return z

    def set_batch_size(self):
        self.block1.batch_size = self.batch_size
        self.block2.batch_size = self.batch_size
        # self.block3.batch_size = self.batch_size
        # self.block4.batch_size = self.batch_size


class TransNetLayerSweepingExplRhs(nn.Module):
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
        return 0

    def relax(self, z_lp1_i):
        """
        :param z_in: [u,v]
        :return: constructs the rhs of the relaxation system.
                DONT USE WITH GRADIENT TAPE ACTIVE
        """
        # 1)  assemble right hand side
        zeros = torch.zeros(size=(self.z_l.size()[0], self.out_features), device=self.device)
        self.rhs = torch.cat((zeros + self.dt * self.bias, self.dt / self.epsilon * self.activation(z_lp1_i)), 1)

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
                                       self.weight) + v_in - self.dt / self.epsilon * (
                        self.z_l[:, self.out_features:] - self.activation(u_in))

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
            # J[:, self.out_features:, :self.out_features] -= b_prime

        # register backward hook
        if z_out.requires_grad:
            z_out.register_hook(
                lambda grad: torch.solve(grad[:, :, None], J.transpose(1, 2))[0][:, :, 0])
        return z_out
