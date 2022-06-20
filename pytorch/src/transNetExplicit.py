import torch
from torch import nn, Tensor
import math

from src.layers import LinearLayer


class TransNetExplicit(nn.Module):
    def __init__(self, units, input_dim, output_dim, num_layers, epsilon, dt, device, steps=50):
        super(TransNetExplicit, self).__init__()
        self.num_layers = num_layers
        self.units = units
        self.steps = steps
        self.linearInput = LinearLayer(input_dim, units)
        self.block1 = TransNetLayerExplict(units, units, epsilon=epsilon, dt=dt, device=device, steps=steps)
        self.block2 = TransNetLayerExplict(units, units, epsilon=epsilon, dt=dt, device=device, steps=steps)
        self.block3 = TransNetLayerExplict(units, units, epsilon=epsilon, dt=dt, device=device, steps=steps)
        self.block4 = TransNetLayerExplict(units, units, epsilon=epsilon, dt=dt, device=device, steps=steps)

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


class TransNetLayerExplict(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    # A:Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, epsilon=0.01, dt=0.1, device="cpu",
                 steps=20, dtype=None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = epsilon * torch.ones(1).to(device)
        self.dt = dt * torch.ones(1).to(device)
        self.device = device
        self.steps = steps
        self.step =0
        self.tol = 1e-4
        
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

        u_in = x[:, :self.out_features]
        v_in = x[:, self.out_features:]
        
        z = u_in
        
        self.step =0

        while self.step < self.steps:
            # Sweeping Step
            #u_star = u_in + self.dt * (torch.matmul(v_in, self.weight) + self.bias)
            #v_star = v_in - self.dt * torch.matmul(u_in, torch.transpose(self.weight, 0, 1))

            # Relaxation Step
            #u_out = u_star
            #v_out = 1.0 / (1 + self.dt / self.epsilon) * (
            #       v_star + self.dt / self.epsilon * self.activation(v_star))
            
            
            u_out = u_in + self.activation(torch.matmul(z,self.weight)+self.bias)*self.dt

            self.step+=1
            err =  torch.norm(u_out-z) #+ torch.norm(v_out-v_in)
            if err <self.tol:
                break
            
            z = u_out
            #v_in = v_out
        v_out = v_in
        #print(err)
        #print("------")
        # Assemble layer solution vector
        y = torch.cat((u_out, v_out), 1)

        return y
