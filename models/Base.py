from numpy import true_divide
import torch
from torch import nn
from torch.nn.init import calculate_gain
from utils import gradient
import math

def activation_name(activation: nn.Module) -> str:
    if activation is nn.Tanh:
        return 'tanh'
    elif activation is nn.ReLU or activation is nn.ELU or activation is nn.GELU:
        return 'relu'
    elif activation is nn.SELU:
        return 'selu'
    elif activation is nn.LeakyReLU:
        return 'leaky_relu'
    elif activation is nn.Sigmoid:
        return 'sigmoid'
    return 'linear'

def linear_layer_with_init(width, height, init=nn.init.xavier_uniform_, activation=None) -> nn.Linear:
    linear = nn.Linear(width, height)
    if init is None or activation is None:
        return linear
    init(linear.weight, gain=nn.init.calculate_gain(activation_name(activation)))
    return linear

class Base(nn.Module):
    
    @torch.no_grad()
    def test(self, x, true_sdf):
        sdf_predict = self.forward(x)
        errors = self.loss_function(sdf_predict, true_sdf) # relative L2 norm of the error
        
        return errors

    def test_gradient(self, x, true_gradient):
        x.requires_grad_(True)
        y = self.forward(x)
        Fx = torch.linalg.norm(gradient(y, x, create_graph=False), dim=1)
        with torch.no_grad():
            errors = self.loss_function(Fx, true_gradient)
            return errors

    def test_residual(self, x):
        x.requires_grad_(True)
        y = self.forward(x)
        with torch.no_grad():
            norm_grad = torch.linalg.norm(gradient(y, x, create_graph=False), dim=1)
            return torch.mean((norm_grad - 1).abs())


# Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
# Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis
class PINN(Base):

    def loss_PDE(self, x, grad=None, loss_lambda=0.1):
        y = self.forward(x)
        p = gradient(y, x)
        norm_p = torch.linalg.norm(p, dim=1)
        self._loss_residual = torch.mean((norm_p - 1)**2)
        if grad is not None:
            self._loss_grad = self.loss_function(p, grad)
            self._loss_PDE = self._loss_grad + loss_lambda * self._loss_residual
        else:
            self._loss_PDE = loss_lambda * self._loss_residual
        return self._loss_PDE

    def loss_SDF(self, x, sdf):
        y = self.forward(x)
        self._loss_SDF = self.loss_function(y, sdf)
        return self._loss_SDF

# Neural Tangent Kernel in PyTorch
# https://github.com/bobby-he/Neural_Tangent_Kernel/blob/master/src/NTK_net.py
class LinearNeuralTangentKernel(nn.Linear): 
    
    def __init__(self, in_features, out_features, bias=True, beta=1.0, w_sig = 1):
        self.beta = beta
        super().__init__(in_features, out_features)
        self.reset_parameters()
        self.w_sig = w_sig
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.w_sig * self.weight/math.sqrt(self.in_features), self.beta * self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta
        )