from typing import Iterable
import torch
from torch import nn
from .MLP import Davies2021 as MLP
from .Base import PINN, Base, linear_layer_with_init
from utils.helpers import gradient

# Understanding and mitigating gradient pathologies in physics-informed neural networks
# Sifan Wang, Yujun Teng, Paris Perdikaris
# The author compared 4 models M1 - M4: the best one is M4 which is an adaptive boundary residue and uses a new architecture network.
# original source code: https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py
class LambdaAdaptive(PINN):
    def __init__(self, alpha=0.9, loss_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_lambda = loss_lambda
        self.alpha = alpha
        self._nn_layers = []

    def _add_nn_layers(self, model):
        if isinstance(model, Iterable):
            for layer in model:
                if isinstance(layer, nn.Linear):
                    self._nn_layers.append(layer)

    def loss_PDE(self, x, grad, adaptive_lambda=False):
        # calculate loss from residual points
        if adaptive_lambda:
            assert(hasattr(self, '_nn_layers'))
            self.zero_grad()
        y = self.forward(x)
        p = gradient(y, x)
        norm_p = torch.linalg.norm(p, dim=1)
        self._loss_PDE = torch.mean((norm_p - 1)**2)
        self._loss_grad = self.loss_function(p, grad)
        
        # get max of residual gradients
        if adaptive_lambda:
            self._loss_PDE.backward()
            with torch.no_grad():
                gradient_res = torch.empty(len(self._nn_layers))
                for i, layer in enumerate(self._nn_layers):
                    if layer.weight.grad is not None:
                        gradient_res[i] = torch.max(torch.abs(layer.weight.grad)).detach().clone()
                self.max_grad_residual = torch.max(gradient_res)
        return (self._loss_grad, self._loss_PDE)

    def loss_SDF(self, x, sdf, adaptive_lambda=False):
        # calculate loss from BC
        if adaptive_lambda:
            assert(hasattr(self, '_nn_layers'))
            self.zero_grad()
        y = self.forward(x)
        self._loss_SDF = self.loss_function(y, sdf)
        # get mean of BC gradients
        if adaptive_lambda:
            self._loss_SDF.backward()
            with torch.no_grad():
                gradient_bcs = torch.empty(len(self._nn_layers))
                for i, layer in enumerate(self._nn_layers):
                    if layer.weight.grad is not None:
                        gradient_bcs[i] = torch.mean(torch.abs(layer.weight.grad)).detach().clone()
                self.mean_grad_bc1 = torch.mean(gradient_bcs)

        return self._loss_SDF
        

    def adaptive_lambda(self, x, sdf, grad):
        self.loss_PDE(x, grad, adaptive_lambda=True)
        self.loss_SDF(x, sdf, adaptive_lambda=True)
        with torch.no_grad():
            self.mean_grad_bc1 += 1e-9
            self.loss_lambda = (self.alpha) * self.loss_lambda + (1.0 - self.alpha) * (self.max_grad_residual / self.mean_grad_bc1)


class M2(MLP, LambdaAdaptive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._add_nn_layers(self.model)

    def loss(self,  x, sdf, grad, adaptive_lambda=False):
        loss_grad, loss_PDE = self.loss_PDE(x, grad, adaptive_lambda)
        self._loss =  self.loss_lambda * (self.loss_SDF(x, sdf, adaptive_lambda) + loss_grad) + loss_PDE
        return self._loss

class M4(Base):

    def __init__(self, width=32, N_layers=8, activation=torch.nn.Softplus(30), last_activation=torch.nn.Softplus(30), **kwarg):
        super().__init__(**kwarg)
        
        self.width = width
        self.N_layers = N_layers
        
        self.u = linear_layer_with_init(3, width, activation=activation)
        self.v = linear_layer_with_init(3, width, activation=activation)
        self.h = linear_layer_with_init(3, width, activation=activation)

        self.h_list = []
        for i in range(N_layers):
            self.h_list.append(linear_layer_with_init(width, width, activation=activation))
            self.add_module(f'h{i}', self.h_list[i])
        
        self.last_h = linear_layer_with_init(width, 1, activation=last_activation)
        self.Φ = activation
        self.last_Φ = last_activation

    def forward(self, input):
        _u = self.Φ(self.u(input))
        _v = self.Φ(self.v(input))
        h = self.Φ(self.h(input))
        h = h * _u + (torch.ones_like(h) - h) * _v
        for i in range(self.N_layers):
            h = self.Φ(self.h_list[i](h))
            h = h * _u + (torch.ones_like(h)-h) * _v
        return torch.squeeze(self.Φ(self.last_h(h)))