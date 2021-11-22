from typing import Iterable
import torch
from torch import nn
from .MLP import Davies2021 as MLP
from .Base import PINN

# Understanding and mitigating gradient pathologies in physics-informed neural networks
# Sifan Wang, Yujun Teng, Paris Perdikaris
# The author compared 4 models M1 - M4: the best one is M4 which is an adaptive boundary residue and uses a new architecture network.
# original source code: https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py
class LambdaAdaptive(PINN):
    def __init__(self, alpha=0.9, **kwargs):
        super().__init__(**kwargs)
        self.loss_lambda = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=False)
        self.alpha = alpha
        self._nn_layers = []

    def _add_nn_layers(self, model):
        if isinstance(model, Iterable):
            for layer in model:
                if isinstance(layer, nn.Linear):
                    self._nn_layers.append(layer)

    def loss_PDE(self, x, adaptive_lambda=False):
        # calculate loss from residual points
        if adaptive_lambda:
            assert(hasattr(self, '_nn_layers'))
            self.zero_grad()
        Fx = self.get_gradient(x)
        self._loss_PDE = self.loss_function(Fx, torch.ones_like(Fx))
        
        # get max of residual gradients
        if adaptive_lambda:
            self._loss_PDE.backward()
            with torch.no_grad():
                gradient_res = torch.empty(len(self._nn_layers))
                for i, layer in enumerate(self._nn_layers):
                    if layer.weight.grad is not None:
                        gradient_res[i] = torch.max(torch.abs(layer.weight.grad)).detach().clone()
                self.max_grad_residual = torch.max(gradient_res)
        return self._loss_PDE

    def loss_SDF(self, x, sdf, adaptive_lambda=False):
        # calculate loss from BC
        if adaptive_lambda:
            assert(hasattr(self, '_nn_layers'))
            self.zero_grad()
        self._loss_SDF = self.loss_function(self.forward(x), sdf)
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
        
    def loss(self, pde_x, bc_x, bc_sdf, neumann_x=None, adaptive_lambda=False):
        self._loss = self.loss_PDE(pde_x, adaptive_lambda)
        self._loss += self.loss_lambda[0] * self.loss_SDF(bc_x, bc_sdf, adaptive_lambda)
        return self._loss

    def adaptive_lambda(self, pde_x, bc_x, bc_sdf):
        self.loss_PDE(pde_x, adaptive_lambda=True)
        self.loss_SDF(bc_x, bc_sdf, adaptive_lambda=True)
        with torch.no_grad():
            self.loss_lambda[0] = (self.alpha) * self.loss_lambda[0] + (1.0 - self.alpha) * (self.max_grad_residual / self.mean_grad_bc1)


class M2(MLP, LambdaAdaptive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._add_nn_layers(self.model)

    def loss(self, pde_x, bc_x, bc_sdf, adaptive_lambda=False):
        
        self._loss = self.loss_PDE(pde_x, adaptive_lambda)
        self._loss += self.loss_lambda[0] * self.loss_SDF(bc_x, bc_sdf, adaptive_lambda)
        
        return self._loss
    
    
class M2_1(M2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def adaptive_lambda(self, pde_x, bc_x, bc_sdf):
        self.loss_PDE(pde_x, adaptive_lambda=True)
        self.loss_SDF(bc_x, bc_sdf, adaptive_lambda=True)
        with torch.no_grad():
            new_lambda = (self.max_grad_residual/self._loss_PDE) / (self.mean_grad_bc1/self._loss_SDF)
            self.loss_lambda[0] = (self.alpha) * self.loss_lambda[0] + (1.0 - self.alpha) * new_lambda

# Physics-informed neural networks combining the proposed learning rate
# annealing algorithm and fully-connected architecture
# https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/93e752b0e3b541818d5cca49b681f4957bc36808/Helmholtz/Helmholtz2D_model_tf.py#L260
class M4(LambdaAdaptive):
    
    def __init__(self, width=32, N_layers=8, activation=torch.nn.Tanh(), last_activation=torch.nn.Tanh(), **kwarg):
        super().__init__(**kwarg)
        self._nn_layers = nn.ModuleList([nn.Linear(3, width), nn.Linear(3, width), nn.Linear(3, width)])
        for i in range(N_layers):
            self._nn_layers.append(nn.Linear(width, width))
        self._nn_layers.append(nn.Linear(width, 1))

        self.width = width
        self.N_layers = N_layers
        self.Φ = activation
        self.last_Φ = last_activation
        self.loss_function = nn.MSELoss(reduction='mean')

    def forward(self, input):
        assert(len(self._nn_layers) == self.N_layers + 4)
        self._u = self.Φ(self._nn_layers[0](input))
        self._v = self.Φ(self._nn_layers[1](input))
        h = self.Φ(self._nn_layers[2](input))
        h = h * self._u + (1-h) * self._v
        for i in range(self.N_layers):
            h = self.Φ(self._nn_layers[3+i](h))
            h = h * self._u + (1-h) * self._v
        return torch.squeeze(self.last_Φ(self._nn_layers[self.N_layers + 3](h)))

class M4_1(M4):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
    
    def adaptive_lambda(self, pde_x, bc_x, bc_sdf):
        self.loss_PDE(pde_x, adaptive_lambda=True)
        self.loss_SDF(bc_x, bc_sdf, adaptive_lambda=True)
        with torch.no_grad():
            new_lambda = (self.max_grad_residual/self._loss_PDE) / (self.mean_grad_bc1/self._loss_SDF)
            self.loss_lambda[0] = (self.alpha) * self.loss_lambda[0] + (1.0 - self.alpha) * new_lambda