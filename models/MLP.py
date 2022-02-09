# -*- coding: utf-8 -*-
from typing import overload
import torch
from torch import nn
from torch.functional import Tensor

from utils.helpers import gradient

from .Base import Base, PINN, linear_layer_with_init

# On the Effectiveness of Weight-Encoded Neural Implicit 3D Shapes
# Thomas Davies, Derek Nowrouzezahrai, Alec Jacobson
# The author used MLP with ReLU and tanh to encode implicit 3D shapes
# see: https://github.com/u2ni/ICML2021/blob/main/neuralImplicitTools/src/model.py
class Davies2021(Base):
    def __init__(
        self,
        width=32, N_layers=8,
        in_feature=3, out_feature=1,
        activation=torch.nn.ReLU(), last_activation=torch.nn.Identity(),
        **kwarg
    ):
        super().__init__(**kwarg)
        _layers = [linear_layer_with_init(in_feature, width, activation=activation), activation]
        #_layers = [nn.Linear(in_feature, width), activation]
        for i in range(N_layers):
            not_last = i < N_layers - 1
            _layers.append(linear_layer_with_init(width, width, activation=activation if not_last else last_activation))
            _layers.append(activation if not_last else last_activation)
        _layers.append(nn.Linear(width, out_feature))
        self.model = nn.Sequential(*_layers)
        
        self.loss_function = nn.MSELoss(reduction='mean')
    
    def forward(self, x):
        return torch.squeeze(self.model(x))

    def loss(self, x, sdf):
        self._loss = self.loss_function(self.forward(x), sdf)
        return self._loss
    

class MLP_PINN(Davies2021):
    def __init__(self, loss_lambda=0.1, **kwarg):
        super().__init__(**kwarg)
        self.loss_lambda = loss_lambda

    def loss_PDE(self, x, grad):
        y = self.forward(x)
        p = gradient(y, x)
        norm_p = torch.linalg.norm(p, dim=1)
        self._loss_grad = self.loss_function(p, grad)
        self._loss_residual =  torch.mean((norm_p - 1)**2)
        self._loss_PDE = self._loss_grad + self.loss_lambda * self._loss_residual
        return self._loss_PDE

    def loss_SDF(self, x, sdf):
        y = self.forward(x)
        self._loss_SDF = self.loss_function(y, sdf)
        return self._loss_SDF

    def loss(self, x, sdf, grad):
        #self._loss = self.loss_lambda[0] * self.loss_PDE(y, x)
        #self._loss = self.loss_lambda[1] * self.loss_SDF(y, sdf)
        #self._loss = self.loss_SDF(y, sdf)
        self._loss = self.loss_SDF(x, sdf) + self.loss_PDE(x, grad)
        return self._loss

class MLP_PINN_dot(MLP_PINN):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def loss_PDE(self, x, grad):
        y = self.forward(x)
        p = gradient(y, x)
        norm_p = torch.linalg.norm(p, dim=1)
        norm_g = torch.linalg.norm(grad, dim=1)
        self._loss_dot = torch.mean(-torch.einsum('ij,ij->i', p, grad)/norm_p/norm_g)
        self._loss_residual =  torch.mean((norm_p - 1)**2)
        self._loss_PDE = self._loss_dot + self.loss_lambda * self._loss_residual
        return self._loss_PDE