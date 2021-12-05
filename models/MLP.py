# -*- coding: utf-8 -*-
from typing import overload
import torch
from torch import nn
from torch.functional import Tensor

from .Base import Base, PINN

# On the Effectiveness of Weight-Encoded Neural Implicit 3D Shapes
# Thomas Davies, Derek Nowrouzezahrai, Alec Jacobson
# The author used MLP with ReLU and tanh to encode implicit 3D shapes
# see: https://github.com/u2ni/ICML2021/blob/main/neuralImplicitTools/src/model.py
class Davies2021(Base):
    def __init__(self, width=200, N_layers=10, activation=torch.nn.ReLU(), last_activation=torch.nn.Tanh(), **kwarg):
        super().__init__()
        _layers = [nn.Linear(3, width), activation]
        for i in range(N_layers):
            _layers.append(nn.Linear(width, width))
            _layers.append(activation)
        _layers.append(nn.Linear(width, 1))
        _layers.append(last_activation)
        self.model = nn.Sequential(*_layers)
        
        self._init_gains = ['relu', 'tanh']
        self.model.apply(self.init_xavier_normal_weights)

        self.loss_function = nn.MSELoss(reduction='mean')
    
    def forward(self, x):
        return torch.squeeze(self.model(x))

    def loss(self, x, sdf):
        self._loss = self.loss_function(self.forward(x), sdf)
        return self._loss
    

class MLP_PINN(Davies2021, PINN):
    def __init__(self, loss_lambda=(1.0, 1.0), **kwarg):
        super().__init__(**kwarg)
        self.loss_lambda = torch.tensor(loss_lambda, requires_grad=False)

    def loss(self, y, residual_x, bc_x, bc_sdf):
        self._loss = self.loss_lambda[0] * self.loss_PDE(y, residual_x)
        self._loss += self.loss_lambda[1] * self.loss_SDF(bc_x, bc_sdf)
        return self._loss
