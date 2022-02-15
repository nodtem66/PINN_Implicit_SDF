# -*- coding: utf-8 -*-
from typing import overload
import torch
from torch import nn
from torch.functional import Tensor

from utils.operator import gradient

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
    

class MLP_PINN(Davies2021, PINN):
    def __init__(self, loss_lambda={}, **kwarg):
        super().__init__(**kwarg)
        self.loss_lambda = loss_lambda

    def loss(self, x, sdf, residual_x=None):
        y = self(x)
        p = gradient(y, x)
        
        if residual_x is not None:
            residual_y = self(residual_x)
            residual_p = gradient(residual_y, residual_x)
        return self.loss_SDF(y, sdf) + 0.1 * self.loss_residual(residual_p if residual_x is not None else p)

    def loss_with_normal(self, x, sdf, grad, residual_x=None):
        y = self(x)
        p = gradient(y, x)
        
        if residual_x is not None:
            residual_y = self(residual_x)
            residual_p = gradient(residual_y, residual_x)

        return self.loss_SDF(y, sdf) + \
            self.loss_lambda.get('loss_normal', 1.0) * self.loss_normal(p, grad) + \
            self.loss_lambda.get('loss_residual', 1.0) * self.loss_residual(residual_p if residual_x is not None else p)
            #self.loss_lambda.get('loss_residual_constraint', 1.0) * self.loss_residual_constraint(residual_p if residual_x is not None else p) + \

    def loss_with_cosine_similarity(self, x, sdf, grad, residual_x=None):
        y = self(x)
        p = gradient(y, x)
        if residual_x is not None:
            residual_p = gradient(self(residual_x), residual_x)
        
        return self.loss_SDF(y, sdf) + \
            self.loss_lambda.get('loss_cosine_similarity', 1.0) * self.loss_cosine_similarity(p, grad) + \
            self.loss_lambda.get('loss_residual', 1.0) * self.loss_residual(residual_p if residual_x is not None else p)