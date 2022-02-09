import torch
from torch import nn
from siren_pytorch import SirenNet
from .Base import PINN
from .Yu2021 import ResidualAdaptive

class Siren_PINN(PINN, ResidualAdaptive):
    def __init__(self, loss_lambda=(1.0, 1.0), **kwarg):
        super().__init__()
        
        self.loss_lambda = loss_lambda
        self.loss_function = nn.MSELoss(reduction='mean')
        self._loss_PDE = -1.
        self._loss_SDF = -1.

        siren_param = {}
        siren_option_names = ['dim_in', 'dim_out', 'dim_hidden', 'num_layers', 'final_activation']
        for option in siren_option_names:
            if option in kwarg:
                siren_param[option] = kwarg[option]
        self.model = SirenNet(**siren_param)

    def forward(self, x):
        return self.model(x).squeeze()

    def loss(self, y=None, residual_x=None, bc_x=None, bc_sdf=None, non_bc_points=None):
        self._loss = torch.tensor(0.0, device=y.device if y is not None else bc_x.device)
        if y is not None and residual_x is not None:
            self._loss += self.loss_lambda[0] * self.loss_PDE(y, residual_x)
        if bc_x is not None and bc_sdf is not None:
            self._loss += self.loss_lambda[1] * self.loss_SDF(bc_x, bc_sdf)
        # if non_bc_points is not None:
        #     f = self.forward(non_bc_points)
        #     self._loss += torch.exp(-f).sum()
        return self._loss

