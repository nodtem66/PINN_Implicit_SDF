from math import floor
import torch
from .Base import PINN
from .Wang2020 import LambdaAdaptive, M4, M4_1, M2, M2_1
from .MLP import Davies2021, MLP_PINN
from utils.helpers import gradient

# Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems
# Jeremy Yu, Lu Lu, Xuhui Meng, George Em Karniadakis
# The author proposed gPINN which add the new constrant of the gradient of residuals
class G_PINN(PINN):

    def loss_gradient_PDE(self, y, x):
        Fxx = gradient(gradient(y, x), x)
        Fxx_size = Fxx.size()
        zeros = torch.zeros_like(Fxx[:, 0], device=Fxx.device)
        self._loss_gradient_PDE = self.loss_function(Fxx[:, 0], zeros)
        for i in range(Fxx_size[1] - 1):
            self._loss_gradient_PDE += self.loss_function(Fxx[:, i+1], zeros)
        return self._loss_gradient_PDE

class LambdaAdaptive_G_PINN(LambdaAdaptive, G_PINN):

    def __init__(self, useRatioOfRatio=False, **kwarg):
        super().__init__(**kwarg)
        self.useRatioOfRatio = useRatioOfRatio
    
    def loss_gradient_PDE(self, y, x, adaptive_lambda=False):
        
        x_size = x.size()
        self._loss_gradient_PDE = torch.tensor(.0, device=x.device)
        self.mean_grad_bc2 = torch.zeros(x_size[1], device=x.device, requires_grad=False)
        zeros = torch.zeros(x_size[0], device=x.device, requires_grad=False)

        # calculate gradient of Fx
        if not adaptive_lambda:
            Fxx = self.get_gradient2(x)
        
            for i in range(x_size[1]):
                self._loss_gradient_PDE += self.loss_lambda[i+1] * self.loss_function(Fxx[:, i], zeros)
        else:
            # calculate loss from BC
            assert(hasattr(self, '_nn_layers'))

            for i in range(x_size[1]):

                if x.device.type == 'cuda':
                    torch.cuda.empty_cache()
                self.zero_grad()

                Fxx = self.get_gradient2(x)
                # get mean of loss gradient PDE
                _loss = self.loss_function(Fxx[:, i], zeros)
                _loss.backward()

                with torch.no_grad():
                    _gradient_bcs = torch.empty(len(self._nn_layers))
                    for j, layer in enumerate(self._nn_layers):
                        if layer.weight.grad is not None:
                            _gradient_bcs[j] = torch.mean(torch.abs(layer.weight.grad)).detach().clone()
                    self.mean_grad_bc2[i] = torch.mean(_gradient_bcs)
                self._loss_gradient_PDE += _loss.detach().clone()
        #endif adaptive_lambda 
        return self._loss_gradient_PDE

    def adaptive_lambda(self, y, pde_x, bc_x, bc_sdf):
        self.loss_PDE(y, pde_x, adaptive_lambda=True)
        self.loss_SDF(bc_x, bc_sdf, adaptive_lambda=True)
        self.loss_gradient_PDE(y, pde_x, adaptive_lambda=True)
        
        with torch.no_grad():
            # Prevent zero dividing error
            self.mean_grad_bc1 += 1e-9
            self.mean_grad_bc2 += 1e-9
            self._loss_PDE += 1e-9
            self._loss_SDF += 1e-9
            self._loss_gradient_PDE += 1e-9

            self.adjust_lambda()

    def adjust_lambda(self):
        # Adjust for lambda BC1
        new_lambda = (self.max_grad_residual/self._loss_PDE) / (self.mean_grad_bc1/self._loss_SDF) if self.useRatioOfRatio else (self.max_grad_residual / self.mean_grad_bc1)
        self.loss_lambda[0] = (self.alpha) * self.loss_lambda[0] + (1.0 - self.alpha) * new_lambda

        # Adjust for lambda BC2
        for i in range(self.loss_lambda.size()[0] - 1):    
            new_lambda = (self.max_grad_residual/self._loss_PDE) / (self.mean_grad_bc2[i]/self._loss_gradient_PDE) if self.useRatioOfRatio else (self.max_grad_residual / self.mean_grad_bc2[i])
            self.loss_lambda[i+1] = (self.alpha) * self.loss_lambda[i+1] + (1.0 - self.alpha) * new_lambda
    
    def loss(self, y, pde_x, bc_x, bc_sdf):
        
        self._loss = self.loss_PDE(y, pde_x)
        self._loss += self.loss_lambda[0] * self.loss_SDF(bc_x, bc_sdf)
        self._loss += self.loss_gradient_PDE(y, pde_x)
            
        return self._loss

# Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems
# Jeremy Yu, Lu Lu, Xuhui Meng, George Em Karniadakis
# The author proposed residual-based adaptive refinement (RAR)
class ResidualAdaptive():
    def __init__(self, **kwargs):
        super().__init__()

    def adjust_samples_from_residual(self, y, x, num_samples=100, variance=0.1):
        Fx = gradient(y, x)
        with torch.no_grad():
            Fx = (torch.linalg.norm(Fx, dim=1) - 1) ** 2
            idx = torch.multinomial(Fx, num_samples, replacement=True)
            return (x[idx] + torch.randn_like(x[idx], device=idx.device) * variance)

class MLP_GPINN(Davies2021, G_PINN):
    def __init__(self, loss_lambda=(1.0, 1.0, 1.0), **kwarg):
        super().__init__(**kwarg)
        self.loss_lambda = loss_lambda
    
    def loss(self, y, pde_x, bc_x, bc_sdf):
        self._loss = self.loss_lambda[0] * self.loss_PDE(y, pde_x)
        self._loss += self.loss_lambda[1] * self.loss_SDF(bc_x, bc_sdf)
        self._loss += self.loss_lambda[2] * self.loss_gradient_PDE(y, pde_x)
        return self._loss

class MLP_PINN_RAR(MLP_PINN, ResidualAdaptive):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

class MLP_GPINN_RAR(MLP_GPINN, ResidualAdaptive):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

class MLP_GPINN_LambdaAdaptive(LambdaAdaptive_G_PINN, Davies2021):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self._add_nn_layers(self.model)

class M2_RAR(M2, ResidualAdaptive):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

class M2_1_RAR(M2_1, ResidualAdaptive):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

class M4_RAR(M4, ResidualAdaptive):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

class M4_1_GPINN(LambdaAdaptive_G_PINN, M4_1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class M4_1_RAR(M4_1, ResidualAdaptive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class M4_1_GPINN_RAR(M4_1_GPINN, ResidualAdaptive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)