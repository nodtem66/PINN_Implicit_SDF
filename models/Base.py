import torch
from torch import nn
from torch.nn.init import calculate_gain

def _calculate_gain(gain: any, index=0):
    if isinstance(gain, str):
        gain = calculate_gain(gain)
    elif isinstance(gain, list):
        assert(index >= 0 and index <= 2)
        if len(gain) == 3:
            gain = _calculate_gain(gain[index])
        elif len(gain) == 2:
            gain = _calculate_gain(gain[index-1 if index > 0 else 0])
        elif len(gain) == 1:
            gain = _calculate_gain(gain[0])
    return gain

class Base(nn.Module):
    
    @torch.no_grad()
    def init_xavier_normal_weights(self, m):
        if isinstance(m, nn.Linear):
            gain = 1
            if hasattr(self, '_init_gains'):
                index = 1
                if m.in_features == 3:
                    index = 0
                elif m.out_features == 1:
                    index = 2    
                gain = _calculate_gain(self._init_gains, index)
            nn.init.xavier_normal_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def get_gradient(self, x):
        _x = x.clone()
        _x.requires_grad = True
        F = self.forward(_x)
        F_x = torch.autograd.grad(
            F, _x,
            grad_outputs=torch.ones_like(F),
            create_graph=True, 
        )[0]
        return torch.linalg.norm(F_x, dim=1)


    def get_gradient2(self, x):
        _x = x.clone()
        _x.requires_grad = True
        F = self.forward(_x)
        Fx = torch.autograd.grad(
            F, _x,
            grad_outputs=torch.ones_like(F),
            create_graph=True, # the graph'll be created for higher-order derivatives
        )[0]
        Fx_norm = torch.linalg.norm(Fx, dim=1)
        Fxx = torch.autograd.grad(
            Fx_norm, _x,
            grad_outputs=torch.ones_like(F),
            create_graph=True,
        )[0]
        #Fxx_norm = torch.linalg.norm(Fxx, dim=1)
        return Fxx
    
    @torch.no_grad()
    def test(self, x, true_sdf):
        sdf_predict = self.forward(x)
        errors = self.loss_function(sdf_predict, true_sdf) # relative L2 norm of the error
        
        return errors

# Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
# Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis
class PINN(Base):

    def loss_SDF(self, x, sdf):
        self._loss_SDF = self.loss_function(self.forward(x), sdf)
        return self._loss_SDF
    
    def loss_PDE(self, x):
        Fx = self.get_gradient(x)
        self._loss_PDE = self.loss_function(Fx, torch.ones_like(Fx))
        return self._loss_PDE