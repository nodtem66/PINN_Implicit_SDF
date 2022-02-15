import torch
from torch import nn
from utils.operator import gradient

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
        sdf_predict = self(x)
        return nn.MSELoss()(sdf_predict, true_sdf) # relative L2 norm of the error

    def test_norm_gradient(self, x, true_norm_grad):
        x.requires_grad_(True)
        y = self(x)
        norm_grad = torch.linalg.norm(gradient(y, x, create_graph=False), dim=1)
        x.requires_grad_(False)
        with torch.no_grad():
            return nn.MSELoss()(norm_grad, true_norm_grad)

    def test_residual(self, x):
        x.requires_grad_(True)
        y = self(x)
        norm_grad = torch.linalg.norm(gradient(y, x, create_graph=False), dim=1)
        x.requires_grad_(False)
        with torch.no_grad():
            return torch.mean((norm_grad - 1).abs())


# Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
# Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis
class PINN(Base):

    def loss_residual(self, p):
        """
        Calculate residual from gradients, :attr:`p`

        Args:
        - :attr:`p`: tensor of gradient

        Example:
        ```
        y = model(x)
        p = gradient(y, x)
        model.loss_residual(p)
        ```
        """
        norm_p = torch.linalg.norm(p, dim=1)
        self._loss_residual = torch.mean((norm_p - 1)**2)
        return self._loss_residual

    def loss_residual_constraint(self, p):
        """
        Calculate loss from gradient, :attr:`p`

        `ReLU(norm(p) - 1)`


        Args:
        - :attr:`p`: tensor of gradient

        Example:
        ```
        y = model(x)
        p = gradient(y, x)
        model.loss_residual_constraint(p)
        ```
        """
        norm_p = torch.linalg.norm(p, dim=1)
        self._loss_residual_constraint = torch.mean(torch.nn.ReLU()(norm_p - 1))
        return self._loss_residual_constraint

    def loss_cosine_similarity(self, p, grad):
        """
        Calculate loss from gradient of model (:attr:`p`) and training data (:attr:`grad`)

        `torch.dot(p,grad)/(norm(p)*norm(grad))`


        Args:
        - :attr:`p`: tensor of gradient
        - :attr:`grad`: tensor of target gradient

        Example:
        ```
        y = model(x)
        p = gradient(y, x)
        model.loss_cosine_similarity(p, grad)
        ```
        """
        norm_p = torch.linalg.norm(p, dim=1)
        norm_g = torch.linalg.norm(grad, dim=1)
        self._loss_cosine_similarity = torch.mean(-torch.einsum('ij,ij->i', p, grad)/norm_p/norm_g)
        return self._loss_cosine_similarity

    def loss_SDF(self, y, sdf):
        """
        Calculate loss from predicted SDF from model (:attr:`y`)
        and SDF from training data (:attr:`sdf`)

        `MSE(y, sdf)`


        Args:
        - :attr:`y`: predicted SDF
        - :attr:`sdf`: target SDF

        Example:
        ```
        y = model(x)
        model.loss_SDF(y, sdf)
        ```
        """
        self._loss_SDF = torch.nn.MSELoss()(y, sdf)
        return self._loss_SDF

    def loss_normal(self, p, grad):
        """
        Calculate loss from gradient of model (:attr:`p`) and training data (:attr:`grad`)

        `MSE(p, (grad / norm(grad)))`

        Args:
        - :attr:`p`: predicted gradient
        - :attr:`grad`: target gradient

        Example:
        ```
        y = model(x)
        p = gradient(y, x)
        model.loss_normal(p, grad)
        ```
        """
        norm_grad = torch.linalg.norm(grad, dim=1)
        normal = grad / norm_grad
        self._loss_normal = torch.nn.MSELoss()(p, normal)
        return self._loss_normal

    def print_loss(self, verbose=False) -> None:
        keys = ['_loss_SDF', '_loss_residual', '_loss_residual_constraint', '_loss_normal', '_loss_cosine_similarity']
        _loss_str = 'Loss: ' 
        for key in keys:
            if hasattr(self, key):
                _loss_str += f'{getattr(self, key):.6f} '
            else:
                _loss_str += 'na '
        if verbose:
            print(_loss_str)
        return _loss_str