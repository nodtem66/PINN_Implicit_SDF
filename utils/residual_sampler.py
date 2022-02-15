import math
import torch
import numpy as np
from scipy.stats import qmc
from .operator import gradient

@torch.no_grad()
def bounding_box(x):
    _min = torch.min(x, dim=0)
    _max = torch.max(x, dim=0)
    return (_min[0], _max[0])

def generate_grid_points(bounds=((-1,-1,-1), (1, 1, 1)), nx=1000, ny=1000, nz=1000, device='cpu'):
    (x0, y0, z0), (x1, y1, z1) = bounds

    X = torch.linspace(x0, x1, nx, device=device)
    Y = torch.linspace(y0, y1, ny, device=device)
    Z = torch.linspace(z0, z1, nz, device=device)

    P = torch.cartesian_prod(X, Y, Z)
    return P

def residual_sum_pdf(net, x, epsilon=1e-4):
    ndim = round(x.shape[0] ** (1/3))
    _min, _max = bounding_box(x)
    dx = (_max - _min) / (ndim-1)
    prod_dx = torch.prod(dx)
    x.requires_grad_(True)
    if x.grad is not None:
        x.grad.zero_()
    y = net.forward(x)
    p = gradient(y, x, False)
    with torch.no_grad():
        norm_p = torch.linalg.norm(p, dim=1)
        residual = (norm_p - 1.0)**2
        pdf = torch.log(residual / epsilon)
        mark = pdf < 0
        pdf[mark] = 0
        return (pdf.sum() * prod_dx)

def residual_likelihood(net, x, sum_pdf=1.0, epsilon=1e-4):
    x.requires_grad_(True)
    if x.grad is not None:
        x.grad.zero_()
    y = net.forward(x)
    p = gradient(y, x, False)
    x.requires_grad_(False)
    with torch.no_grad():
        norm_p = torch.linalg.norm(p, dim=1)
        residual = (norm_p - 1.0)**2
        pdf = torch.log(residual / epsilon)
        mark = pdf < 0
        pdf[mark] = 0
        return pdf / sum_pdf

def random_points_from_residual(net, num_points=1000, grid_points=None):
    assert(grid_points is not None)
    assert(len(grid_points.shape) == 2)
    assert(grid_points.shape[1] == 3)

    device = grid_points.device
    with torch.no_grad():
        _min, _max = bounding_box(grid_points)
        _scale = _max - _min
        _num_randoms = round(num_points*2)
        x = torch.rand((_num_randoms,), device=device)*_scale[0] + _min[0]
        y = torch.rand((_num_randoms,), device=device)*_scale[1] + _min[1]
        z = torch.rand((_num_randoms,), device=device)*_scale[2] + _min[2]
        points = torch.vstack((x,y,z)).T

    sum_pdf = residual_sum_pdf(net, grid_points)
    likelihood = residual_likelihood(net, points, sum_pdf=sum_pdf)
    mask = torch.rand((_num_randoms,), device=device) > likelihood
    points = points[mask]
    return points[:num_points]

class ResidualSampler():
    def __init__(self, model: torch.nn.Module, l_bounds=(-1, -1, -1), u_bounds=(1, 1, 1), num_grid_points=100*100*100, device='cpu'):
        # Variables in numpy 
        self.sampler = qmc.Sobol(d=3, scramble=False)
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds
        self.num_grid_points = 2 ** math.ceil(math.log2(num_grid_points))
        # Variables in torch
        self.model = model
        self.device =device

    def set_bounds_from_points(self, points: torch.Tensor) -> None:
        self.device = points.device
        _min, _max = bounding_box(points)
        self.l_bounds = _min.cpu().detach().numpy()
        self.u_bounds = _max.cpu().detach().numpy()
        self.sampler.reset()
    
    def expand_bounds(self, scale_offset:float=0.1) -> None:
        radius_offset = 0.5 * np.abs(self.u_bounds - self.l_bounds) * scale_offset 
        self.l_bounds -= radius_offset
        self.u_bounds += radius_offset
        self.sampler.reset()

    def random(self, n=100) -> torch.Tensor:
        grid_points = self.sampler.random(n=self.num_grid_points)
        grid_points = qmc.scale(grid_points, l_bounds=self.l_bounds, u_bounds=self.u_bounds)
        grid_points = torch.from_numpy(grid_points.astype(np.float32)).to(self.device)
        return random_points_from_residual(self.model, num_points=n, grid_points=grid_points)

    def append_random_totensor(self, source: torch.Tensor, n=100) -> torch.Tensor:
        points = self.random(n=n)
        with torch.no_grad():
            return torch.cat((source, points), 0)

    def __str__(self):
        return f'ResidualSampler(l_bounds={self.l_bounds}, u_bounds={self.u_bounds}, num_grid_points={self.num_grid_points})'