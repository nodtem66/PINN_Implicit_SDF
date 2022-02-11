# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import List, Tuple
import copy

# https://en.wikipedia.org/wiki/Spherical_coordinate_system
@torch.no_grad()
def cartesian_to_spherical(x):
    assert x.shape[1] == 3
    r = torch.norm(x, dim=1)
    norm_xy = torch.norm(x[:, 0:2], dim=1)
    theta = torch.atan2(norm_xy, x[:, 2])
    phi = torch.atan2(x[:, 1], x[:, 0])

    return torch.vstack((r, theta, phi)).transpose(dim0=0, dim1=1)


@torch.no_grad()
def spherical_to_cartesian(x):
    _x = x[:, 0] * torch.cos(x[:, 2]) * torch.sin(x[:, 1])
    _y = x[:, 0] * torch.sin(x[:, 2]) * torch.sin(x[:, 1])
    _z = x[:, 0] * torch.cos(x[:, 1])
    return torch.vstack((_x, _y, _z)).transpose(dim0=0, dim1=1)


def laplace(y, x, create_graph=True):
    grad = gradient(y, x)
    return divergance(grad, x)


def divergance(y, x, create_graph=True):
    if not x.requires_grad:
        x.requires_grad_(True)
    if x.grad is not None:
        x.grad.zero_()
    div = 0.0
    for i in range(y.shape[-1]):
        if x.grad is not None:
            x.grad.zero_()
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=create_graph)[0][
            ..., i : i + 1
        ]
    return div


def gradient(y, x, create_graph=True):
    if x.grad is not None:
        x.grad.zero_()
    grad = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
    )[0]
    return grad


def jacobian(func, x, create_graph=False, vectorize=False):
    if x.grad is not None:
        x.grad.zero_()
    return torch.autograd.functional.jacobian(func, x, create_graph=create_graph, vectorize=vectorize)