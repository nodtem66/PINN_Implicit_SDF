# -*- coding: utf-8 -*-
import torch

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