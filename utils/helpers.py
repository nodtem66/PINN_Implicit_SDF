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


Tensor = torch.Tensor
FloatTensor = torch.FloatTensor


def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


def compute_NTK(model, func, x):
    """

    @param model: model with vector output (not scalar output!) the parameters of which we want to compute the Jacobian for
    @param x: input since any gradients requires some input
    @return: either store jac directly in parameters or store them differently

    we'll be working on a copy of the model because we don't want to interfere with the optimizers and other functionality
    """

    jac_model = copy.deepcopy(model)  # because we're messing around with parameters (deleting, reinstating etc)
    all_params, all_names = extract_weights(jac_model)  # "deparameterize weights"
    load_weights(jac_model, all_names, all_params)  # reinstate all weights as plain tensors

    def param_as_input_func(model, x, param):
        load_weights(model, [name], [param])  # name is from the outer scope
        out = func(model, x)
        return out

    sum_jac = torch.zeros((x.shape[0], x.shape[0]), requires_grad=False, device=x.device)
    for i, (name, param) in enumerate(zip(all_names, all_params)):
        jac = torch.autograd.functional.jacobian(
            lambda param: param_as_input_func(jac_model, x, param),
            param,
            strict=True if i == 0 else False,
            vectorize=False if i == 0 else True,
        )
        if len(jac.shape) == 2:
            sum_jac += (jac @ jac.T)
        elif len(jac.shape) == 3:
            sum_jac += torch.einsum('ijk,kjl->il', jac, jac.T)
    
    del jac_model  # cleaning up

    return sum_jac
