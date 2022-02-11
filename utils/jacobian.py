# from: https://github.com/ChenAo-Phys/pytorch-Jacobian
from __future__ import annotations

import types
from functools import partial
from typing import Any

import torch
from torch import nn
from torchinfo.layer_info import LayerInfo
from torchinfo.torchinfo import *

def get_layer_info(
    model: nn.Module,
    input_size: INPUT_SIZE_TYPE | None = None,
    input_data: INPUT_DATA_TYPE | None = None,
    batch_dim: int | None = None,
    cache_forward_pass: bool | None = None,
    device: torch.device | str | None = None,
    dtypes: list[torch.dtype] | None = None,
    **kwargs: Any,
) -> list[LayerInfo]:
    """
    Extend torchinfo to get the LayerInfo from pytorch module.
    This function returns the summary of the given PyTorch model:
        1) Layer names,
        2) input/output shapes,
        3) kernel shape,
        4) # of parameters,
        5) # of operations (Mult-Adds)
    NOTE: If neither input_data or input_size are provided, no forward pass through the
    network is performed, and the provided model information is limited to layer names.
    Args:
        model (nn.Module):
                PyTorch model to summarize. The model should be fully in either train()
                or eval() mode. If layers are not all in the same mode, running summary
                may have side effects on batchnorm or dropout statistics. If you
                encounter an issue with this, please open a GitHub issue.
        input_size (Sequence of Sizes):
                Shape of input data as a List/Tuple/torch.Size
                (dtypes must match model input, default is FloatTensors).
                You should include batch size in the tuple.
                Default: None
        input_data (Sequence of Tensors):
                Arguments for the model's forward pass (dtypes inferred).
                If the forward() function takes several parameters, pass in a list of
                args or a dict of kwargs (if your forward() function takes in a dict
                as its only argument, wrap it in a list).
                Default: None
        batch_dim (int):
                Batch_dimension of input data. If batch_dim is None, assume
                input_data / input_size contains the batch dimension, which is used
                in all calculations. Else, expand all tensors to contain the batch_dim.
                Specifying batch_dim can be an runtime optimization, since if batch_dim
                is specified, torchinfo uses a batch size of 1 for the forward pass.
                Default: None
        cache_forward_pass (bool):
                If True, cache the run of the forward() function using the model
                class name as the key. If the forward pass is an expensive operation,
                this can make it easier to modify the formatting of your model
                summary, e.g. changing the depth or enabled column types, especially
                in Jupyter Notebooks.
                WARNING: Modifying the model architecture or input data/input size when
                this feature is enabled does not invalidate the cache or re-run the
                forward pass, and can cause incorrect summaries as a result.
                Default: False
        depth (int):
                Depth of nested layers to display (e.g. Sequentials).
                Nested layers below this depth will not be displayed in the summary.
                Default: 3
        device (torch.Device):
                Uses this torch device for model and input_data.
                If not specified, uses result of torch.cuda.is_available().
                Default: None
        dtypes (List[torch.dtype]):
                If you use input_size, torchinfo assumes your input uses FloatTensors.
                If your model use a different data type, specify that dtype.
                For multiple inputs, specify the size of both inputs, and
                also specify the types of each parameter here.
                Default: None
        **kwargs:
                Other arguments used in `model.forward` function. Passing *args is no
                longer supported.
    Return:
        LayerInfo list
                See torchinfo/model_statistics.py for more information.
    """
    if cache_forward_pass is None:
        # In the future, this may be enabled by default in Jupyter Notebooks
        cache_forward_pass = False

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validate_user_params(input_data, input_size, [], 25, 0)

    x, correct_input_size = process_input(
        input_data, input_size, batch_dim, device, dtypes
    )
    return forward_pass(
        model, x, batch_dim, cache_forward_pass, device, **kwargs
    )


def layers_for_jacobian(layer_info: list[LayerInfo]):
    for layer in layer_info:
        if layer.num_params and layer.is_leaf_layer:
            yield layer

def extend(model, input_shape, **kwargs):
    if not isinstance(model, nn.Module):
        raise TypeError("model should be a nn.Module")
    if not isinstance(input_shape, tuple):
        raise TypeError("input_shape should be a tuple")

    device = next(model.parameters()).device

    weight_input_list = []
    weight_output_list = []
    weight_repeat_list = []
    bias_output_list = []
    bias_repeat_list = []
    model._layer_info_list = get_layer_info(model, (1,) + input_shape, **kwargs)

    with torch.no_grad():
        for layerinfo in layers_for_jacobian(model._layer_info_list):
            # for all layers with parameters
            # store parameters and clear bias for future calculation
            module = layerinfo.module
            x = torch.zeros(layerinfo.input_size, device=device)
            y = torch.zeros(layerinfo.output_size, device=device)

            if module.weight is not None:
                initial_weight = module.weight.data.clone()
            if module.bias is not None:
                initial_bias = module.bias.data.clone()
                module.bias.data = torch.zeros_like(module.bias)

            if module.weight is not None:
                Nweight = module.weight.numel()
                weight_input = []
                weight_output = []
                weight_repeat = torch.zeros(
                    Nweight, dtype=torch.long, device=device
                )
                Xeye = torch.eye(x.numel(), device=device).reshape(
                    (-1,) + x.shape[1:]
                )
                for i in range(Nweight):
                    weight = torch.zeros(Nweight, device=device)
                    weight[i] = 1.0
                    module.weight.data = weight.reshape(module.weight.shape)
                    # output of module is of dimension (j,k)
                    out = module(Xeye).reshape(x.numel(), y.numel())
                    if (out[out.abs() > 1e-5] - 1.0).abs().max() > 1e-5:
                        raise RuntimeError(
                            "the network is not written in the standard form, see https://github.com/ChenAo-Phys/pytorch-Jacobian"
                        )
                    nonzero = torch.nonzero(out > 0.5, as_tuple=False)
                    weight_input.append(nonzero[:, 0])
                    weight_output.append(nonzero[:, 1])
                    weight_repeat[i] = nonzero.shape[0]
                weight_input_list.append(torch.cat(weight_input, dim=0))
                weight_output_list.append(torch.cat(weight_output, dim=0))
                weight_repeat_list.append(weight_repeat)
                module.weight.data = initial_weight
            else:
                weight_input_list.append(None)
                weight_output_list.append(None)
                weight_repeat_list.append(None)

            if module.bias is not None:
                Nbias = module.bias.numel()
                bias_output = []
                bias_repeat = torch.zeros(Nbias, dtype=torch.long, device=device)
                for i in range(Nbias):
                    bias = torch.zeros(Nbias, device=device)
                    bias[i] = 1.0
                    module.bias.data = bias.reshape(module.bias.shape)
                    out = module(x).reshape(-1)
                    if (out[out.abs() > 1e-5] - 1.0).abs().max() > 1e-5:
                        raise RuntimeError(
                            "the network is not written in the standard form, see https://github.com/ChenAo-Phys/pytorch-Jacobian"
                        )
                    nonzero = torch.nonzero(out > 0.5, as_tuple=False)
                    bias_output.append(nonzero[:, 0])
                    bias_repeat[i] = nonzero.shape[0]
                bias_output_list.append(torch.cat(bias_output, dim=0))
                bias_repeat_list.append(bias_repeat)
                module.bias.data = initial_bias
            else:
                bias_output_list.append(None)
                bias_repeat_list.append(None)

    if not hasattr(model, "_Jacobian_shape_dict"):
        model._Jacobian_shape_dict = {}
    model._Jacobian_shape_dict[input_shape] = (
        weight_input_list,
        weight_output_list,
        weight_repeat_list,
        bias_output_list,
        bias_repeat_list,
    )

    # assign jacobian method to model
    def jacobian(self, as_tuple=False):
        shape = self.input_shape
        is_extended_model = hasattr(self, "_layer_info_list") and hasattr(self, "_Jacobian_shape_dict")
        if is_extended_model and shape in self._Jacobian_shape_dict:
            (
                weight_input_list,
                weight_output_list,
                weight_repeat_list,
                bias_output_list,
                bias_repeat_list,
            ) = self._Jacobian_shape_dict[shape]
        else:
            raise RuntimeError(
                "model or specific input shape is not extended for jacobian calculation"
            )

        device = next(model.parameters()).device
        jac = []
        layer = 0
        for layerinfo in layers_for_jacobian(self._layer_info_list):
            # for all layers with parameters
            # store parameters and clear bias for future calculation
            module = layerinfo.module

            weight_input = weight_input_list[layer]
            weight_output = weight_output_list[layer]
            weight_repeat = weight_repeat_list[layer]
            bias_output = bias_output_list[layer]
            bias_repeat = bias_repeat_list[layer]
            x = self.x_in[layer]
            N = x.shape[0]
            dz_dy = self.gradient[layer].reshape(N, -1)

            if weight_repeat is not None:
                Nweight = weight_repeat.shape[0]
                dz_dy_select = dz_dy[:, weight_output]
                x_select = x.reshape(N, -1)[:, weight_input]
                repeat = torch.repeat_interleave(weight_repeat)
                dz_dW = torch.zeros(N, Nweight, device=device).index_add_(
                    1, repeat, dz_dy_select * x_select
                )
                if as_tuple:
                    dz_dW = dz_dW.reshape((N,) + module.weight.shape)
                jac.append(dz_dW)
            if bias_repeat is not None:
                Nbias = bias_repeat.shape[0]
                dz_dy_select = dz_dy[:, bias_output]
                repeat = torch.repeat_interleave(bias_repeat)
                dz_db = torch.zeros(N, Nbias, device=device).index_add_(
                    1, repeat, dz_dy_select
                )
                if as_tuple:
                    dz_db = dz_db.reshape((N,) + module.bias.shape)
                jac.append(dz_db)
            layer += 1

        if as_tuple:
            return tuple(jac)
        else:
            return torch.cat(jac, dim=1)

    if not hasattr(model, "jacobian"):
        model.jacobian = types.MethodType(jacobian, model)


class JacobianMode:
    def __init__(self, model):
        self.model = model
        if not isinstance(model, nn.Module):
            raise TypeError("model should be a nn.Module")

    def __enter__(self):
        model = self.model
        model.x_in = []
        model.gradient = []
        self.forward_pre_hook = []
        self.backward_hook = []

        def record_input_shape(self, input):
            model.input_shape = input[0].shape[1:]

        def record_forward(self, input, layer):
            model.x_in[layer] = input[0].detach()

        def record_backward(self, grad_input, grad_output, layer):
            model.gradient[layer] = grad_output[0]

        module0 = next(layers_for_jacobian(model._layer_info_list)).module
        self.first_forward_hook = module0.register_forward_pre_hook(record_input_shape)

        layer = 0
        for layerinfo in layers_for_jacobian(model._layer_info_list):
            module = layerinfo.module
            model.x_in.append(None)
            model.gradient.append(None)
            self.forward_pre_hook.append(
                module.register_forward_pre_hook(
                    partial(record_forward, layer=layer)
                )
            )
            self.backward_hook.append(
                module.register_full_backward_hook(partial(record_backward, layer=layer))
            )
            layer += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.first_forward_hook.remove()
        for hook in self.forward_pre_hook:
            hook.remove()
        for hook in self.backward_hook:
            hook.remove()

        del self.model.input_shape
        del self.model.x_in
        del self.model.gradient
