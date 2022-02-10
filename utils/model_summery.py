from __future__ import annotations

from typing import (Any,)

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
    Summarize the given PyTorch model. Summarized information includes:
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