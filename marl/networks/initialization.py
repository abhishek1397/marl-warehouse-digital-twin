"""Weight initialization strategies for PyTorch neural network modules."""

import math
from typing import Optional

torch_nn = None
import torch
import torch.nn as nn


def init_orthogonal(module: nn.Module, gain: float = 1.0, bias_const: float = 0.0) -> None:
    """Initializes Linear or Conv layers with orthogonal matrix weights."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias_const)


def init_xavier(module: nn.Module, gain: float = 1.0, bias_const: float = 0.0) -> None:
    """Initializes layers with Xavier uniform weights."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias_const)


def init_kaiming(
    module: nn.Module, nonlinearity: str = "relu", bias_const: float = 0.0
) -> None:
    """Initializes layers with Kaiming uniform weights."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias_const)


def init_weights(
    module: nn.Module,
    init_type: str = "orthogonal",
    gain: float = 1.0,
    bias_const: float = 0.0,
) -> None:
    """Applies named weight initialization strategy across sub-modules.

    Args:
        module: PyTorch module.
        init_type: Strategy ('orthogonal', 'xavier_uniform', 'kaiming_uniform', 'normal', 'uniform').
        gain: Scaling gain multiplier.
        bias_const: Constant value for bias vector.
    """
    init_key = init_type.lower()

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if init_key == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_key in {"xavier", "xavier_uniform"}:
                nn.init.xavier_uniform_(m.weight, gain=gain)
            elif init_key in {"kaiming", "kaiming_uniform"}:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif init_key == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=gain)
            elif init_key == "uniform":
                nn.init.uniform_(m.weight, a=-gain, b=gain)

            if m.bias is not None:
                nn.init.constant_(m.bias, bias_const)
