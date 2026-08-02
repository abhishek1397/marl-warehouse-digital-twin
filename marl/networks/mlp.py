"""Configurable Multi-Layer Perceptron (MLP) module."""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from marl.networks.base import BaseNetwork
from marl.networks.initialization import init_weights


def get_activation_fn(activation_name: str) -> nn.Module:
    """Returns PyTorch activation module for given string name."""
    name = activation_name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(0.2)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity" or name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


class MLP(BaseNetwork):
    """Multi-Layer Perceptron (MLP) with configurable layers, norm, dropout, and initialization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_batch_norm: bool = False,
        use_residual: bool = False,
        init_type: str = "orthogonal",
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.hidden_dims: List[int] = hidden_dims if hidden_dims is not None else [128, 128]
        self.use_residual: bool = use_residual

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(get_activation_fn(activation))

            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(get_activation_fn(output_activation))

        self.network: nn.Sequential = nn.Sequential(*layers)

        # Apply weight initialization
        init_weights(self.network, init_type=init_type, gain=gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        x = x.to(self.device)
        out = self.network(x)
        if self.use_residual and x.shape == out.shape:
            return x + out
        return out
