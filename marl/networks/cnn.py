"""CNN feature extractor module for 2D spatial grid/image observations."""

from typing import List, Optional

import torch
import torch.nn as nn

from marl.networks.base import BaseNetwork
from marl.networks.initialization import init_weights
from marl.networks.mlp import get_activation_fn


class CNNFeatureExtractor(BaseNetwork):
    """Convolutional Neural Network (CNN) feature extractor for 2D spatial inputs."""

    def __init__(
        self,
        input_channels: int = 1,
        output_dim: int = 128,
        channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
        activation: str = "relu",
        init_type: str = "orthogonal",
    ) -> None:
        super().__init__()
        self.input_channels: int = input_channels
        self.output_dim: int = output_dim

        channels_list = channels if channels is not None else [16, 32, 64]
        kernel_list = kernel_sizes if kernel_sizes is not None else [3, 3, 3]
        stride_list = strides if strides is not None else [1, 1, 1]
        pad_list = padding if padding is not None else [1, 1, 1]

        conv_layers: List[nn.Module] = []
        prev_ch = input_channels

        for ch, k, s, p in zip(channels_list, kernel_list, stride_list, pad_list):
            conv_layers.append(nn.Conv2d(prev_ch, ch, kernel_size=k, stride=s, padding=p))
            conv_layers.append(get_activation_fn(activation))
            prev_ch = ch

        conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        conv_layers.append(nn.Flatten())
        conv_layers.append(nn.Linear(prev_ch, output_dim))
        conv_layers.append(get_activation_fn(activation))

        self.conv_network = nn.Sequential(*conv_layers)
        init_weights(self.conv_network, init_type=init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 2D spatial image tensor (B, C, H, W) or (B, H, W)."""
        x = x.to(self.device)
        if x.dim() == 3:  # (B, H, W) -> add channel dim
            x = x.unsqueeze(1)

        x = x.to(torch.float32)
        return self.conv_network(x)
