"""CNNCentralizedCritic processing 2D multi-channel spatial grid tensor S for Spatial MAPPO."""

import torch
import torch.nn as nn


class CNNCentralizedCritic(nn.Module):
    """CNN-based Centralized Value Network V(S_spatial) processing multi-channel 2D spatial warehouse grids."""

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.hidden_dim: int = hidden_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.value_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes state value V(S_spatial) for 2D spatial grid input tensor."""
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Shape (1, C, H, W)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.adaptive_pool(out)
        flat = out.view(out.size(0), -1)

        return self.value_head(flat)

    def get_activation_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spatial feature activation maps from third convolution layer."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.conv3(out)
