"""FeatureExtractor module for processing vector, spatial, and Gymnasium Dict observations."""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict as GymDict, Space

from marl.networks.base import BaseNetwork
from marl.networks.cnn import CNNFeatureExtractor
from marl.networks.mlp import MLP


class FeatureExtractor(BaseNetwork):
    """Unified Feature Extractor handling vector states, spatial grids, and Gymnasium Dict observations."""

    def __init__(
        self,
        observation_space: Union[Space, int, GymDict],
        output_dim: int = 128,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        init_type: str = "orthogonal",
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self._output_dim: int = output_dim
        self.is_dict_space: bool = False
        self.is_spatial: bool = False

        if isinstance(observation_space, int):
            self.input_dim = observation_space
            self.extractor = MLP(
                input_dim=self.input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims or [128, 128],
                activation=activation,
                init_type=init_type,
            )
        elif hasattr(observation_space, "spaces") or isinstance(observation_space, (dict, GymDict)) or hasattr(observation_space, "items"):
            self.is_dict_space = True
            if hasattr(observation_space, "spaces"):
                self.spaces_dict = observation_space.spaces
            elif hasattr(observation_space, "items"):
                self.spaces_dict = dict(observation_space.items())
            else:
                self.spaces_dict = {}

            total_flat_dim = 0
            for key, space in self.spaces_dict.items():
                if hasattr(space, "shape"):
                    total_flat_dim += int(np.prod(space.shape))
                elif isinstance(space, torch.Tensor):
                    total_flat_dim += space.numel()

            self.extractor = MLP(
                input_dim=total_flat_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims or [128, 128],
                activation=activation,
                init_type=init_type,
            )
        elif hasattr(observation_space, "shape"):
            shape = observation_space.shape
            if len(shape) >= 2:  # Spatial image/grid (C, H, W) or (H, W)
                self.is_spatial = True
                ch = shape[0] if len(shape) == 3 else 1
                self.extractor = CNNFeatureExtractor(
                    input_channels=ch,
                    output_dim=output_dim,
                    activation=activation,
                    init_type=init_type,
                )
            else:
                self.input_dim = shape[0]
                self.extractor = MLP(
                    input_dim=self.input_dim,
                    output_dim=output_dim,
                    hidden_dims=hidden_dims or [128, 128],
                    activation=activation,
                    init_type=init_type,
                )
        else:
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")

    @property
    def output_dim(self) -> int:
        """Returns the output feature dimension."""
        return self._output_dim

    def forward(
        self, obs: Union[torch.Tensor, Dict[str, Union[torch.Tensor, Any]]]
    ) -> torch.Tensor:
        """Extracts features from vector, spatial, or dict observation inputs."""
        if self.is_dict_space and isinstance(obs, dict):
            is_batched = False
            batch_size = 1

            for k, v in obs.items():
                v_arr = v if isinstance(v, (torch.Tensor, np.ndarray)) else np.array(v)
                if hasattr(self, "spaces_dict") and k in self.spaces_dict:
                    space_shape = self.spaces_dict[k].shape
                    if v_arr.ndim > len(space_shape):
                        is_batched = True
                        batch_size = v_arr.shape[0]
                        break

            flattened_parts = []
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    v_tensor = v.to(self.device).to(torch.float32)
                else:
                    v_tensor = torch.tensor(v, dtype=torch.float32, device=self.device)

                if is_batched:
                    v_flat = v_tensor.reshape(batch_size, -1)
                else:
                    v_flat = v_tensor.reshape(1, -1)

                flattened_parts.append(v_flat)

            if flattened_parts:
                concat_obs = torch.cat(flattened_parts, dim=1)
                return self.extractor(concat_obs)

        if isinstance(obs, torch.Tensor):
            obs = obs.to(self.device).to(torch.float32)
            if self.is_spatial:
                space_shape = self.observation_space.shape
                if len(space_shape) == 2:
                    if obs.dim() == 2:
                        obs = obs.unsqueeze(0).unsqueeze(1)
                    elif obs.dim() == 3:
                        obs = obs.unsqueeze(1)
                elif len(space_shape) == 3:
                    if obs.dim() == 3:
                        obs = obs.unsqueeze(0)
            elif obs.dim() == 1:
                obs = obs.unsqueeze(0)

            return self.extractor(obs)

        raise TypeError(f"Unexpected observation input type: {type(obs)}")
