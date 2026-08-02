"""MiniBatchSampler generating mini-batch indices for PPO/MAPPO optimization epochs."""

from typing import Iterator, List

import torch


class MiniBatchSampler:
    """Generates mini-batch index slices over multiple training epochs."""

    def __init__(
        self,
        dataset_size: int,
        mini_batch_size: int,
        num_epochs: int = 4,
        shuffle: bool = True,
    ) -> None:
        self.dataset_size: int = dataset_size
        self.mini_batch_size: int = min(mini_batch_size, dataset_size)
        self.num_epochs: int = num_epochs
        self.shuffle: bool = shuffle

    def sample_indices(self) -> Iterator[torch.Tensor]:
        """Yields mini-batch index tensors across num_epochs."""
        if self.dataset_size == 0:
            return

        for _ in range(self.num_epochs):
            if self.shuffle:
                permutation = torch.randperm(self.dataset_size)
            else:
                permutation = torch.arange(self.dataset_size)

            for start_idx in range(0, self.dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, self.dataset_size)
                yield permutation[start_idx:end_idx]
