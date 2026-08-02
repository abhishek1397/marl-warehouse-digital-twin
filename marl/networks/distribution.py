"""Reusable probability distribution wrappers for discrete and continuous action spaces."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal


class CategoricalDistribution:
    """Wrapper for categorical probability distributions (Discrete action space)."""

    def __init__(self, logits: torch.Tensor, temperature: float = 1.0) -> None:
        self.logits: torch.Tensor = logits / max(temperature, 1e-6)
        self.dist: Categorical = Categorical(logits=self.logits)

    def sample(self) -> torch.Tensor:
        """Samples an action from the distribution."""
        return self.dist.sample()

    def mode(self) -> torch.Tensor:
        """Returns the greedy deterministic action (argmax of logits)."""
        return torch.argmax(self.logits, dim=-1)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Calculates log probability log P(a) of given action."""
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """Calculates policy entropy H(pi)."""
        return self.dist.entropy()

    @property
    def probs(self) -> torch.Tensor:
        """Returns action probabilities."""
        return self.dist.probs


class NormalDistribution:
    """Wrapper for Gaussian normal probability distributions (Continuous action space)."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        self.loc: torch.Tensor = loc
        self.scale: torch.Tensor = scale
        self.dist: Normal = Normal(loc=self.loc, scale=self.scale)

    def sample(self) -> torch.Tensor:
        """Samples continuous action."""
        return self.dist.sample()

    def mode(self) -> torch.Tensor:
        """Returns distribution mean (deterministic action)."""
        return self.loc

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Calculates log probability density of action."""
        return self.dist.log_prob(action).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Calculates continuous distribution entropy."""
        return self.dist.entropy().sum(dim=-1)


class IndependentDistribution:
    """Wrapper for independent multi-variable probability distributions."""

    def __init__(self, base_dist: Union[Categorical, Normal], reinterpreted_batch_ndims: int = 1) -> None:
        self.dist: Independent = Independent(base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()
