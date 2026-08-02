"""NetworkFactory for instantiating neural network modules from configuration or string names."""

from typing import Any, Dict, Type

from marl.networks.actor import ActorNetwork
from marl.networks.base import BaseNetwork
from marl.networks.cnn import CNNFeatureExtractor
from marl.networks.critic import CriticNetwork
from marl.networks.feature_extractor import FeatureExtractor
from marl.networks.mlp import MLP
from marl.networks.policy_network import PolicyNetwork
from marl.networks.shared_actor_critic import SharedActorCritic
from marl.networks.value_network import ValueNetwork


class NetworkFactory:
    """Factory pattern providing creation methods for PyTorch neural network modules."""

    _REGISTRY: Dict[str, Type[BaseNetwork]] = {
        "mlp": MLP,
        "cnn": CNNFeatureExtractor,
        "feature_extractor": FeatureExtractor,
        "actor": ActorNetwork,
        "critic": CriticNetwork,
        "shared_actor_critic": SharedActorCritic,
        "value": ValueNetwork,
        "policy": PolicyNetwork,
    }

    @classmethod
    def create(cls, network_type: str, **kwargs: Any) -> BaseNetwork:
        """Instantiates a network module by type name string.

        Args:
            network_type: Name string ('mlp', 'actor', 'critic', 'shared_actor_critic', etc.).
            **kwargs: Arguments passed to module constructor.

        Returns:
            Instantiated BaseNetwork sub-class.
        """
        key = network_type.lower()
        if key not in cls._REGISTRY:
            raise KeyError(
                f"Unknown network type '{network_type}'. Supported types: {list(cls._REGISTRY.keys())}"
            )

        network_cls = cls._REGISTRY[key]
        return network_cls(**kwargs)

    @classmethod
    def register(cls, name: str, network_cls: Type[BaseNetwork]) -> None:
        """Registers a custom network class with the factory."""
        cls._REGISTRY[name.lower()] = network_cls
