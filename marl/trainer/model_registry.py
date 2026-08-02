"""ModelRegistry providing decorator-based registration for future MARL algorithms."""

from typing import Any, Callable, Dict, List, Type


class ModelRegistry:
    """Registry pattern allowing future RL algorithms, networks, and optimizers to register dynamically."""

    _policies: Dict[str, Type[Any]] = {}
    _networks: Dict[str, Type[Any]] = {}
    _optimizers: Dict[str, Type[Any]] = {}
    _evaluators: Dict[str, Type[Any]] = {}

    @classmethod
    def register_policy(cls, name: str) -> Callable[[Type[Any]], Type[Any]]:
        """Decorator to register a MARL policy class (e.g., 'mappo', 'qmix', 'ippo')."""
        def decorator(policy_cls: Type[Any]) -> Type[Any]:
            cls._policies[name.lower()] = policy_cls
            return policy_cls
        return decorator

    @classmethod
    def get_policy(cls, name: str) -> Type[Any]:
        """Retrieves a registered policy class by name."""
        key = name.lower()
        if key not in cls._policies:
            raise KeyError(f"Policy '{name}' is not registered. Available: {list(cls._policies.keys())}")
        return cls._policies[key]

    @classmethod
    def register_network(cls, name: str) -> Callable[[Type[Any]], Type[Any]]:
        """Decorator to register a neural network architecture class."""
        def decorator(net_cls: Type[Any]) -> Type[Any]:
            cls._networks[name.lower()] = net_cls
            return net_cls
        return decorator

    @classmethod
    def get_network(cls, name: str) -> Type[Any]:
        """Retrieves a registered network class by name."""
        key = name.lower()
        if key not in cls._networks:
            raise KeyError(f"Network '{name}' is not registered. Available: {list(cls._networks.keys())}")
        return cls._networks[key]

    @classmethod
    def register_optimizer(cls, name: str) -> Callable[[Type[Any]], Type[Any]]:
        """Decorator to register an optimizer class."""
        def decorator(opt_cls: Type[Any]) -> Type[Any]:
            cls._optimizers[name.lower()] = opt_cls
            return opt_cls
        return decorator

    @classmethod
    def get_optimizer(cls, name: str) -> Type[Any]:
        """Retrieves a registered optimizer class by name."""
        key = name.lower()
        if key not in cls._optimizers:
            raise KeyError(f"Optimizer '{name}' is not registered. Available: {list(cls._optimizers.keys())}")
        return cls._optimizers[key]

    @classmethod
    def register_evaluator(cls, name: str) -> Callable[[Type[Any]], Type[Any]]:
        """Decorator to register an evaluation harness class."""
        def decorator(eval_cls: Type[Any]) -> Type[Any]:
            cls._evaluators[name.lower()] = eval_cls
            return eval_cls
        return decorator

    @classmethod
    def get_evaluator(cls, name: str) -> Type[Any]:
        """Retrieves a registered evaluator class by name."""
        key = name.lower()
        if key not in cls._evaluators:
            raise KeyError(f"Evaluator '{name}' is not registered. Available: {list(cls._evaluators.keys())}")
        return cls._evaluators[key]

    @classmethod
    def list_registered(cls) -> Dict[str, List[str]]:
        """Lists all currently registered components."""
        return {
            "policies": list(cls._policies.keys()),
            "networks": list(cls._networks.keys()),
            "optimizers": list(cls._optimizers.keys()),
            "evaluators": list(cls._evaluators.keys()),
        }
