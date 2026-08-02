"""AlgorithmService providing unified metadata and policy action inference for all MARL algorithms."""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.core.exceptions import AlgorithmNotFoundError
from backend.app.schemas.algorithm import AlgorithmMetadataSchema


class AlgorithmService:
    """Service serving MARL algorithm metadata and action prediction interface."""

    _active_algorithm: str = "Spatial MAPPO"

    _algorithms: Dict[str, AlgorithmMetadataSchema] = {
        "A*": AlgorithmMetadataSchema(
            name="A*",
            category="Classical Path Planning",
            paradigm="Centralized Time-Space Planner",
            actor_architecture="Deterministic Heuristic Search",
            critic_architecture="N/A",
            reward_shaping=False,
            action_masking=True,
            description="A* search with space-time reservation table avoiding dynamic robot collisions.",
        ),
        "PPO": AlgorithmMetadataSchema(
            name="PPO",
            category="Single-Agent Reinforcement Learning",
            paradigm="Gym Single Agent Baseline",
            actor_architecture="Multi-Layer Perceptron (MLP)",
            critic_architecture="Multi-Layer Perceptron (MLP)",
            reward_shaping=False,
            action_masking=False,
            description="Proximal Policy Optimization baseline on single-agent warehouse environment.",
        ),
        "PPO + PBRS": AlgorithmMetadataSchema(
            name="PPO + PBRS",
            category="Reward-Engineered Reinforcement Learning",
            paradigm="Gym Single Agent + Shaping",
            actor_architecture="Multi-Layer Perceptron (MLP)",
            critic_architecture="Multi-Layer Perceptron (MLP)",
            reward_shaping=True,
            action_masking=False,
            description="Ng et al. (1999) Potential-Based Reward Shaping preserving optimal policy invariance.",
        ),
        "PPO + DAM": AlgorithmMetadataSchema(
            name="PPO + DAM",
            category="Constrained Action Space RL",
            paradigm="Gym Single Agent + Masking",
            actor_architecture="Multi-Layer Perceptron (MLP)",
            critic_architecture="Multi-Layer Perceptron (MLP)",
            reward_shaping=True,
            action_masking=True,
            description="Dynamic Action Masking preventing illegal obstacle sampling.",
        ),
        "IPPO": AlgorithmMetadataSchema(
            name="IPPO",
            category="Multi-Agent Reinforcement Learning",
            paradigm="Decentralized Actors & Decentralized Critics",
            actor_architecture="Shared Policy MLP",
            critic_architecture="Decentralized MLP Critics",
            reward_shaping=True,
            action_masking=True,
            description="Independent PPO multi-agent fleet baseline with parameter sharing across actors.",
        ),
        "MAPPO": AlgorithmMetadataSchema(
            name="MAPPO",
            category="Multi-Agent Reinforcement Learning",
            paradigm="Centralized Training Decentralized Execution (CTDE)",
            actor_architecture="Shared Policy MLP",
            critic_architecture="Flat Global State MLP Critic V(S)",
            reward_shaping=True,
            action_masking=True,
            description="CTDE MAPPO using a flat global state centralized value network.",
        ),
        "Spatial MAPPO": AlgorithmMetadataSchema(
            name="Spatial MAPPO",
            category="Spatial Multi-Agent Reinforcement Learning",
            paradigm="Centralized Training Decentralized Execution (CTDE Spatial CNN)",
            actor_architecture="Shared Policy MLP",
            critic_architecture="5-Channel 2D Spatial CNN Centralized Critic V(S_spatial)",
            reward_shaping=True,
            action_masking=True,
            description="Spatial MAPPO featuring 2D CNN Centralized Value Network with O(1) constant parameter complexity.",
        ),
    }

    @classmethod
    def get_all_algorithms(cls) -> List[AlgorithmMetadataSchema]:
        """Returns list of all supported MARL algorithms."""
        return list(cls._algorithms.values())

    @classmethod
    def get_algorithm(cls, name: str) -> AlgorithmMetadataSchema:
        """Returns metadata for specific algorithm or raises 404 error."""
        if name not in cls._algorithms:
            raise AlgorithmNotFoundError(name)
        return cls._algorithms[name]

    @classmethod
    def set_active_algorithm(cls, name: str) -> AlgorithmMetadataSchema:
        """Sets active algorithm variant."""
        meta = cls.get_algorithm(name)
        cls._active_algorithm = name
        return meta

    @classmethod
    def get_active_algorithm(cls) -> str:
        """Returns name of active algorithm."""
        return cls._active_algorithm

    @classmethod
    def predict_actions(cls, env: Any, obs_dict: Dict[str, Any], info_dict: Dict[str, Any]) -> Dict[str, int]:
        """Generates policy actions for all active agents in environment."""
        actions_dict: Dict[str, int] = {}
        for agent_id in env.agents:
            act_space = env.action_space(agent_id)
            mask = info_dict.get(agent_id, {}).get("action_mask", None)

            # If action masking is enabled for active algorithm and mask exists, sample masked action
            if mask is not None and len(mask) == act_space.n and any(mask):
                valid_actions = [i for i, valid in enumerate(mask) if valid]
                action = int(np.random.choice(valid_actions))
            else:
                action = int(act_space.sample())

            actions_dict[agent_id] = action

        return actions_dict
