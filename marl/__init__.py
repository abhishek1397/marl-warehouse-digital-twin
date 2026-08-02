"""Multi-Agent Reinforcement Learning Package, Gym Environment, and PettingZoo ParallelEnv Registration."""

import gymnasium as gym

from marl.action import ActionMapper, ActionResult
from marl.agent_action import AgentActionMapper
from marl.agent_manager import AgentManager
from marl.agent_observation import AgentObservationEncoder
from marl.agent_reward import AgentRewardEngine
from marl.communication import CommunicationManager
from marl.config import EnvConfig
from marl.environment import WarehouseGymEnv
from marl.episode import EpisodeManager
from marl.multi_agent_config import MultiAgentEnvConfig
from marl.observation import ObservationEncoder
from marl.parallel_env import WarehouseParallelEnv
from marl.rendering import EnvironmentRenderer
from marl.reward import RewardEngine
from marl.spaces import get_action_space, get_observation_space
from marl.utils import set_seed

# Register WarehouseGymEnv into Gymnasium registry
gym.register(
    id="Warehouse-v0",
    entry_point="marl.environment:WarehouseGymEnv",
)

__all__ = [
    "WarehouseGymEnv",
    "WarehouseParallelEnv",
    "EnvConfig",
    "MultiAgentEnvConfig",
    "AgentManager",
    "AgentObservationEncoder",
    "AgentActionMapper",
    "AgentRewardEngine",
    "CommunicationManager",
    "ObservationEncoder",
    "ActionMapper",
    "ActionResult",
    "RewardEngine",
    "EpisodeManager",
    "EnvironmentRenderer",
    "get_action_space",
    "get_observation_space",
    "set_seed",
]
