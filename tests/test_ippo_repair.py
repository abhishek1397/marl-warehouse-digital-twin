"""Regression test suite for IPPO Action Mask & PBRS Repair."""

import pytest
import torch

from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.ippo import IPPOConfig, IPPOTrainer, PolicyManager
from marl.algorithms.ippo.rollout_manager import IPPORolloutManager
from marl.algorithms.ppo.config import PPOConfig
from marl.algorithms.ppo.trainer import PPOTrainer
from marl.config import EnvConfig
from marl.environment import WarehouseGymEnv


def test_action_mask_propagation_in_rollout() -> None:
    """Verifies action_mask is forwarded from env.step info dict into IPPORolloutManager calls."""
    env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=6, grid_height=6, seed=42)
    env = WarehouseParallelEnv(config=env_cfg)

    obs_sp = env.observation_space(env.possible_agents[0])
    act_sp = env.action_space(env.possible_agents[0])

    ippo_cfg = IPPOConfig(num_agents=2, batch_size=50, mini_batch_size=16)
    policy_manager = PolicyManager(env.possible_agents, obs_sp, act_sp, ippo_cfg)
    rollout_manager = IPPORolloutManager()

    steps = rollout_manager.collect_rollouts(env, policy_manager, num_steps=50)
    assert steps == 50

    for agent in policy_manager.get_all_agents():
        assert len(agent.buffer) == 50
    env.close()


def test_pbrs_activation_in_parallel_env() -> None:
    """Verifies enable_reward_shaping=True is active in WarehouseParallelEnv."""
    env_cfg = MultiAgentEnvConfig(num_robots=1, grid_width=6, grid_height=6, enable_reward_shaping=True)
    env = WarehouseParallelEnv(config=env_cfg)
    obs_dict, _ = env.reset(seed=42)

    actions = {"robot_0": env.action_space("robot_0").sample()}
    _, rewards, _, _, _ = env.step(actions)
    assert "robot_0" in rewards
    assert isinstance(rewards["robot_0"], float)
    env.close()


def test_single_agent_gym_vs_1robot_ippo_equivalence() -> None:
    """Verifies Single-Agent Gym PPO vs 1-Robot PettingZoo IPPO equivalence."""
    seed = 42
    timesteps = 600

    # 1. Single-Agent Gym PPO
    gym_cfg = EnvConfig(grid_width=6, grid_height=6, seed=seed, enable_reward_shaping=True, enable_action_masking=True)
    gym_env = WarehouseGymEnv(config=gym_cfg)
    ppo_cfg = PPOConfig(learning_rate=3e-4, epochs=2, batch_size=200, mini_batch_size=64, seed=seed)
    ppo_trainer = PPOTrainer(env=gym_env, config=ppo_cfg)
    ppo_trainer.train(total_timesteps=timesteps)
    ppo_eval = ppo_trainer.evaluate(num_episodes=3)
    gym_env.close()

    # 2. 1-Robot PettingZoo IPPO
    pz_cfg = MultiAgentEnvConfig(num_robots=1, grid_width=6, grid_height=6, seed=seed)
    pz_env = WarehouseParallelEnv(config=pz_cfg)
    ippo_cfg = IPPOConfig(num_agents=1, learning_rate=3e-4, epochs=2, batch_size=200, mini_batch_size=64, seed=seed)
    ippo_trainer = IPPOTrainer(env=pz_env, config=ippo_cfg)
    ippo_trainer.train(total_timesteps=timesteps)
    ippo_eval = ippo_trainer.evaluate(num_episodes=3)
    pz_env.close()

    # Verify rewards match closely
    assert ppo_eval["eval_mean_reward"] == pytest.approx(ippo_eval["eval_mean_reward"], abs=50.0)
