"""Comprehensive test suite for marl/algorithms/mappo package."""

import os
import pytest
import torch

from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.mappo import (
    CentralizedValueNetwork,
    MAPPOAgent,
    MAPPOBatchBuilder,
    MAPPOCheckpointHandler,
    MAPPOConfig,
    MAPPOEvaluator,
    MAPPOMetricsTracker,
    MAPPOTrainer,
    SharedPolicyManager,
)
from marl.trainer.checkpoint_manager import CheckpointManager
from marl.trainer.config import CheckpointSubConfig


def test_mappo_config() -> None:
    cfg = MAPPOConfig(num_agents=4, shared_policy=True, centralized_critic=True, actor_lr=1e-3)
    assert cfg.num_agents == 4
    assert cfg.shared_policy is True
    assert cfg.centralized_critic is True
    assert cfg.actor_lr == 1e-3


def test_centralized_value_network() -> None:
    critic = CentralizedValueNetwork(state_dim=64, hidden_dim=32)
    dummy_state = torch.randn(8, 64)
    val_out = critic(dummy_state)
    assert val_out.shape == (8, 1)


def test_shared_policy_manager() -> None:
    env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=6, grid_height=6)
    env = WarehouseParallelEnv(config=env_cfg)
    obs_sp = env.observation_space(env.possible_agents[0])
    act_sp = env.action_space(env.possible_agents[0])

    spm = SharedPolicyManager(env.possible_agents, obs_sp, act_sp, MAPPOConfig(shared_policy=True))
    assert len(spm.get_all_agents()) == 1
    assert spm.get_agent("robot_0") is spm.get_agent("robot_1")
    env.close()


def test_mappo_metrics_tracker() -> None:
    fair = MAPPOMetricsTracker.compute_jains_fairness([10.0, 10.0])
    assert pytest.approx(fair, 1e-3) == 1.0

    agg = MAPPOMetricsTracker.aggregate_multi_agent_metrics({"r0": 5.0, "r1": 15.0}, {"r0": 0.2}, critic_loss=0.1)
    assert agg["mean_reward"] == 10.0
    assert agg["critic_loss"] == 0.1


def test_mappo_trainer_and_evaluator_cycle(tmp_path) -> None:
    env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=6, grid_height=6)
    env = WarehouseParallelEnv(config=env_cfg)
    mappo_cfg = MAPPOConfig(num_agents=2, batch_size=100, mini_batch_size=32, eval_interval=100, eval_episodes=2)

    trainer = MAPPOTrainer(env=env, config=mappo_cfg)
    trainer.train(total_timesteps=100)

    eval_metrics = trainer.evaluate(num_episodes=2)
    assert "eval_mean_reward" in eval_metrics
    assert "eval_jains_fairness" in eval_metrics

    # Test checkpoint handler
    ckpt_mgr = CheckpointManager(checkpoint_dir=str(tmp_path), config=CheckpointSubConfig())
    handler = MAPPOCheckpointHandler(ckpt_mgr)
    path = handler.save_checkpoint(trainer.policy_manager, trainer.centralized_critic, step=100)
    assert os.path.exists(path)

    payload = handler.load_checkpoint(trainer.policy_manager, trainer.centralized_critic, path)
    assert payload["step"] == 100

    env.close()
