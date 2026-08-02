"""Comprehensive test suite for marl/algorithms/ippo package."""

import os
import pytest
import torch

from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.ippo import (
    IPPOAgent,
    IPPOCheckpointHandler,
    IPPOConfig,
    IPPOEvaluator,
    IPPOMetricsTracker,
    IPPOTrainer,
    PolicyManager,
)
from marl.networks.policy_network import PolicyNetwork
from marl.trainer.checkpoint_manager import CheckpointManager
from marl.trainer.config import CheckpointSubConfig


def test_ippo_config() -> None:
    cfg = IPPOConfig(num_agents=4, shared_policy=True, learning_rate=1e-3)
    assert cfg.num_agents == 4
    assert cfg.shared_policy is True
    assert cfg.learning_rate == 1e-3


def test_policy_manager_independent_and_shared() -> None:
    env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=6, grid_height=6)
    env = WarehouseParallelEnv(config=env_cfg)
    obs_sp = env.observation_space(env.possible_agents[0])
    act_sp = env.action_space(env.possible_agents[0])

    # Mode 1: Independent parameters
    pm_indep = PolicyManager(env.possible_agents, obs_sp, act_sp, IPPOConfig(shared_policy=False))
    assert len(pm_indep.get_all_agents()) == 2
    assert pm_indep.get_agent("robot_0") is not pm_indep.get_agent("robot_1")

    # Mode 2: Shared parameters
    pm_shared = PolicyManager(env.possible_agents, obs_sp, act_sp, IPPOConfig(shared_policy=True))
    assert len(pm_shared.get_all_agents()) == 1
    assert pm_shared.get_agent("robot_0") is pm_shared.get_agent("robot_1")
    env.close()


def test_ippo_metrics_tracker() -> None:
    # 1. Jain's fairness index
    fair_equal = IPPOMetricsTracker.compute_jains_fairness([10.0, 10.0, 10.0])
    assert pytest.approx(fair_equal, 1e-3) == 1.0

    fair_unequal = IPPOMetricsTracker.compute_jains_fairness([10.0, 0.0, 0.0])
    assert fair_unequal < 1.0

    # 2. Metrics aggregation
    agg = IPPOMetricsTracker.aggregate_multi_agent_metrics({"a0": 10.0, "a1": 20.0}, {"a0": 0.5})
    assert agg["mean_reward"] == 15.0
    assert "jains_fairness" in agg


def test_ippo_trainer_and_evaluator_cycle(tmp_path) -> None:
    env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=6, grid_height=6)
    env = WarehouseParallelEnv(config=env_cfg)
    ippo_cfg = IPPOConfig(num_agents=2, batch_size=100, mini_batch_size=32, eval_interval=100, eval_episodes=2)

    trainer = IPPOTrainer(env=env, config=ippo_cfg)
    trainer.train(total_timesteps=100)

    eval_metrics = trainer.evaluate(num_episodes=2)
    assert "eval_mean_reward" in eval_metrics
    assert "eval_jains_fairness" in eval_metrics

    # Test checkpoint handler
    ckpt_mgr = CheckpointManager(checkpoint_dir=str(tmp_path), config=CheckpointSubConfig())
    handler = IPPOCheckpointHandler(ckpt_mgr)
    path = handler.save_checkpoint(trainer.policy_manager, step=100)
    assert os.path.exists(path)

    meta = handler.load_checkpoint(trainer.policy_manager, path)
    assert meta["step"] == 100

    env.close()
