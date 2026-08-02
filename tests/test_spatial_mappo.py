"""Comprehensive test suite for marl/algorithms/spatial_mappo package."""

import os
import pytest
import torch

from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.spatial_mappo import (
    CNNCentralizedCritic,
    SpatialMAPPOAgent,
    SpatialMAPPOBatchBuilder,
    SpatialMAPPOCheckpointHandler,
    SpatialMAPPOConfig,
    SpatialMAPPOEvaluator,
    SpatialFeatureVisualizer,
    SpatialMAPPOMetricsTracker,
    SpatialMAPPOTrainer,
    WarehouseSpatialEncoder,
)
from marl.trainer.checkpoint_manager import CheckpointManager
from marl.trainer.config import CheckpointSubConfig


def test_spatial_mappo_config() -> None:
    cfg = SpatialMAPPOConfig(num_agents=4, cnn_channels=5, actor_lr=1e-3)
    assert cfg.num_agents == 4
    assert cfg.cnn_channels == 5
    assert cfg.actor_lr == 1e-3


def test_warehouse_spatial_encoder() -> None:
    env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=8, grid_height=8)
    env = WarehouseParallelEnv(config=env_cfg)
    env.reset(seed=42)

    encoder = WarehouseSpatialEncoder(in_channels=5)
    spatial_arr = encoder.encode_spatial_state(
        warehouse=env._warehouse,
        fleet=env._fleet,
        charging_stations=env._charging_stations,
        shelves=env._shelves,
    )
    assert spatial_arr.shape == (5, 8, 8)
    env.close()


def test_cnn_centralized_critic_variable_grids() -> None:
    critic = CNNCentralizedCritic(in_channels=5, hidden_dim=64)

    # Test 6x6 grid
    dummy_6x6 = torch.randn(4, 5, 6, 6)
    out_6x6 = critic(dummy_6x6)
    assert out_6x6.shape == (4, 1)

    # Test 20x20 grid
    dummy_20x20 = torch.randn(2, 5, 20, 20)
    out_20x20 = critic(dummy_20x20)
    assert out_20x20.shape == (2, 1)


def test_spatial_feature_visualizer(tmp_path) -> None:
    critic = CNNCentralizedCritic(in_channels=5)
    dummy_tensor = torch.randn(1, 5, 8, 8)
    out_img = SpatialFeatureVisualizer.visualize_activation_maps(
        critic=critic,
        spatial_tensor=dummy_tensor,
        output_dir=str(tmp_path),
        filename="test_act.png",
    )
    assert os.path.exists(out_img)


def test_spatial_mappo_metrics_tracker() -> None:
    fair = SpatialMAPPOMetricsTracker.compute_jains_fairness([10.0, 10.0])
    assert pytest.approx(fair, 1e-3) == 1.0

    agg = SpatialMAPPOMetricsTracker.aggregate_multi_agent_metrics({"r0": 5.0, "r1": 15.0}, {"r0": 0.2}, critic_loss=0.1)
    assert agg["mean_reward"] == 10.0
    assert agg["cnn_critic_loss"] == 0.1


def test_spatial_mappo_trainer_and_evaluator_cycle(tmp_path) -> None:
    env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=6, grid_height=6)
    env = WarehouseParallelEnv(config=env_cfg)
    smappo_cfg = SpatialMAPPOConfig(num_agents=2, batch_size=100, mini_batch_size=32, eval_interval=100, eval_episodes=2)

    trainer = SpatialMAPPOTrainer(env=env, config=smappo_cfg)
    trainer.train(total_timesteps=100)

    eval_metrics = trainer.evaluate(num_episodes=2)
    assert "eval_mean_reward" in eval_metrics
    assert "eval_jains_fairness" in eval_metrics

    # Test checkpoint handler
    ckpt_mgr = CheckpointManager(checkpoint_dir=str(tmp_path), config=CheckpointSubConfig())
    handler = SpatialMAPPOCheckpointHandler(ckpt_mgr)
    path = handler.save_checkpoint(trainer.policy_manager, trainer.cnn_centralized_critic, step=100)
    assert os.path.exists(path)

    payload = handler.load_checkpoint(trainer.policy_manager, trainer.cnn_centralized_critic, path)
    assert payload["step"] == 100

    env.close()
