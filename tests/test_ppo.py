"""Comprehensive test suite for marl/algorithms/ppo package."""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from marl import EnvConfig, WarehouseGymEnv
from marl.algorithms.ppo import (
    PPOCheckpointHandler,
    PPOConfig,
    PPOEvaluator,
    PPOLoss,
    PPOLossOutput,
    PPOMetricsTracker,
    PPOOptimizer,
    PPOLearningRateScheduler,
    PPOTrainer,
)
from marl.algorithms.ppo.utils import clip_advantages, compute_explained_variance
from marl.networks.policy_network import PolicyNetwork
from marl.storage.batch import Batch
from marl.trainer.checkpoint_manager import CheckpointManager


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


def test_ppo_config() -> None:
    cfg = PPOConfig(learning_rate=1e-4, clip_eps=0.1)
    assert cfg.learning_rate == 1e-4
    assert cfg.clip_eps == 0.1
    assert cfg.gamma == 0.99


def test_ppo_loss_computation() -> None:
    cfg = PPOConfig()
    loss_engine = PPOLoss(cfg)

    policy = PolicyNetwork(observation_space=16, action_dim=4, use_shared_critic=True)

    obs = torch.randn(8, 16)
    actions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    advantages = torch.randn(8)
    returns = torch.randn(8)
    values = torch.randn(8)
    old_log_probs = torch.randn(8)
    masks = torch.ones(8)

    batch = Batch(
        observations=obs,
        actions=actions,
        advantages=advantages,
        returns=returns,
        values=values,
        old_log_probs=old_log_probs,
        masks=masks,
    )

    loss_out = loss_engine.compute_loss(policy, batch)
    assert isinstance(loss_out, PPOLossOutput)
    assert loss_out.total_loss.ndim == 0
    assert not torch.isnan(loss_out.total_loss)


def test_ppo_optimizer() -> None:
    param = torch.nn.Parameter(torch.tensor([10.0, 20.0], requires_grad=True))
    opt = PPOOptimizer(parameters=[param], lr=0.01, max_grad_norm=0.5)

    loss = (param ** 2).sum()
    grad_norm = opt.step(loss)

    assert grad_norm > 0.0
    assert opt.get_lr() == 0.01

    opt.set_lr(0.001)
    assert opt.get_lr() == 0.001


def test_learning_rate_scheduler() -> None:
    # Constant
    s_const = PPOLearningRateScheduler(initial_lr=0.01, total_timesteps=100, schedule_type="constant")
    assert s_const.get_lr(50) == 0.01

    # Linear
    s_lin = PPOLearningRateScheduler(initial_lr=0.01, total_timesteps=100, schedule_type="linear")
    assert s_lin.get_lr(0) == 0.01
    assert abs(s_lin.get_lr(50) - 0.005) < 1e-5
    assert s_lin.get_lr(100) == 0.0

    # Cosine
    s_cos = PPOLearningRateScheduler(initial_lr=0.01, total_timesteps=100, schedule_type="cosine")
    assert s_cos.get_lr(0) == 0.01
    assert s_cos.get_lr(100) == 0.0


def test_ppo_evaluator() -> None:
    env_cfg = EnvConfig(grid_width=6, grid_height=6, max_episode_steps=20, seed=42)
    env = WarehouseGymEnv(config=env_cfg)

    policy = PolicyNetwork(observation_space=env.observation_space, action_dim=env.action_space.n)
    evaluator = PPOEvaluator()

    metrics = evaluator.evaluate(env, policy, num_episodes=2, seed=42)
    assert "eval_mean_reward" in metrics
    assert "eval_success_rate" in metrics
    env.close()


def test_ppo_checkpoint_handler(temp_dir) -> None:
    ckpt_mgr = CheckpointManager(checkpoint_dir=temp_dir)
    handler = PPOCheckpointHandler(ckpt_mgr)

    policy = PolicyNetwork(observation_space=16, action_dim=4)
    param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    optimizer = PPOOptimizer(parameters=[param])

    saved_path = handler.save_checkpoint(policy=policy, optimizer=optimizer, step=10, is_best=True)
    assert os.path.exists(saved_path)

    loaded_payload = handler.load_checkpoint(saved_path, policy=policy, optimizer=optimizer)
    assert loaded_payload["step"] == 10


def test_ppo_metrics_tracker() -> None:
    tracker = PPOMetricsTracker()
    loss_out = PPOLossOutput(
        policy_loss=torch.tensor(0.1),
        value_loss=torch.tensor(0.2),
        entropy_loss=torch.tensor(0.3),
        total_loss=torch.tensor(0.4),
        approx_kl=torch.tensor(0.01),
        clip_fraction=torch.tensor(0.05),
    )

    tracker.record_update(loss_out, grad_norm=0.5, lr=0.0003)
    summary = tracker.get_summary()

    assert summary["policy_loss"] == pytest.approx(0.1, 1e-4)
    assert summary["learning_rate"] == pytest.approx(0.0003, 1e-4)

    tracker.reset()
    assert tracker.get_summary()["policy_loss"] == 0.0


def test_ppo_utils() -> None:
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_pred = torch.tensor([1.1, 1.9, 3.1, 3.9])
    exp_var = compute_explained_variance(y_pred, y_true)
    assert exp_var > 0.9

    advs = torch.tensor([-20.0, 0.0, 20.0])
    clipped = clip_advantages(advs, max_val=10.0)
    assert clipped[0].item() == -10.0
    assert clipped[2].item() == 10.0


def test_ppo_trainer_full_cycle() -> None:
    env_cfg = EnvConfig(grid_width=6, grid_height=6, max_episode_steps=20, seed=42)
    env = WarehouseGymEnv(config=env_cfg)

    ppo_cfg = PPOConfig(
        learning_rate=1e-3,
        epochs=2,
        batch_size=40,
        mini_batch_size=10,
        eval_interval=20,
        eval_episodes=2,
        seed=42,
    )

    trainer = PPOTrainer(env=env, config=ppo_cfg)
    steps = trainer.collect_rollouts(num_steps=40)
    assert steps == 40

    loss_out = trainer.update()
    assert isinstance(loss_out, PPOLossOutput)

    eval_m = trainer.evaluate(num_episodes=1)
    assert "eval_mean_reward" in eval_m

    # Run short training run
    train_summary = trainer.train(total_timesteps=80)
    assert isinstance(train_summary, dict)

    env.close()
