"""PPOTrainer orchestrating trajectory collection, GAE, PPO mini-batch updates, evaluation, and logging."""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from marl.algorithms.ppo.checkpoint import PPOCheckpointHandler
from marl.algorithms.ppo.config import PPOConfig
from marl.algorithms.ppo.evaluator import PPOEvaluator
from marl.algorithms.ppo.loss import PPOLoss, PPOLossOutput
from marl.algorithms.ppo.metrics import PPOMetricsTracker
from marl.algorithms.ppo.optimizer import PPOOptimizer
from marl.algorithms.ppo.scheduler import PPOLearningRateScheduler
from marl.environment import WarehouseGymEnv
from marl.networks.policy_network import PolicyNetwork
from marl.storage.rollout_buffer import RolloutBuffer
from marl.storage.transition import Transition
from marl.trainer.checkpoint_manager import CheckpointManager
from marl.trainer.config import ExperimentConfig
from marl.trainer.experiment_manager import ExperimentManager
from marl.trainer.logger import UnifiedLogger
from marl.trainer.seed import seed_everything


class PPOTrainer:
    """Production-grade single-agent PPO trainer following Schulman et al. (2017)."""

    def __init__(
        self,
        env: WarehouseGymEnv,
        config: Optional[PPOConfig] = None,
        exp_config: Optional[ExperimentConfig] = None,
    ) -> None:
        self.env: WarehouseGymEnv = env
        self.config: PPOConfig = config or PPOConfig()
        self.exp_config: ExperimentConfig = exp_config or ExperimentConfig()

        # Seed environment and PyTorch
        seed_everything(self.config.seed)

        # Setup experiment manager, logger, and checkpoint manager
        self.exp_mgr: ExperimentManager = ExperimentManager(
            base_dir="runs",
            experiment_name="ppo_run",
            config=self.exp_config,
        )
        self.logger: UnifiedLogger = UnifiedLogger(
            log_dir=self.exp_mgr.logs_dir,
            config=self.exp_config.logging,
        )
        self.ckpt_manager: CheckpointManager = CheckpointManager(
            checkpoint_dir=self.exp_mgr.checkpoints_dir,
            config=self.exp_config.checkpoint,
        )
        self.ckpt_handler: PPOCheckpointHandler = PPOCheckpointHandler(self.ckpt_manager)

        # Initialize Policy Network with shared actor-critic backbone
        self.policy: PolicyNetwork = PolicyNetwork(
            observation_space=self.env.observation_space,
            action_dim=self.env.action_space.n,
            use_shared_critic=True,
            feature_dim=self.exp_config.network.hidden_dim,
            activation=self.exp_config.network.activation,
        ).to_device(self.config.device)

        # Initialize Optimizer, Scheduler, Loss, and Storage Buffer
        self.optimizer: PPOOptimizer = PPOOptimizer(
            parameters=self.policy.parameters(),
            lr=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
        )
        self.scheduler: PPOLearningRateScheduler = PPOLearningRateScheduler(
            initial_lr=self.config.learning_rate,
            total_timesteps=self.exp_config.training.total_timesteps,
            schedule_type=self.config.scheduler_type,
        )
        self.loss_engine: PPOLoss = PPOLoss(self.config)
        self.buffer: RolloutBuffer = RolloutBuffer(
            capacity=self.config.batch_size,
            device=self.config.device,
        )
        self.evaluator: PPOEvaluator = PPOEvaluator()
        self.metrics_tracker: PPOMetricsTracker = PPOMetricsTracker()

        self.current_timestep: int = 0
        self.best_eval_reward: float = -float("inf")

    def collect_rollouts(self, num_steps: int) -> int:
        """Collects environment rollout trajectories into RolloutBuffer."""
        self.policy.eval()
        obs, info = self.env.reset()
        steps_collected = 0

        for _ in range(num_steps):
            mask = info.get("action_mask", None)
            with torch.no_grad():
                action, log_prob = self.policy.act(obs, mask=mask, deterministic=False)
                # Compute state value V(s)
                if self.policy.use_shared_critic:
                    _, val = self.policy(obs)
                    val_float = float(val.item() if val.numel() == 1 else val[0].item())
                else:
                    val_float = 0.0

            if isinstance(action, torch.Tensor):
                action_int = int(action.item() if action.numel() == 1 else action[0].item())
                log_prob_float = float(log_prob.item() if log_prob.numel() == 1 else log_prob[0].item())
            else:
                action_int = int(action)
                log_prob_float = float(log_prob)

            next_obs, reward, terminated, truncated, info = self.env.step(action_int)
            steps_collected += 1
            self.current_timestep += 1

            trans = Transition(
                observation=obs,
                action=action_int,
                reward=reward,
                next_observation=next_obs,
                terminated=terminated,
                truncated=truncated,
                value_estimate=val_float,
                log_prob=log_prob_float,
                agent_id="agent_0",
                timestep=self.current_timestep,
            )
            self.buffer.insert(trans)

            obs = next_obs
            if terminated or truncated:
                obs, info = self.env.reset()

        self.policy.train()
        return steps_collected

    def update(self) -> PPOLossOutput:
        """Performs multi-epoch mini-batch PPO optimization updates."""
        # 1. Compute GAE advantages and target returns
        self.buffer.compute_returns_and_advantages(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_adv=True,
        )

        last_loss_out: Optional[PPOLossOutput] = None

        # 2. Multi-epoch mini-batch training loop
        for batch in self.buffer.get_generator(
            mini_batch_size=self.config.mini_batch_size,
            num_epochs=self.config.epochs,
        ):
            # Compute loss
            loss_out = self.loss_engine.compute_loss(self.policy, batch)

            # Apply learning rate step from scheduler
            current_lr = self.scheduler.get_lr(self.current_timestep)
            self.optimizer.set_lr(current_lr)

            # Perform optimizer step with gradient norm clipping
            grad_norm = self.optimizer.step(loss_out.total_loss)

            # Track update metrics
            self.metrics_tracker.record_update(loss_out, grad_norm, current_lr)
            last_loss_out = loss_out

        self.buffer.clear()
        return last_loss_out if last_loss_out is not None else PPOLossOutput(
            policy_loss=torch.tensor(0.0),
            value_loss=torch.tensor(0.0),
            entropy_loss=torch.tensor(0.0),
            total_loss=torch.tensor(0.0),
            approx_kl=torch.tensor(0.0),
            clip_fraction=torch.tensor(0.0),
        )

    def train(self, total_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """Runs the main PPO training loop.

        Args:
            total_timesteps: Target total timesteps for training run.

        Returns:
            Dictionary of final training and evaluation metrics.
        """
        target_steps = total_timesteps or self.exp_config.training.total_timesteps
        self.logger.log_info(f"Starting PPO training for {target_steps:,} timesteps on device '{self.config.device}'...")

        while self.current_timestep < target_steps:
            # 1. Collect rollout batch
            self.collect_rollouts(num_steps=self.config.batch_size)

            # 2. Update PPO policy parameters
            loss_out = self.update()

            # 3. Log training metrics
            summary = self.metrics_tracker.get_summary()
            self.logger.log_metrics(summary, step=self.current_timestep)

            # 4. Periodic Evaluation
            if self.current_timestep % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(num_episodes=self.config.eval_episodes)
                self.logger.log_metrics(eval_metrics, step=self.current_timestep)

                is_best = eval_metrics["eval_mean_reward"] > self.best_eval_reward
                if is_best:
                    self.best_eval_reward = eval_metrics["eval_mean_reward"]

                # 5. Checkpoint saving
                self.save_checkpoint(step=self.current_timestep, is_best=is_best, metadata=eval_metrics)

        self.logger.log_info(f"PPO training completed successfully at step {self.current_timestep:,}.")
        self.logger.close()
        return self.metrics_tracker.get_summary()

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluates policy deterministically without exploration."""
        return self.evaluator.evaluate(
            env=self.env,
            policy=self.policy,
            num_episodes=num_episodes,
            seed=self.config.seed,
        )

    def save_checkpoint(
        self, step: int, is_best: bool = False, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Saves current PPO policy, optimizer, and scheduler states."""
        return self.ckpt_handler.save_checkpoint(
            policy=self.policy,
            optimizer=self.optimizer,
            step=step,
            is_best=is_best,
            metadata=metadata,
        )

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Loads policy and optimizer states from checkpoint path."""
        return self.ckpt_handler.load_checkpoint(path, self.policy, self.optimizer)
