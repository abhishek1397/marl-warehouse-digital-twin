"""IPPOTrainer module orchestrating multi-agent rollout collection, independent PPO updates, logging, and evaluation."""

import os
from typing import Dict, Optional

from marl.algorithms.ippo.checkpoint import IPPOCheckpointHandler
from marl.algorithms.ippo.config import IPPOConfig
from marl.algorithms.ippo.evaluator import IPPOEvaluator
from marl.algorithms.ippo.metrics import IPPOMetricsTracker
from marl.algorithms.ippo.policy_manager import PolicyManager
from marl.algorithms.ippo.rollout_manager import IPPORolloutManager
from marl.parallel_env import WarehouseParallelEnv
from marl.trainer.checkpoint_manager import CheckpointManager
from marl.trainer.config import ExperimentConfig
from marl.trainer.experiment_manager import ExperimentManager
from marl.trainer.logger import UnifiedLogger
from marl.trainer.seed import seed_everything


class IPPOTrainer:
    """Orchestrates Independent PPO (IPPO) multi-agent training, rollout collection, independent policy updates, and evaluation."""

    def __init__(
        self,
        env: WarehouseParallelEnv,
        config: Optional[IPPOConfig] = None,
    ) -> None:
        self.env: WarehouseParallelEnv = env
        self.config: IPPOConfig = config or IPPOConfig(num_agents=len(env.possible_agents))

        # Seed environment and PyTorch
        seed_everything(self.config.seed)

        # Multi-Agent Policy Manager
        obs_space = self.env.observation_space(self.env.possible_agents[0])
        act_space = self.env.action_space(self.env.possible_agents[0])

        self.policy_manager: PolicyManager = PolicyManager(
            agent_ids=list(self.env.possible_agents),
            observation_space=obs_space,
            action_space=act_space,
            config=self.config,
        )

        self.rollout_manager: IPPORolloutManager = IPPORolloutManager()
        self.evaluator: IPPOEvaluator = IPPOEvaluator(env=self.env)

        # Setup infrastructure logging and checkpointing
        self.exp_config: ExperimentConfig = ExperimentConfig(seed=self.config.seed or 42)
        self.exp_mgr: ExperimentManager = ExperimentManager(
            base_dir="runs",
            experiment_name="ippo_run",
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
        self.ckpt_handler: IPPOCheckpointHandler = IPPOCheckpointHandler(self.ckpt_manager)

        self.current_timestep: int = 0

    def train(self, total_timesteps: int = 10000) -> None:
        """Executes multi-agent IPPO training loop across specified total timesteps."""
        self.logger.log_info(f"Starting IPPO training for {total_timesteps:,} timesteps ({self.config.num_agents} agents, Shared={self.config.shared_policy})...")

        while self.current_timestep < total_timesteps:
            steps_collected = self.rollout_manager.collect_rollouts(
                env=self.env,
                policy_manager=self.policy_manager,
                num_steps=self.config.batch_size,
            )
            self.current_timestep += steps_collected

            # Update all agent policies independently
            loss_dict = self.policy_manager.update_all()

            # Log metrics
            first_agent_id = list(loss_dict.keys())[0]
            loss_out = loss_dict[first_agent_id]
            metrics = {
                "policy_loss": loss_out.policy_loss,
                "value_loss": loss_out.value_loss,
                "entropy": loss_out.entropy_loss,
                "total_loss": loss_out.total_loss,
                "approx_kl": loss_out.approx_kl,
                "clip_fraction": loss_out.clip_fraction,
            }
            self.logger.log_metrics(metrics, step=self.current_timestep)

            # Evaluation check
            if self.current_timestep % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(num_episodes=self.config.eval_episodes)
                self.logger.log_metrics(eval_metrics, step=self.current_timestep)

        self.logger.log_info(f"IPPO training completed successfully at step {self.current_timestep:,}.")

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluates current multi-agent policies."""
        return self.evaluator.evaluate(
            policy_manager=self.policy_manager,
            num_episodes=num_episodes,
            seed=self.config.seed,
        )
