"""MAPPOTrainer orchestrating MAPPO CTDE rollout collection, centralized value estimation, updates, logging, and evaluation."""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from marl.algorithms.mappo.centralized_critic import CentralizedValueNetwork
from marl.algorithms.mappo.checkpoint import MAPPOCheckpointHandler
from marl.algorithms.mappo.config import MAPPOConfig
from marl.algorithms.mappo.evaluator import MAPPOEvaluator
from marl.algorithms.mappo.metrics import MAPPOMetricsTracker
from marl.algorithms.mappo.rollout_manager import MAPPORolloutManager
from marl.algorithms.mappo.shared_policy import SharedPolicyManager
from marl.parallel_env import WarehouseParallelEnv
from marl.trainer.checkpoint_manager import CheckpointManager
from marl.trainer.config import ExperimentConfig
from marl.trainer.experiment_manager import ExperimentManager
from marl.trainer.logger import UnifiedLogger
from marl.trainer.seed import seed_everything


class MAPPOTrainer:
    """Orchestrates Multi-Agent PPO (MAPPO) CTDE training, centralized value network optimization, and evaluation."""

    def __init__(
        self,
        env: WarehouseParallelEnv,
        config: Optional[MAPPOConfig] = None,
    ) -> None:
        self.env: WarehouseParallelEnv = env
        self.config: MAPPOConfig = config or MAPPOConfig(num_agents=len(env.possible_agents))

        # Seed environment and PyTorch
        seed_everything(self.config.seed)

        # Multi-Agent Shared Policy Manager (Decentralized Actors)
        obs_space = self.env.observation_space(self.env.possible_agents[0])
        act_space = self.env.action_space(self.env.possible_agents[0])

        self.policy_manager: SharedPolicyManager = SharedPolicyManager(
            agent_ids=list(self.env.possible_agents),
            observation_space=obs_space,
            action_space=act_space,
            config=self.config,
        )

        # Centralized Value Network V(S)
        global_state_shape = self.env.state().shape
        state_dim = int(torch.tensor(global_state_shape).prod().item())
        self.centralized_critic: CentralizedValueNetwork = CentralizedValueNetwork(
            state_dim=state_dim,
            hidden_dim=self.config.critic_hidden_dim,
        ).to(self.config.device)

        self.critic_optimizer: Adam = Adam(self.centralized_critic.parameters(), lr=self.config.critic_lr)
        self.critic_loss_fn: nn.MSELoss = nn.MSELoss()

        self.rollout_manager: MAPPORolloutManager = MAPPORolloutManager()
        self.evaluator: MAPPOEvaluator = MAPPOEvaluator(env=self.env)

        # Setup infrastructure logging and checkpointing
        self.exp_config: ExperimentConfig = ExperimentConfig(seed=self.config.seed or 42)
        self.exp_mgr: ExperimentManager = ExperimentManager(
            base_dir="runs",
            experiment_name="mappo_run",
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
        self.ckpt_handler: MAPPOCheckpointHandler = MAPPOCheckpointHandler(self.ckpt_manager)

        self.current_timestep: int = 0

    def train(self, total_timesteps: int = 10000) -> None:
        """Executes multi-agent MAPPO CTDE training loop across specified total timesteps."""
        self.logger.log_info(f"Starting MAPPO CTDE training for {total_timesteps:,} timesteps ({self.config.num_agents} agents, Centralized Critic=True)...")

        while self.current_timestep < total_timesteps:
            steps_collected = self.rollout_manager.collect_rollouts(
                env=self.env,
                policy_manager=self.policy_manager,
                critic=self.centralized_critic,
                num_steps=self.config.batch_size,
            )
            self.current_timestep += steps_collected

            # 1. Update Centralized Value Network V(S)
            global_state = self.env.state()
            state_tensor = torch.from_numpy(global_state).float().unsqueeze(0).to(self.config.device)
            val_pred = self.centralized_critic(state_tensor)

            first_agent = self.policy_manager.get_all_agents()[0]
            target_returns = first_agent.buffer.returns[:1].to(self.config.device) if first_agent.buffer.returns is not None else torch.zeros(1, device=self.config.device)

            critic_loss = self.critic_loss_fn(val_pred.view(-1), target_returns.view(-1))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.centralized_critic.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()

            # 2. Update shared decentralized actor policies
            loss_dict = self.policy_manager.update_all()

            # Log metrics
            first_agent_id = list(loss_dict.keys())[0]
            loss_out = loss_dict[first_agent_id]
            metrics = {
                "actor_loss": loss_out.policy_loss,
                "critic_loss": float(critic_loss.item()),
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

        self.logger.log_info(f"MAPPO CTDE training completed successfully at step {self.current_timestep:,}.")

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluates current MAPPO decentralized actor policies."""
        return self.evaluator.evaluate(
            policy_manager=self.policy_manager,
            num_episodes=num_episodes,
            seed=self.config.seed,
        )
