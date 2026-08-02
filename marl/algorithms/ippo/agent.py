"""IPPOAgent module encapsulating an individual agent's policy, optimizer, buffer, and loss engine."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from gymnasium.spaces import Space

from marl.algorithms.ippo.config import IPPOConfig
from marl.algorithms.ppo.config import PPOConfig
from marl.algorithms.ppo.loss import PPOLoss, PPOLossOutput
from marl.algorithms.ppo.optimizer import PPOOptimizer
from marl.algorithms.ppo.scheduler import PPOLearningRateScheduler
from marl.networks.policy_network import PolicyNetwork
from marl.storage import RolloutBuffer, Transition


class IPPOAgent:
    """Encapsulates an independent PPO agent owning policy, optimizer, rollout buffer, and loss engine."""

    def __init__(
        self,
        agent_id: str,
        policy: PolicyNetwork,
        config: IPPOConfig,
    ) -> None:
        self.agent_id: str = agent_id
        self.config: IPPOConfig = config
        self.policy: PolicyNetwork = policy.to_device(config.device)

        self.ppo_config: PPOConfig = PPOConfig(
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.batch_size,
            mini_batch_size=config.mini_batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_eps=config.clip_eps,
            entropy_coef=config.entropy_coef,
            value_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            device=config.device,
        )

        self.optimizer: PPOOptimizer = PPOOptimizer(
            parameters=self.policy.parameters(),
            lr=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
        )

        self.scheduler: PPOLearningRateScheduler = PPOLearningRateScheduler(
            initial_lr=config.learning_rate,
        )

        self.buffer: RolloutBuffer = RolloutBuffer(
            capacity=config.batch_size,
            device=config.device,
        )

        self.loss_engine: PPOLoss = PPOLoss(self.ppo_config)

    def act(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        mask: Optional[Union[torch.Tensor, Any]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects action and returns (action_tensor, log_prob_tensor)."""
        return self.policy.act(obs, mask=mask, deterministic=deterministic)

    def predict(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        mask: Optional[Union[torch.Tensor, Any]] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Inference helper returning action tensor."""
        return self.policy.predict(obs, mask=mask, deterministic=deterministic)

    def insert_transition(self, transition: Transition) -> None:
        """Inserts step transition into agent rollout buffer."""
        self.buffer.insert(transition)

    def compute_advantages(self, last_value: float = 0.0) -> None:
        """Computes GAE advantages and target returns for agent trajectory."""
        self.buffer.compute_returns_and_advantages(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_adv=True,
        )

    def update(self) -> PPOLossOutput:
        """Performs PPO optimization update on agent policy."""
        last_loss = PPOLossOutput(
            policy_loss=torch.tensor(0.0),
            value_loss=torch.tensor(0.0),
            entropy_loss=torch.tensor(0.0),
            total_loss=torch.tensor(0.0),
            approx_kl=torch.tensor(0.0),
            clip_fraction=torch.tensor(0.0),
        )

        for batch in self.buffer.get_generator(
            mini_batch_size=self.config.mini_batch_size,
            num_epochs=self.config.epochs,
        ):
            loss_out = self.loss_engine.compute_loss(self.policy, batch)
            self.optimizer.step(loss_out.total_loss)
            last_loss = loss_out

        self.buffer.clear()
        return last_loss
