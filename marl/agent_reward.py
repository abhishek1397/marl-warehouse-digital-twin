"""AgentRewardEngine for individual, team, and hybrid multi-agent reward allocation."""

from typing import Dict

from marl.action import ActionResult
from marl.multi_agent_config import MultiAgentEnvConfig
from marl.reward import RewardEngine
from simulator.robot import Robot


class AgentRewardEngine:
    """Computes individual, team, and hybrid multi-agent reward dictionaries."""

    def __init__(self, config: MultiAgentEnvConfig) -> None:
        self.config: MultiAgentEnvConfig = config
        self.single_reward_engine: RewardEngine = RewardEngine(config)

    def calculate_joint_rewards(
        self, action_results: Dict[str, ActionResult], fleet: Dict[str, Robot]
    ) -> Dict[str, float]:
        """Calculates step rewards for all active agents based on reward_mode.

        Modes:
            - "individual": Each agent receives its own individual reward.
            - "team": All agents receive the mean team reward.
            - "hybrid": Weighted combination of individual and team rewards.

        Returns:
            Dict mapping agent_id -> scalar reward float.
        """
        indiv_rewards: Dict[str, float] = {}

        for agent_id, robot in fleet.items():
            res = action_results.get(agent_id, ActionResult(action=4))
            indiv_rewards[agent_id] = self.single_reward_engine.calculate_reward(res, robot)

        if self.config.reward_mode == "individual":
            return indiv_rewards

        # Calculate mean team reward
        total_fleet_size = len(fleet)
        mean_team_reward = (
            sum(indiv_rewards.values()) / total_fleet_size if total_fleet_size > 0 else 0.0
        )

        if self.config.reward_mode == "team":
            return {a_id: mean_team_reward for a_id in fleet}

        # Hybrid mode: (1 - w) * individual + w * team
        w = self.config.team_reward_weight
        hybrid_rewards: Dict[str, float] = {}
        for a_id, indiv_r in indiv_rewards.items():
            hybrid_rewards[a_id] = (1.0 - w) * indiv_r + w * mean_team_reward

        return hybrid_rewards
