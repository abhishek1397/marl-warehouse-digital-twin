"""SharedPolicyManager managing shared actor network parameters across MAPPO agents."""

from typing import Dict, List

from gymnasium.spaces import Space

from marl.algorithms.mappo.agent import MAPPOAgent
from marl.algorithms.mappo.config import MAPPOConfig
from marl.algorithms.ppo.loss import PPOLossOutput
from marl.networks.policy_network import PolicyNetwork


class SharedPolicyManager:
    """Manages actor policy instantiation and lookup for MAPPO under shared parameter mode."""

    def __init__(
        self,
        agent_ids: List[str],
        observation_space: Space,
        action_space: Space,
        config: MAPPOConfig,
    ) -> None:
        self.agent_ids: List[str] = agent_ids
        self.config: MAPPOConfig = config
        self.agents: Dict[str, MAPPOAgent] = {}

        if self.config.shared_policy:
            # Mode: Shared actor parameters across all robots
            shared_policy = PolicyNetwork(
                observation_space=observation_space,
                action_dim=action_space.n,
                use_shared_critic=False,
                feature_dim=64,
            )
            shared_agent = MAPPOAgent(
                agent_id="shared_actor",
                policy=shared_policy,
                config=config,
            )
            for agent_id in agent_ids:
                self.agents[agent_id] = shared_agent
        else:
            # Mode: Independent actor parameters per robot
            for agent_id in agent_ids:
                indep_policy = PolicyNetwork(
                    observation_space=observation_space,
                    action_dim=action_space.n,
                    use_shared_critic=False,
                    feature_dim=64,
                )
                self.agents[agent_id] = MAPPOAgent(
                    agent_id=agent_id,
                    policy=indep_policy,
                    config=config,
                )

    def get_agent(self, agent_id: str) -> MAPPOAgent:
        """Retrieves MAPPOAgent instance for a given agent_id."""
        return self.agents[agent_id]

    def get_all_agents(self) -> List[MAPPOAgent]:
        """Returns unique MAPPOAgent instances."""
        if self.config.shared_policy:
            return [self.agents[self.agent_ids[0]]]
        return [self.agents[agent_id] for agent_id in self.agent_ids]

    def update_all(self) -> Dict[str, PPOLossOutput]:
        """Performs PPO optimization update across all unique agents."""
        loss_dict: Dict[str, PPOLossOutput] = {}
        unique_agents = self.get_all_agents()
        for agent in unique_agents:
            loss_dict[agent.agent_id] = agent.update()
        return loss_dict
