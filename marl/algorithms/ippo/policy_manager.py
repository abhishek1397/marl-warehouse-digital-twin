"""PolicyManager module handling independent and shared parameter policy modes for IPPO."""

from typing import Dict, List

from gymnasium.spaces import Space

from marl.algorithms.ippo.agent import IPPOAgent
from marl.algorithms.ippo.config import IPPOConfig
from marl.algorithms.ppo.loss import PPOLossOutput
from marl.networks.policy_network import PolicyNetwork


class PolicyManager:
    """Manages multi-agent policy instantiation and lookup under Independent or Shared parameter modes."""

    def __init__(
        self,
        agent_ids: List[str],
        observation_space: Space,
        action_space: Space,
        config: IPPOConfig,
    ) -> None:
        self.agent_ids: List[str] = agent_ids
        self.config: IPPOConfig = config
        self.agents: Dict[str, IPPOAgent] = {}

        if self.config.shared_policy:
            # Mode 2: Shared parameters across all agents
            shared_policy = PolicyNetwork(
                observation_space=observation_space,
                action_dim=action_space.n,
                use_shared_critic=True,
                feature_dim=64,
            )
            shared_agent = IPPOAgent(
                agent_id="shared",
                policy=shared_policy,
                config=config,
            )
            for agent_id in agent_ids:
                self.agents[agent_id] = shared_agent
        else:
            # Mode 1: Independent parameters per agent
            for agent_id in agent_ids:
                indep_policy = PolicyNetwork(
                    observation_space=observation_space,
                    action_dim=action_space.n,
                    use_shared_critic=True,
                    feature_dim=64,
                )
                self.agents[agent_id] = IPPOAgent(
                    agent_id=agent_id,
                    policy=indep_policy,
                    config=config,
                )

    def get_agent(self, agent_id: str) -> IPPOAgent:
        """Retrieves IPPOAgent instance for a given agent_id."""
        return self.agents[agent_id]

    def get_all_agents(self) -> List[IPPOAgent]:
        """Returns unique IPPOAgent instances."""
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
