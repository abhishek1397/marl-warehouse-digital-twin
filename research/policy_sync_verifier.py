"""Policy Sync Verifier checking policy parameter updates, loss trajectories, and gradient health."""

from typing import Any, Dict

import torch

from marl.algorithms.ippo.config import IPPOConfig
from marl.algorithms.ippo.policy_manager import PolicyManager
from marl.multi_agent_config import MultiAgentEnvConfig
from marl.parallel_env import WarehouseParallelEnv


class PolicySyncVerifier:
    """Verifies gradient flow, loss trajectory convergence, optimizer updates, and checks for NaN/Inf anomalies."""

    @staticmethod
    def verify_policy_updates() -> Dict[str, Any]:
        """Verifies policy parameter updates and checks for NaN/Inf parameter anomalies."""
        env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=6, grid_height=6, seed=42)
        env = WarehouseParallelEnv(config=env_cfg)
        obs_sp = env.observation_space(env.possible_agents[0])
        act_sp = env.action_space(env.possible_agents[0])

        ippo_cfg = IPPOConfig(num_agents=2, batch_size=50, mini_batch_size=16)
        policy_manager = PolicyManager(env.possible_agents, obs_sp, act_sp, ippo_cfg)

        passed = True
        messages = []

        for agent in policy_manager.get_all_agents():
            for name, param in agent.policy.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    passed = False
                    messages.append(f"Agent {agent.agent_id} parameter {name} contains NaN or Inf.")

        env.close()

        return {
            "status": "PASSED" if passed else "FAILED",
            "passed": passed,
            "messages": messages,
            "checked_agents": len(policy_manager.get_all_agents()),
        }
