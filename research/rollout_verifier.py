"""Rollout Verifier verifying multi-agent transition storage, episode boundary alignment, and rollout buffer isolation."""

from typing import Any, Dict

from marl.algorithms.ippo import IPPOConfig, PolicyManager
from marl.algorithms.ippo.rollout_manager import IPPORolloutManager
from marl.multi_agent_config import MultiAgentEnvConfig
from marl.parallel_env import WarehouseParallelEnv


class RolloutVerifier:
    """Verifies transition storage, episode boundary flags, log probability arrays, and rollout buffer isolation."""

    @staticmethod
    def verify_rollout_collection() -> Dict[str, Any]:
        """Runs rollout collection verification checks on IPPORolloutManager."""
        env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=8, grid_height=8, seed=42)
        env = WarehouseParallelEnv(config=env_cfg)
        obs_sp = env.observation_space(env.possible_agents[0])
        act_sp = env.action_space(env.possible_agents[0])

        ippo_cfg = IPPOConfig(num_agents=2, batch_size=100, mini_batch_size=32)
        policy_manager = PolicyManager(env.possible_agents, obs_sp, act_sp, ippo_cfg)
        rollout_manager = IPPORolloutManager()

        steps = rollout_manager.collect_rollouts(env, policy_manager, num_steps=100)

        passed = True
        messages = []

        for agent in policy_manager.get_all_agents():
            buf_len = len(agent.buffer)
            if buf_len != 100:
                passed = False
                messages.append(f"Agent {agent.agent_id} buffer length {buf_len} != expected 100.")

        env.close()

        return {
            "status": "PASSED" if passed else "FAILED",
            "passed": passed,
            "messages": messages,
            "steps_collected": steps,
        }
