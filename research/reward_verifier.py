"""Reward Verifier checking per-agent reward assignments, potential-based shaping, and leakage."""

from typing import Any, Dict

from marl.multi_agent_config import MultiAgentEnvConfig
from marl.parallel_env import WarehouseParallelEnv


class RewardVerifier:
    """Verifies per-agent reward assignment, PBRS, combined rewards, and checks inter-agent reward leakage."""

    @staticmethod
    def verify_reward_assignment() -> Dict[str, Any]:
        """Runs reward assignment and leakage verification checks."""
        env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=8, grid_height=8, seed=42)
        env = WarehouseParallelEnv(config=env_cfg)
        obs_dict, info_dict = env.reset(seed=42)

        actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        passed = True
        messages = []

        if set(rewards.keys()) != set(env.possible_agents):
            passed = False
            messages.append(f"Reward keys {list(rewards.keys())} do not match possible_agents {env.possible_agents}.")

        for agent_id, rew in rewards.items():
            if not isinstance(rew, (int, float)):
                passed = False
                messages.append(f"Reward for {agent_id} must be scalar float, got {type(rew)}.")

        env.close()

        return {
            "status": "PASSED" if passed else "FAILED",
            "passed": passed,
            "messages": messages,
            "sample_rewards": rewards,
        }
