"""PettingZoo API Verifier checking WarehouseParallelEnv compliance with PettingZoo Parallel API standards."""

from typing import Any, Dict, List, Tuple

import numpy as np

from marl.multi_agent_config import MultiAgentEnvConfig
from marl.parallel_env import WarehouseParallelEnv


class PettingZooAPIVerifier:
    """Verifies PettingZoo Parallel API compliance for WarehouseParallelEnv."""

    @staticmethod
    def verify_api_compliance() -> Dict[str, Any]:
        """Runs compliance checks on reset, step, observation spaces, and action spaces."""
        env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=8, grid_height=8, seed=42)
        env = WarehouseParallelEnv(config=env_cfg)

        passed = True
        messages = []

        # 1. Test reset()
        try:
            obs_dict, info_dict = env.reset(seed=42)
            if not isinstance(obs_dict, dict):
                passed = False
                messages.append("reset() must return an observation dictionary.")
            if set(obs_dict.keys()) != set(env.agents):
                passed = False
                messages.append(f"reset() keys {list(obs_dict.keys())} do not match env.agents {env.agents}.")
        except Exception as e:
            passed = False
            messages.append(f"reset() raised exception: {str(e)}")

        # 2. Test step()
        try:
            actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            if not isinstance(next_obs, dict) or not isinstance(rewards, dict):
                passed = False
                messages.append("step() outputs must be dictionaries.")
            if not isinstance(terminations, dict) or not isinstance(truncations, dict):
                passed = False
                messages.append("terminations and truncations must be dictionaries.")
        except Exception as e:
            passed = False
            messages.append(f"step() raised exception: {str(e)}")

        # 3. Test spaces
        try:
            for agent_id in env.possible_agents:
                obs_sp = env.observation_space(agent_id)
                act_sp = env.action_space(agent_id)
                if obs_sp is None or act_sp is None:
                    passed = False
                    messages.append(f"Spaces for {agent_id} must not be None.")
        except Exception as e:
            passed = False
            messages.append(f"Space verification raised exception: {str(e)}")

        env.close()

        status_str = "PASSED" if passed else "FAILED"
        return {
            "status": status_str,
            "passed": passed,
            "messages": messages,
            "num_agents_tested": 2,
        }
