"""IPPORolloutManager collecting multi-agent rollout trajectories on PettingZoo WarehouseParallelEnv."""

from typing import Dict, List

import torch

from marl.algorithms.ippo.policy_manager import PolicyManager
from marl.parallel_env import WarehouseParallelEnv
from marl.storage import Transition


class IPPORolloutManager:
    """Collects trajectory rollouts across all agents in PettingZoo WarehouseParallelEnv."""

    def __init__(self) -> None:
        self.timestep_count: int = 0

    def collect_rollouts(
        self,
        env: WarehouseParallelEnv,
        policy_manager: PolicyManager,
        num_steps: int,
    ) -> int:
        """Collects multi-agent rollout transitions into agent buffers."""
        obs_dict, info_dict = env.reset()
        steps_collected = 0

        for _ in range(num_steps):
            actions_dict = {}
            log_probs_dict = {}
            values_dict = {}

            with torch.no_grad():
                for agent_id in env.agents:
                    if agent_id in obs_dict:
                        agent = policy_manager.get_agent(agent_id)
                        agent.policy.eval()
                        obs = obs_dict[agent_id]
                        mask = info_dict.get(agent_id, {}).get("action_mask", None)

                        act, log_prob = agent.act(obs, mask=mask, deterministic=False)
                        act_int = int(act.item() if isinstance(act, torch.Tensor) and act.numel() == 1 else act)
                        log_prob_float = float(log_prob.item() if isinstance(log_prob, torch.Tensor) and log_prob.numel() == 1 else log_prob)

                        # Value estimate
                        if agent.policy.use_shared_critic:
                            _, val = agent.policy(obs)
                            val_float = float(val.item() if isinstance(val, torch.Tensor) and val.numel() == 1 else val)
                        else:
                            val_float = 0.0

                        actions_dict[agent_id] = act_int
                        log_probs_dict[agent_id] = log_prob_float
                        values_dict[agent_id] = val_float

            next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
            steps_collected += 1
            self.timestep_count += 1

            for agent_id in env.possible_agents:
                if agent_id in obs_dict and agent_id in actions_dict:
                    agent = policy_manager.get_agent(agent_id)
                    term = terminations.get(agent_id, False)
                    trunc = truncations.get(agent_id, False)
                    rew = float(rewards.get(agent_id, 0.0))

                    trans = Transition(
                        observation=obs_dict[agent_id],
                        action=actions_dict[agent_id],
                        reward=rew,
                        next_observation=next_obs.get(agent_id, obs_dict[agent_id]),
                        terminated=term,
                        truncated=trunc,
                        value_estimate=values_dict[agent_id],
                        log_prob=log_probs_dict[agent_id],
                        agent_id=agent_id,
                        timestep=self.timestep_count,
                    )
                    agent.insert_transition(trans)

            obs_dict, info_dict = next_obs, infos
            if not env.agents:
                obs_dict, info_dict = env.reset()

        # Compute GAE advantages for all unique agents
        for agent in policy_manager.get_all_agents():
            agent.compute_advantages(last_value=0.0)

        return steps_collected
