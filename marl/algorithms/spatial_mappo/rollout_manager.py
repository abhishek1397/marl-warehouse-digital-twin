"""SpatialMAPPORolloutManager collecting multi-channel 2D spatial grid tensors S for CNNCentralizedCritic in S-MAPPO."""

from typing import Dict, List, Optional

import torch

from marl.algorithms.mappo.shared_policy import SharedPolicyManager
from marl.algorithms.spatial_mappo.cnn_critic import CNNCentralizedCritic
from marl.algorithms.spatial_mappo.spatial_encoder import WarehouseSpatialEncoder
from marl.parallel_env import WarehouseParallelEnv
from marl.storage import Transition


class SpatialMAPPORolloutManager:
    """Collects rollout trajectories storing local actor obs and multi-channel 2D spatial state tensors for CNNCentralizedCritic."""

    def __init__(self) -> None:
        self.encoder: WarehouseSpatialEncoder = WarehouseSpatialEncoder(in_channels=5)
        self.timestep_count: int = 0

    def collect_rollouts(
        self,
        env: WarehouseParallelEnv,
        policy_manager: SharedPolicyManager,
        critic: CNNCentralizedCritic,
        num_steps: int,
    ) -> int:
        """Collects multi-agent rollout transitions into agent buffers using Spatial CNN Centralized Critic V(S_spatial)."""
        obs_dict, info_dict = env.reset()
        steps_collected = 0

        for _ in range(num_steps):
            actions_dict = {}
            log_probs_dict = {}
            values_dict = {}

            # 1. Encode 5-channel 2D spatial grid state tensor S_spatial in R^(5 x H x W)
            spatial_np = self.encoder.encode_spatial_state(
                warehouse=env._warehouse,
                fleet=env._fleet,
                charging_stations=env._charging_stations,
                shelves=env._shelves,
            )
            state_tensor = torch.from_numpy(spatial_np).float().unsqueeze(0)

            with torch.no_grad():
                val_tensor = critic(state_tensor)
                global_val = float(val_tensor.item())

                for agent_id in env.agents:
                    if agent_id in obs_dict:
                        agent = policy_manager.get_agent(agent_id)
                        agent.policy.eval()
                        obs = obs_dict[agent_id]
                        mask = info_dict.get(agent_id, {}).get("action_mask", None)

                        act, log_prob = agent.act(obs, mask=mask, deterministic=False)
                        act_int = int(act.item() if isinstance(act, torch.Tensor) and act.numel() == 1 else act)
                        log_prob_float = float(log_prob.item() if isinstance(log_prob, torch.Tensor) and log_prob.numel() == 1 else log_prob)

                        actions_dict[agent_id] = act_int
                        log_probs_dict[agent_id] = log_prob_float
                        values_dict[agent_id] = global_val

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

        # Compute GAE advantages for all unique agents using Spatial CNN Centralized Critic values
        for agent in policy_manager.get_all_agents():
            agent.compute_advantages()

        return steps_collected
