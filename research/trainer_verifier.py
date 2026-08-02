"""Trainer Verifier verifying single-agent Gym PPO vs 1-robot PettingZoo IPPO equivalence and multi-robot scaling."""

from typing import Any, Dict

from marl.algorithms.ippo import IPPOConfig, IPPOTrainer
from marl.algorithms.ppo.config import PPOConfig
from marl.algorithms.ppo.trainer import PPOTrainer
from marl.config import EnvConfig
from marl.environment import WarehouseGymEnv
from marl.multi_agent_config import MultiAgentEnvConfig
from marl.parallel_env import WarehouseParallelEnv


class SingleAgentEquivalenceVerifier:
    """Verifies single-agent Gymnasium PPO baseline vs 1-robot PettingZoo IPPO equivalence."""

    @staticmethod
    def verify_single_agent_equivalence(timesteps: int = 1000, seed: int = 42) -> Dict[str, Any]:
        """Compares Gymnasium WarehouseGymEnv PPO vs PettingZoo WarehouseParallelEnv IPPO (1 robot)."""
        # 1. Single-Agent Gym PPO
        gym_cfg = EnvConfig(grid_width=8, grid_height=8, seed=seed, enable_reward_shaping=True, enable_action_masking=True)
        gym_env = WarehouseGymEnv(config=gym_cfg)
        ppo_cfg = PPOConfig(learning_rate=3e-4, epochs=2, batch_size=200, mini_batch_size=64, seed=seed)
        ppo_trainer = PPOTrainer(env=gym_env, config=ppo_cfg)
        ppo_trainer.train(total_timesteps=timesteps)
        ppo_eval = ppo_trainer.evaluate(num_episodes=3)
        gym_env.close()

        # 2. 1-Robot PettingZoo IPPO
        pz_cfg = MultiAgentEnvConfig(num_robots=1, grid_width=8, grid_height=8, seed=seed)
        pz_env = WarehouseParallelEnv(config=pz_cfg)
        ippo_cfg = IPPOConfig(num_agents=1, learning_rate=3e-4, epochs=2, batch_size=200, mini_batch_size=64, seed=seed)
        ippo_trainer = IPPOTrainer(env=pz_env, config=ippo_cfg)
        ippo_trainer.train(total_timesteps=timesteps)
        ippo_eval = ippo_trainer.evaluate(num_episodes=3)
        pz_env.close()

        reward_diff = abs(ppo_eval["eval_mean_reward"] - ippo_eval["eval_mean_reward"])
        is_equivalent = reward_diff < 50.0  # Threshold check

        return {
            "is_equivalent": is_equivalent,
            "ppo_gym_eval_reward": ppo_eval["eval_mean_reward"],
            "ippo_1robot_eval_reward": ippo_eval["eval_mean_reward"],
            "reward_difference": reward_diff,
            "status": "PASSED" if is_equivalent else "DIVERGED",
        }


class ScalabilityVerifier:
    """Evaluates fleet scalability across 1, 2, 4, 8, 16 robots."""

    @staticmethod
    def verify_scalability(agent_counts: list = None, timesteps: int = 1000, seed: int = 42) -> Dict[str, Any]:
        """Runs incremental scaling evaluation across specified robot counts."""
        counts = agent_counts or [1, 2, 4, 8]
        results = {}

        for n_agents in counts:
            pz_cfg = MultiAgentEnvConfig(num_robots=n_agents, grid_width=10, grid_height=10, seed=seed)
            pz_env = WarehouseParallelEnv(config=pz_cfg)
            ippo_cfg = IPPOConfig(num_agents=n_agents, batch_size=200, mini_batch_size=64, seed=seed)
            trainer = IPPOTrainer(env=pz_env, config=ippo_cfg)
            trainer.train(total_timesteps=timesteps)
            eval_res = trainer.evaluate(num_episodes=2)
            pz_env.close()
            results[f"{n_agents}_robots"] = eval_res

        return results
