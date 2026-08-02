"""MultiSeedExperimentRunner orchestrating 10-seed experiments and tracking per-seed artifacts."""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl import EnvConfig, WarehouseGymEnv
from marl.algorithms.ppo import PPOConfig, PPOTrainer
from research.policy_evaluator import PolicyEvaluator
from research.success_analyzer import SuccessMetricsAnalyzer


class MultiSeedExperimentRunner:
    """Runs 10 independent random seed experiments and maintains structured experiment tracking directories."""

    SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    @staticmethod
    def run_multi_seed_experiments(
        seeds: Optional[List[int]] = None,
        total_timesteps: int = 3000,
        base_dir: str = "experiments",
    ) -> Dict[str, Any]:
        """Executes full training and evaluation runs for each random seed."""
        seed_list = seeds or MultiSeedExperimentRunner.SEEDS
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", base_dir))
        os.makedirs(base_path, exist_ok=True)

        seed_results: Dict[int, Dict[str, float]] = {}

        print("=" * 75)
        print(f"   MULTI-SEED STATISTICAL EVALUATION ({len(seed_list)} SEEDS)")
        print("=" * 75)

        for s in seed_list:
            seed_dir = os.path.join(base_path, f"seed_{s}")
            plots_dir = os.path.join(seed_dir, "plots")
            ckpt_dir = os.path.join(seed_dir, "checkpoint")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)

            # 1. Environment and PPO config
            env_cfg = EnvConfig(grid_width=8, grid_height=8, seed=s, enable_reward_shaping=True, enable_action_masking=True)
            ppo_cfg = PPOConfig(learning_rate=3e-4, epochs=3, batch_size=300, mini_batch_size=64, seed=s)

            # Write config.yaml
            cfg_dict = {"env_config": env_cfg.__dict__, "ppo_config": ppo_cfg.__dict__}
            with open(os.path.join(seed_dir, "config.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(cfg_dict, f)

            # 2. Train PPO Agent
            print(f"\n[Seed {s}] Training PPO + PBRS + DAM...")
            env = WarehouseGymEnv(config=env_cfg)
            trainer = PPOTrainer(env=env, config=ppo_cfg)
            trainer.train(total_timesteps=total_timesteps)

            # Save checkpoint
            trainer.ckpt_handler.save_checkpoint(
                policy=trainer.policy,
                optimizer=trainer.optimizer,
                step=total_timesteps,
            )

            # 3. Evaluate Policy
            evaluator = PolicyEvaluator(env=env, policy=trainer.policy)
            trajs = evaluator.evaluate_policy(num_episodes=5, seed=s)
            metrics = SuccessMetricsAnalyzer.compute_success_metrics(trajs)
            mean_rew = float(np.mean([t.total_reward for t in trajs])) if trajs else 0.0
            metrics["mean_reward"] = mean_rew
            seed_results[s] = metrics
            env.close()

            # Save evaluation.json
            with open(os.path.join(seed_dir, "evaluation.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            # Save metrics.csv
            with open(os.path.join(seed_dir, "metrics.csv"), "w", encoding="utf-8") as f:
                f.write("metric,value\n")
                for k, v in metrics.items():
                    f.write(f"{k},{v}\n")

            # Save README.md
            with open(os.path.join(seed_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(f"# Experiment Tracking: Seed {s}\n\nMean Reward: {mean_rew:.2f}\nSuccess Rate: {metrics['success_rate']*100:.1f}%\n")

            print(f"         Seed {s} -> Mean Reward: {mean_rew:.2f} | Success Rate: {metrics['success_rate']*100:.1f}%")

        return seed_results


if __name__ == "__main__":
    MultiSeedExperimentRunner.run_multi_seed_experiments()
