"""RewardDecompositionAnalyzer separating base environment rewards from potential-based shaping rewards."""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from research.trajectory_recorder import EpisodeTrajectory


class RewardDecompositionAnalyzer:
    """Analyzes and plots raw environment rewards vs. potential-based shaping rewards."""

    @staticmethod
    def compute_reward_breakdown(trajectories: List[EpisodeTrajectory]) -> Dict[str, float]:
        """Computes aggregate environment, potential, and total reward components."""
        if not trajectories:
            return {"env_reward": 0.0, "potential_reward": 0.0, "total_reward": 0.0}

        n = len(trajectories)
        total_env = sum(t.total_env_reward for t in trajectories)
        total_pot = sum(t.total_potential_reward for t in trajectories)
        total_all = sum(t.total_reward for t in trajectories)

        return {
            "mean_env_reward": float(total_env / n),
            "mean_potential_reward": float(total_pot / n),
            "mean_total_reward": float(total_all / n),
        }

    @staticmethod
    def plot_stacked_reward_curves(trajectory: EpisodeTrajectory, output_path: str) -> None:
        """Plots step-by-step cumulative environment reward and potential shaping reward."""
        steps = [s.timestep for s in trajectory.steps]
        env_rewards = [s.env_reward for s in trajectory.steps]
        pot_rewards = [s.potential_reward for s in trajectory.steps]

        cum_env = np.cumsum(env_rewards)
        cum_pot = np.cumsum(pot_rewards)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, cum_env, label="Environment Reward (Base)", color="#e74c3c", linewidth=2)
        ax.plot(steps, cum_pot, label="PBRS Potential Reward", color="#2ecc71", linewidth=2)
        ax.plot(steps, cum_env + cum_pot, label="Total Combined Reward", color="#3498db", linestyle="--", linewidth=2)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(f"Episode {trajectory.episode_id} Reward Component Decomposition")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
