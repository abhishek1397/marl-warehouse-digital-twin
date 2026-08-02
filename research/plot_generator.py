"""PublicationPlotGenerator producing IEEE/Springer publication-quality graphics."""

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class PublicationPlotGenerator:
    """Generates publication-quality plots: learning curves with 95% CI, box/violin plots, CDFs, and heatmaps."""

    @staticmethod
    def plot_learning_curves_with_ci(
        steps: List[int],
        seed_runs: List[List[float]],
        output_path: str,
        title: str = "Multi-Seed PPO Learning Curves (95% CI Shaded)",
    ) -> None:
        """Plots mean learning curve across seeds with 95% confidence interval shaded error band."""
        runs_matrix = np.asarray(seed_runs, dtype=float)
        mean_curve = np.mean(runs_matrix, axis=0)
        std_curve = np.std(runs_matrix, axis=0, ddof=1) if len(seed_runs) > 1 else np.zeros_like(mean_curve)
        sem_curve = std_curve / np.sqrt(len(seed_runs))
        ci95 = sem_curve * 1.96

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, mean_curve, color="#2ecc71", linewidth=2.5, label="PPO + PBRS + DAM Mean")
        ax.fill_between(steps, mean_curve - ci95, mean_curve + ci95, color="#2ecc71", alpha=0.25, label="95% Confidence Interval")

        ax.set_xlabel("Environment Timesteps")
        ax.set_ylabel("Evaluation Mean Reward")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_reward_box_violin(
        arm_rewards: Dict[str, List[float]], output_path: str
    ) -> None:
        """Renders comparative box plots and violin plots of reward distributions."""
        names = list(arm_rewards.keys())
        data = [arm_rewards[n] for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Box Plot
        ax1.boxplot(data, tick_labels=names, patch_artist=True,
                    boxprops=dict(facecolor="#3498db", color="#2980b9"),
                    medianprops=dict(color="red", linewidth=2))
        ax1.set_title("Reward Distribution Box Plot")
        ax1.set_ylabel("Mean Reward")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # Violin Plot
        ax2.violinplot(data, showmeans=True, showmedians=True)
        ax2.set_xticks(range(1, len(names) + 1))
        ax2.set_xticklabels(names)
        ax2.set_title("Reward Density Violin Plot")
        ax2.set_ylabel("Mean Reward")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_cdf_distribution(
        arm_rewards: Dict[str, List[float]], output_path: str
    ) -> None:
        """Renders Empirical Cumulative Distribution Function (CDF) curves."""
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6"]

        for idx, (name, rewards) in enumerate(arm_rewards.items()):
            sorted_data = np.sort(rewards)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            color = colors[idx % len(colors)]
            ax.plot(sorted_data, cdf, label=name, color=color, linewidth=2)

        ax.set_xlabel("Evaluation Reward")
        ax.set_ylabel("Empirical Cumulative Probability P(X <= x)")
        ax.set_title("Reward Distribution Empirical CDF")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_generalization_heatmap(
        gen_results: Dict[str, Dict[str, float]], output_path: str
    ) -> None:
        """Renders 2D generalization performance matrix heatmap across grid dimensions."""
        sizes = list(gen_results.keys())
        rewards = [gen_results[s].get("mean_completion_time", 0.0) for s in sizes]
        succ_rates = [gen_results[s].get("success_rate", 0.0) * 100.0 for s in sizes]

        matrix = np.array([succ_rates])

        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=100)

        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes)
        ax.set_yticks([0])
        ax.set_yticklabels(["Success Rate (%)"])
        ax.set_title("Zero-Shot Layout Generalization Performance Matrix")

        for j in range(len(sizes)):
            ax.text(j, 0, f"{succ_rates[j]:.1f}%", ha="center", va="center", color="black", fontweight="bold")

        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.3, label="Task Success Rate (%)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
