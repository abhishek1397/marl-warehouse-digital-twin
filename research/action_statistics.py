"""ActionStatisticsAnalyzer computing action distributions, frequencies, and entropy metrics."""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from research.trajectory_recorder import EpisodeTrajectory


class ActionStatisticsAnalyzer:
    """Computes action usage statistics, frequencies, and entropy across evaluation trajectories."""

    ACTION_NAMES = ["Move Up", "Move Down", "Move Left", "Move Right", "Wait", "Pick", "Drop", "Charge"]

    @staticmethod
    def compute_action_frequencies(trajectories: List[EpisodeTrajectory]) -> Dict[str, float]:
        """Calculates action counts, percentages, and entropy across all recorded steps."""
        total_steps = sum(len(t.steps) for t in trajectories)
        if total_steps == 0:
            return {name: 0.0 for name in ActionStatisticsAnalyzer.ACTION_NAMES}

        counts = np.zeros(8, dtype=int)
        for t in trajectories:
            for step in t.steps:
                if 0 <= step.action < 8:
                    counts[step.action] += 1

        freqs = counts / total_steps
        stats = {ActionStatisticsAnalyzer.ACTION_NAMES[i]: float(freqs[i]) for i in range(8)}

        # Aggregated summary percentages
        stats["movement_pct"] = float(np.sum(freqs[:4]) * 100.0)
        stats["wait_pct"] = float(freqs[4] * 100.0)
        stats["pick_pct"] = float(freqs[5] * 100.0)
        stats["drop_pct"] = float(freqs[6] * 100.0)
        stats["charge_pct"] = float(freqs[7] * 100.0)

        # Action distribution entropy
        non_zero = freqs[freqs > 0]
        stats["action_entropy"] = float(-np.sum(non_zero * np.log(non_zero)))

        return stats

    @staticmethod
    def plot_action_distribution(stats: Dict[str, float], output_path: str) -> None:
        """Plots action distribution bar chart."""
        names = ActionStatisticsAnalyzer.ACTION_NAMES
        vals = [stats[name] * 100.0 for name in names]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(names, vals, color="#3498db")
        ax.set_ylabel("Execution Frequency (%)")
        ax.set_title("PPO Policy Action Distribution Analysis")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
