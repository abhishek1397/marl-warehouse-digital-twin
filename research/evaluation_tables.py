"""EvaluationTableGenerator emitting clean text and Markdown summary tables."""

from typing import Any, Dict


class EvaluationTableGenerator:
    """Generates formatted summary tables for Markdown and terminal output."""

    @staticmethod
    def generate_statistical_summary_table(stats: Dict[str, float]) -> str:
        """Formats descriptive statistical summary table in Markdown."""
        lines = [
            "| Statistic Metric | Value |",
            "| :--- | :---: |",
            fr"| **Mean ($\mu$)** | `{stats.get('mean', 0.0):.4f}` |",
            f"| **Median ($M$)** | `{stats.get('median', 0.0):.4f}` |",
            fr"| **Variance ($\sigma^2$)** | `{stats.get('var', 0.0):.4f}` |",
            fr"| **Std Deviation ($\sigma$)** | `{stats.get('std', 0.0):.4f}` |",
            f"| **95% CI Lower** | `{stats.get('ci_lower', 0.0):.4f}` |",
            f"| **95% CI Upper** | `{stats.get('ci_upper', 0.0):.4f}` |",
            f"| **Minimum** | `{stats.get('min', 0.0):.4f}` |",
            f"| **Maximum** | `{stats.get('max', 0.0):.4f}` |",
            f"| **Interquartile Range (IQR)** | `{stats.get('iqr', 0.0):.4f}` |",
            f"| **Coefficient of Variation ($CV$)** | `{stats.get('cv', 0.0):.4f}` |",
        ]
        return "\n".join(lines)

    @staticmethod
    def generate_ablation_summary_table(arm_stats: Dict[str, Dict[str, float]]) -> str:
        """Formats 4-arm ablation summary table in Markdown."""
        lines = [
            "| Experimental Arm | Mean Reward | Success Rate (%) | Collisions | Completion Time |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for arm_name, metrics in arm_stats.items():
            lines.append(
                f"| **{arm_name}** | `{metrics.get('mean_reward', 0.0):.2f}` | "
                f"`{metrics.get('success_rate', 0.0)*100:.1f}%` | `{metrics.get('mean_collisions', 0.0):.2f}` | "
                f"`{metrics.get('mean_completion_time', 0.0):.1f}` |"
            )
        return "\n".join(lines)
