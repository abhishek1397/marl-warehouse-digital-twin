"""LaTeXTableGenerator producing publication-ready IEEE/Springer LaTeX table code blocks."""

from typing import Any, Dict


class LaTeXTableGenerator:
    """Generates publication-ready LaTeX tables formatted for IEEE transactions and Springer LNCS manuscripts."""

    @staticmethod
    def generate_performance_latex_table(arm_metrics: Dict[str, Dict[str, float]]) -> str:
        """Generates IEEE-style Performance Comparison Table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\caption{Ablation Performance Comparison Across Policy Variants}",
            r"\label{tab:ablation_performance}",
            r"\centering",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"\textbf{Experimental Arm} & \textbf{Mean Reward} & \textbf{Success Rate (\%)} & \textbf{Collisions} & \textbf{Completion Time} \\",
            r"\hline",
        ]

        for arm_name, metrics in arm_metrics.items():
            lines.append(
                f"  {arm_name:<25} & ${metrics.get('mean_reward', 0.0):8.2f}$ & "
                f"${metrics.get('success_rate', 0.0)*100:5.1f}\\%$ & "
                f"${metrics.get('mean_collisions', 0.0):5.2f}$ & "
                f"${metrics.get('mean_completion_time', 0.0):5.1f}$ \\\\"
            )

        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    @staticmethod
    def generate_statistical_summary_latex_table(stats: Dict[str, float]) -> str:
        """Generates IEEE-style Multi-Seed Statistical Summary Table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\caption{Statistical Metrics of Trained Policy Across 10 Random Seeds}",
            r"\label{tab:statistical_summary}",
            r"\centering",
            r"\begin{tabular}{lc}",
            r"\hline",
            r"\textbf{Metric Parameter} & \textbf{Statistical Value} \\",
            r"\hline",
            f"  Mean ($\\mu$) & ${stats.get('mean', 0.0):.4f}$ \\\\",
            f"  Median ($M$) & ${stats.get('median', 0.0):.4f}$ \\\\",
            f"  Variance ($\\sigma^2$) & ${stats.get('var', 0.0):.4f}$ \\\\",
            f"  Standard Deviation ($\\sigma$) & ${stats.get('std', 0.0):.4f}$ \\\\",
            f"  95\\% CI Lower & ${stats.get('ci_lower', 0.0):.4f}$ \\\\",
            f"  95\\% CI Upper & ${stats.get('ci_upper', 0.0):.4f}$ \\\\",
            f"  Minimum & ${stats.get('min', 0.0):.4f}$ \\\\",
            f"  Maximum & ${stats.get('max', 0.0):.4f}$ \\\\",
            f"  Interquartile Range (IQR) & ${stats.get('iqr', 0.0):.4f}$ \\\\",
            f"  Coefficient of Variation ($CV$) & ${stats.get('cv', 0.0):.4f}$ \\\\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    @staticmethod
    def generate_generalization_latex_table(gen_results: Dict[str, Dict[str, float]]) -> str:
        """Generates IEEE-style Zero-Shot Generalization Table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\caption{Zero-Shot Policy Generalization Matrix Across Unseen Layout Dimensions}",
            r"\label{tab:generalization_matrix}",
            r"\centering",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"\textbf{Grid Size} & \textbf{Success Rate (\%)} & \textbf{Pickup Rate (\%)} & \textbf{Delivery Rate (\%)} & \textbf{Mean Distance} \\",
            r"\hline",
        ]

        for size_tag, metrics in gen_results.items():
            lines.append(
                f"  {size_tag:<15} & ${metrics.get('success_rate', 0.0)*100:5.1f}\\%$ & "
                f"${metrics.get('pickup_rate', 0.0)*100:5.1f}\\%$ & "
                f"${metrics.get('delivery_rate', 0.0)*100:5.1f}\\%$ & "
                f"${metrics.get('mean_distance_travelled', 0.0):5.1f}$ \\\\"
            )

        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)
