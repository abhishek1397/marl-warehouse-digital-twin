"""EpisodeAnimationExporter exporting 2D rendered frames and animated GIF playback."""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from research.trajectory_recorder import EpisodeTrajectory
from simulator.position import Position


class EpisodeAnimationExporter:
    """Exports rendered 2D trajectory animation frames and GIFs with HUD telemetry overlays."""

    ACTION_NAMES = ["Move Up", "Move Down", "Move Left", "Move Right", "Wait", "Pick", "Drop", "Charge"]

    @staticmethod
    def render_frame(
        trajectory: EpisodeTrajectory,
        step_idx: int,
        grid_width: int,
        grid_height: int,
        output_path: str,
    ) -> None:
        """Renders single 2D grid frame with HUD telemetry overlay."""
        step = trajectory.steps[step_idx]
        fig, ax = plt.subplots(figsize=(6, 6))

        grid_matrix = np.zeros((grid_height, grid_width))
        ax.imshow(grid_matrix, cmap="binary", origin="upper", alpha=0.2)
        ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

        # Plot historical path up to step_idx
        xs = [s.position.x for s in trajectory.steps[: step_idx + 1]]
        ys = [s.position.y for s in trajectory.steps[: step_idx + 1]]
        ax.plot(xs, ys, color="#3498db", linestyle="--", linewidth=1.5, marker="o", markersize=3)

        # Plot current robot position
        ax.scatter([step.position.x], [step.position.y], c="blue", marker="s", s=150, label="Robot", zorder=5)

        # Plot goal if available
        if step.goal_position:
            ax.scatter([step.goal_position.x], [step.goal_position.y], c="purple", marker="*", s=200, label="Target Goal", zorder=5)

        # HUD Overlay text
        act_name = EpisodeAnimationExporter.ACTION_NAMES[step.action] if 0 <= step.action < 8 else "Unknown"
        hud_text = (
            f"Step: {step.timestep} | Action: {act_name}\n"
            f"Reward: {step.reward:+.2f} | Battery: {step.battery_level:.1f}%\n"
            f"Carrying Pkg: {step.carrying_package} | Task: {step.task_status}"
        )
        ax.text(
            0.02, 0.98, hud_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        ax.set_xlim(-0.5, grid_width - 0.5)
        ax.set_ylim(grid_height - 0.5, -0.5)
        ax.set_title(f"Episode {trajectory.episode_id} Playback")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def export_episode_gif(
        trajectory: EpisodeTrajectory,
        grid_width: int,
        grid_height: int,
        output_gif_path: str,
        fps: int = 4,
    ) -> None:
        """Exports animated GIF playback of an evaluation trajectory episode."""
        if not trajectory.steps:
            return

        fig, ax = plt.subplots(figsize=(6, 6))

        def update(frame_idx: int) -> None:
            ax.clear()
            step = trajectory.steps[frame_idx]

            grid_matrix = np.zeros((grid_height, grid_width))
            ax.imshow(grid_matrix, cmap="binary", origin="upper", alpha=0.2)
            ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

            xs = [s.position.x for s in trajectory.steps[: frame_idx + 1]]
            ys = [s.position.y for s in trajectory.steps[: frame_idx + 1]]
            ax.plot(xs, ys, color="#3498db", linestyle="--", linewidth=1.5, marker="o", markersize=3)
            ax.scatter([step.position.x], [step.position.y], c="blue", marker="s", s=150, zorder=5)

            if step.goal_position:
                ax.scatter([step.goal_position.x], [step.goal_position.y], c="purple", marker="*", s=200, zorder=5)

            act_name = EpisodeAnimationExporter.ACTION_NAMES[step.action] if 0 <= step.action < 8 else "Unknown"
            hud = f"Step: {step.timestep} | Act: {act_name} | Rew: {step.reward:+.2f} | Battery: {step.battery_level:.0f}%"
            ax.text(0.02, 0.98, hud, transform=ax.transAxes, fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlim(-0.5, grid_width - 0.5)
            ax.set_ylim(grid_height - 0.5, -0.5)
            ax.set_title(f"Episode {trajectory.episode_id} Animated Playback")

        anim = FuncAnimation(fig, update, frames=len(trajectory.steps), interval=1000 // fps)
        anim.save(output_gif_path, writer=PillowWriter(fps=fps))
        plt.close()
