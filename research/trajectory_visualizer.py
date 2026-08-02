"""TrajectoryVisualizer rendering 2D spatial path overlays and goal transitions."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from research.trajectory_recorder import EpisodeTrajectory
from simulator.position import Position


class TrajectoryVisualizer:
    """Renders 2D grid path overlays showing robot movement trajectories and warehouse entities."""

    @staticmethod
    def plot_trajectory_path(
        trajectory: EpisodeTrajectory,
        grid_width: int,
        grid_height: int,
        output_path: str,
        obstacles: Optional[List[Position]] = None,
        charging_stations: Optional[List[Position]] = None,
    ) -> None:
        """Plots 2D grid trajectory showing start position, path, pickup, drop, and obstacle locations."""
        fig, ax = plt.subplots(figsize=(7, 7))

        # 1. Grid matrix background
        grid_matrix = np.zeros((grid_height, grid_width))
        if obstacles:
            for obs in obstacles:
                if 0 <= obs.x < grid_width and 0 <= obs.y < grid_height:
                    grid_matrix[obs.y, obs.x] = 1.0

        ax.imshow(grid_matrix, cmap="binary", origin="upper", alpha=0.3)
        ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

        # 2. Plot Charging Stations
        if charging_stations:
            cs_x = [cs.x for cs in charging_stations]
            cs_y = [cs.y for cs in charging_stations]
            ax.scatter(cs_x, cs_y, c="gold", marker="p", s=150, label="Charging Station", zorder=3)

        # 3. Extract trajectory path coordinates
        if trajectory.steps:
            xs = [s.position.x for s in trajectory.steps]
            ys = [s.position.y for s in trajectory.steps]

            # Path line
            ax.plot(xs, ys, color="#3498db", linestyle="-", linewidth=2.5, marker="o", markersize=4, label="Robot Path", zorder=4)

            # Start point
            ax.scatter([xs[0]], [ys[0]], c="green", marker="s", s=120, label="Start Position", zorder=5)

            # Pickups
            pickup_steps = [s for s in trajectory.steps if s.is_pickup]
            if pickup_steps:
                ax.scatter([s.position.x for s in pickup_steps], [s.position.y for s in pickup_steps],
                           c="orange", marker="^", s=150, label="Pickup Event", zorder=6)

            # Deliveries
            delivery_steps = [s for s in trajectory.steps if s.is_delivery]
            if delivery_steps:
                ax.scatter([s.position.x for s in delivery_steps], [s.position.y for s in delivery_steps],
                           c="purple", marker="*", s=200, label="Delivery Event", zorder=6)

        ax.set_xlim(-0.5, grid_width - 0.5)
        ax.set_ylim(grid_height - 0.5, -0.5)  # Inverted Y for grid orientation
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"Episode {trajectory.episode_id} Robot Trajectory Path")
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
