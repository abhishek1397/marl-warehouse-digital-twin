"""SuccessMetricsAnalyzer evaluating performance metrics for successful and failed episodes."""

from typing import Dict, List

import numpy as np

from research.trajectory_recorder import EpisodeTrajectory


class SuccessMetricsAnalyzer:
    """Computes comprehensive task completion, efficiency, and battery metrics."""

    @staticmethod
    def compute_success_metrics(trajectories: List[EpisodeTrajectory]) -> Dict[str, float]:
        """Calculates success rates, delivery stats, distance travelled, and battery consumption."""
        if not trajectories:
            return {
                "success_rate": 0.0,
                "pickup_rate": 0.0,
                "delivery_rate": 0.0,
                "mean_completion_time": 0.0,
                "mean_distance_travelled": 0.0,
                "mean_battery_consumption": 0.0,
                "mean_collisions": 0.0,
                "mean_idle_pct": 0.0,
            }

        n = len(trajectories)
        success_count = sum(1 for t in trajectories if t.is_success)
        pickup_count = sum(1 for t in trajectories if t.total_pickups > 0)
        delivery_count = sum(1 for t in trajectories if t.total_deliveries > 0)

        # Completion times for successful episodes
        success_lens = [t.episode_length for t in trajectories if t.is_success]
        mean_comp_time = float(np.mean(success_lens)) if success_lens else 0.0

        # Distance travelled (movement actions 0-3)
        distances = []
        battery_uses = []
        collisions = []
        idle_pcts = []

        for t in trajectories:
            m_dist = sum(1 for s in t.steps if s.action in (0, 1, 2, 3))
            distances.append(m_dist)
            collisions.append(t.total_collisions)

            if t.steps:
                b_used = float(t.steps[0].battery_level - t.steps[-1].battery_level)
                battery_uses.append(max(0.0, b_used))
                w_count = sum(1 for s in t.steps if s.action == 4)
                idle_pcts.append((w_count / len(t.steps)) * 100.0)

        return {
            "success_rate": float(success_count / n),
            "pickup_rate": float(pickup_count / n),
            "delivery_rate": float(delivery_count / n),
            "mean_completion_time": mean_comp_time,
            "mean_distance_travelled": float(np.mean(distances)) if distances else 0.0,
            "mean_battery_consumption": float(np.mean(battery_uses)) if battery_uses else 0.0,
            "mean_collisions": float(np.mean(collisions)) if collisions else 0.0,
            "mean_idle_pct": float(np.mean(idle_pcts)) if idle_pcts else 0.0,
        }
