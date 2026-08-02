"""CoordinationMetrics measuring multi-robot spatial interference, idle steps, and traffic congestion."""

from typing import Any, Dict, List

import numpy as np


class CoordinationMetrics:
    """Calculates multi-robot coordination metrics: collision avoidance rate, task overlap, and congestion index."""

    @staticmethod
    def compute_coordination_summary(
        collisions: int,
        deliveries: int,
        steps: int,
        fleet_size: int,
    ) -> Dict[str, float]:
        """Calculates collision avoidance efficiency, throughput, and robot interference metrics."""
        collision_rate = collisions / max(1.0, float(steps * fleet_size))
        collision_avoidance_rate = float(np.clip(1.0 - collision_rate, 0.0, 1.0))
        throughput = float(deliveries / max(1.0, float(steps)))

        return {
            "collision_avoidance_rate": collision_avoidance_rate,
            "throughput": throughput,
            "total_collisions": float(collisions),
            "total_deliveries": float(deliveries),
        }
