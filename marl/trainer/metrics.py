"""TrainingMetricsTracker accumulating episode returns, moving averages, and warehouse metrics."""

from collections import deque
from typing import Dict


class TrainingMetricsTracker:
    """Accumulates training performance metrics and computes moving averages."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size: int = window_size

        self.episode_count: int = 0
        self.total_steps: int = 0

        self.rewards: deque[float] = deque(maxlen=window_size)
        self.lengths: deque[int] = deque(maxlen=window_size)
        self.successes: deque[float] = deque(maxlen=window_size)
        self.collisions: deque[int] = deque(maxlen=window_size)
        self.completed_tasks: deque[int] = deque(maxlen=window_size)
        self.battery_usages: deque[float] = deque(maxlen=window_size)

    def record_episode(
        self,
        reward: float,
        length: int,
        success: bool,
        collisions: int = 0,
        completed_tasks: int = 0,
        battery_used: float = 0.0,
    ) -> None:
        """Records metrics for a completed episode."""
        self.episode_count += 1
        self.total_steps += length

        self.rewards.append(float(reward))
        self.lengths.append(int(length))
        self.successes.append(1.0 if success else 0.0)
        self.collisions.append(int(collisions))
        self.completed_tasks.append(int(completed_tasks))
        self.battery_usages.append(float(battery_used))

    def get_summary(self) -> Dict[str, float]:
        """Returns dictionary of aggregated metrics and moving averages."""
        if not self.rewards:
            return {
                "episode_count": 0.0,
                "mean_reward": 0.0,
                "mean_episode_length": 0.0,
                "success_rate": 0.0,
                "mean_collisions": 0.0,
                "mean_completed_tasks": 0.0,
                "mean_battery_used": 0.0,
            }

        n = len(self.rewards)
        return {
            "episode_count": float(self.episode_count),
            "total_steps": float(self.total_steps),
            "mean_reward": float(sum(self.rewards) / n),
            "mean_episode_length": float(sum(self.lengths) / n),
            "success_rate": float(sum(self.successes) / n),
            "mean_collisions": float(sum(self.collisions) / n),
            "mean_completed_tasks": float(sum(self.completed_tasks) / n),
            "mean_battery_used": float(sum(self.battery_usages) / n),
        }

    def reset(self) -> None:
        """Resets all metrics history."""
        self.episode_count = 0
        self.total_steps = 0
        self.rewards.clear()
        self.lengths.clear()
        self.successes.clear()
        self.collisions.clear()
        self.completed_tasks.clear()
        self.battery_usages.clear()
