"""EpisodeManager class tracking episode progress, termination conditions, and info metrics."""

from typing import Any, Dict, Tuple

from marl.config import EnvConfig
from simulator.robot import Robot
from simulator.task_manager import TaskManager


class EpisodeManager:
    """Manages episode execution lifecycle, termination, truncation, and diagnostic info."""

    def __init__(self, config: EnvConfig) -> None:
        self.config: EnvConfig = config
        self.current_step: int = 0
        self.cumulative_reward: float = 0.0

    def start_episode(self) -> None:
        """Resets episode step counter and cumulative return."""
        self.current_step = 0
        self.cumulative_reward = 0.0

    def check_step(
        self, step_reward: float, robot: Robot, task_manager: TaskManager
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """Updates step count and evaluates episode termination/truncation flags.

        Returns:
            Tuple of (terminated, truncated, info_dict).
        """
        self.current_step += 1
        self.cumulative_reward += step_reward

        is_success = task_manager.all_tasks_completed() and task_manager.total_tasks > 0
        is_battery_empty = robot.battery_level <= 0.0
        is_timeout = self.current_step >= self.config.max_episode_steps

        terminated = is_success or is_battery_empty
        truncated = is_timeout and not terminated

        info = {
            "step": self.current_step,
            "cumulative_reward": round(self.cumulative_reward, 2),
            "completed_deliveries": len(task_manager.get_completed_tasks()),
            "total_tasks": task_manager.total_tasks,
            "robot_battery": round(float(robot.battery_level), 2),
            "is_success": is_success,
            "is_battery_empty": is_battery_empty,
            "is_timeout": is_timeout,
        }

        return terminated, truncated, info
