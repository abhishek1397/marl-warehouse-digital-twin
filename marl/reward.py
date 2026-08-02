"""RewardEngine class calculating step rewards based on Configurable EnvConfig parameters."""

from marl.action import ActionResult
from marl.config import EnvConfig
from simulator.robot import Robot, RobotState


class RewardEngine:
    """Computes scalar reinforcement rewards based on action outcomes and state transitions."""

    def __init__(self, config: EnvConfig) -> None:
        self.config: EnvConfig = config

    def calculate_reward(
        self, action_result: ActionResult, robot: Robot
    ) -> float:
        """Calculates step reward for a single agent transition.

        Args:
            action_result: Diagnostics output from ActionMapper execution.
            robot: Target Robot instance.

        Returns:
            Calculated scalar reward float.
        """
        # Base step time penalty
        reward: float = self.config.step_time_penalty

        if not action_result.is_valid:
            if action_result.is_collision:
                reward += self.config.collision_penalty
            else:
                reward += self.config.invalid_action_penalty
        else:
            # Valid action executed
            if action_result.action == 4:  # Wait action
                reward += self.config.waiting_penalty

            if action_result.picked_package:
                reward += self.config.package_pickup_reward

            if action_result.dropped_package:
                reward += self.config.successful_delivery_reward

            if action_result.docked_charging or robot.state == RobotState.CHARGING:
                reward += self.config.successful_charging_reward

        # Battery depleted failure penalty
        if robot.battery_level <= 0.0:
            reward += self.config.battery_empty_penalty

        return reward
