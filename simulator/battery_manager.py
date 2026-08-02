"""BatteryManager class handling battery depletion, charging dynamics, and power warnings."""

from typing import Dict

from simulator.robot import Robot, RobotState


class BatteryManager:
    """Manages battery drain, replenishment, low battery, and critical battery warnings across the robot fleet."""

    def __init__(
        self,
        idle_drain_rate: float = 0.05,
        move_drain_rate: float = 0.2,
        loaded_drain_rate: float = 0.4,
        charging_rate: float = 10.0,
        low_battery_threshold: float = 20.0,
        critical_battery_threshold: float = 5.0,
    ) -> None:
        self.idle_drain_rate: float = idle_drain_rate
        self.move_drain_rate: float = move_drain_rate
        self.loaded_drain_rate: float = loaded_drain_rate
        self.charging_rate: float = charging_rate
        self.low_battery_threshold: float = low_battery_threshold
        self.critical_battery_threshold: float = critical_battery_threshold

    def is_low_battery(self, robot: Robot) -> bool:
        """Returns True if robot battery percentage is at or below low battery threshold."""
        return robot.battery_percentage <= self.low_battery_threshold

    def is_critical_battery(self, robot: Robot) -> bool:
        """Returns True if robot battery percentage is at or below critical battery threshold."""
        return robot.battery_percentage <= self.critical_battery_threshold

    def update_robot_battery(self, robot: Robot) -> None:
        """Updates robot battery level for a single simulation step based on current state."""
        if robot.state == RobotState.CHARGING:
            robot.charge_battery(self.charging_rate)
            robot.increment_charging_time()
        elif robot.carrying_package is not None:
            robot.consume_battery(self.loaded_drain_rate)
        elif robot.state in {
            RobotState.MOVING_TO_PICKUP,
            RobotState.MOVING_TO_DROP,
            RobotState.MOVING_TO_CHARGE,
        }:
            robot.consume_battery(self.move_drain_rate)
        elif robot.is_idle():
            robot.consume_battery(self.idle_drain_rate)
            robot.increment_idle_time()
        else:
            robot.consume_battery(self.idle_drain_rate)

    def update_fleet(self, fleet: Dict[str, Robot]) -> None:
        """Updates battery consumption for all robots in the fleet."""
        for robot in fleet.values():
            self.update_robot_battery(robot)
