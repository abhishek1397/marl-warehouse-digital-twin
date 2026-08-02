"""PotentialFunction implementations for potential-based reward shaping."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from marl.reward_shaping.distance_metrics import manhattan_distance
from simulator.charging_station import ChargingStation
from simulator.position import Position
from simulator.robot import Robot
from simulator.task import Task


class PotentialFunction(ABC):
    """Abstract Base Class for Potential Functions Phi(s)."""

    @abstractmethod
    def compute_potential(
        self,
        robot: Robot,
        task: Optional[Task] = None,
        charging_stations: Optional[Dict[str, ChargingStation]] = None,
    ) -> float:
        """Calculates state potential scalar float Phi(s)."""
        pass


class ManhattanPotential(PotentialFunction):
    """Potential function based on negative Manhattan distance to active goal target."""

    def __init__(self, low_battery_threshold: float = 20.0) -> None:
        self.low_battery_threshold: float = low_battery_threshold

    def get_active_goal(
        self,
        robot: Robot,
        task: Optional[Task] = None,
        charging_stations: Optional[Dict[str, ChargingStation]] = None,
    ) -> Position:
        """Determines active goal position based on robot battery, carrying state, and task."""
        # 1. Low battery emergency override -> Goal = nearest charging station
        if robot.battery_level < self.low_battery_threshold and charging_stations:
            nearest_station = min(
                charging_stations.values(),
                key=lambda cs: manhattan_distance(robot.position, cs.position),
            )
            return nearest_station.position

        # 2. Carrying package -> Goal = Task drop location
        if robot.carrying_package is not None and task is not None:
            return task.drop_position

        # 3. Unladen with active task -> Goal = Task pickup location
        if task is not None:
            return task.pickup_position

        # Default self-position goal
        return robot.position

    def compute_potential(
        self,
        robot: Robot,
        task: Optional[Task] = None,
        charging_stations: Optional[Dict[str, ChargingStation]] = None,
    ) -> float:
        """Computes Phi(s) = -ManhattanDistance(robot.position, active_goal)."""
        goal = self.get_active_goal(robot, task=task, charging_stations=charging_stations)
        dist = manhattan_distance(robot.position, goal)
        return float(-dist)
