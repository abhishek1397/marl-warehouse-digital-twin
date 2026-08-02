"""TrafficController class managing multi-robot coordination, congestion analysis, and deadlock prevention."""

from typing import Dict, List, Set, Tuple

from simulator.position import Position
from simulator.robot import Robot, RobotState
from simulator.warehouse import Warehouse


class TrafficController:
    """Coordinates fleet movement, detects spatial congestion, and resolves potential grid deadlocks."""

    def __init__(self) -> None:
        self.collisions_prevented: int = 0
        self.deadlocks_prevented: int = 0
        self.replans_count: int = 0
        self.total_waiting_time: int = 0

        # Tracks last positions for deadlock detection: robot_id -> (Position, unchanged_steps)
        self._stalled_history: Dict[str, Tuple[Position, int]] = {}

    def detect_congestion(
        self,
        fleet: Dict[str, Robot],
        radius: int = 2,
        density_threshold: int = 3,
    ) -> List[Position]:
        """Identifies grid hot-spot positions where robot density exceeds the threshold.

        Args:
            fleet: Active robot fleet.
            radius: Neighborhood distance to measure local density.
            density_threshold: Minimum neighboring robots to classify as congested.

        Returns:
            List of congested central Positions.
        """
        congested_spots: List[Position] = []
        robot_positions = [r.position for r in fleet.values()]

        for pos in set(robot_positions):
            nearby_count = sum(
                1 for other_pos in robot_positions if pos.manhattan_distance(other_pos) <= radius
            )
            if nearby_count >= density_threshold:
                congested_spots.append(pos)

        return congested_spots

    def detect_deadlocks(
        self,
        fleet: Dict[str, Robot],
        stall_threshold: int = 4,
    ) -> List[str]:
        """Identifies robots that have been stalled in non-idle movement states without progress.

        Args:
            fleet: Active robot fleet.
            stall_threshold: Number of consecutive steps at same position before declaring deadlock risk.

        Returns:
            List of robot IDs flagged for deadlock resolution.
        """
        deadlocked_robots: List[str] = []

        for r_id, robot in fleet.items():
            if robot.is_idle() or robot.is_charging():
                self._stalled_history[r_id] = (robot.position, 0)
                continue

            last_pos, stall_count = self._stalled_history.get(r_id, (robot.position, 0))

            if robot.position == last_pos:
                new_stall = stall_count + 1
                self._stalled_history[r_id] = (robot.position, new_stall)
                if new_stall >= stall_threshold:
                    deadlocked_robots.append(r_id)
            else:
                self._stalled_history[r_id] = (robot.position, 0)

        return deadlocked_robots

    def record_collision_prevented(self, count: int = 1) -> None:
        """Increments collision prevention counter."""
        self.collisions_prevented += count

    def record_deadlock_prevented(self, count: int = 1) -> None:
        """Increments deadlock prevention counter."""
        self.deadlocks_prevented += count

    def record_replan(self, count: int = 1) -> None:
        """Increments replan counter."""
        self.replans_count += count

    def record_waiting_time(self, steps: int = 1) -> None:
        """Increments robot waiting steps counter."""
        self.total_waiting_time += steps

    def reset_stats(self) -> None:
        """Resets all traffic statistics."""
        self.collisions_prevented = 0
        self.deadlocks_prevented = 0
        self.replans_count = 0
        self.total_waiting_time = 0
        self._stalled_history.clear()
