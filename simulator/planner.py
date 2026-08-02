"""MultiRobotPlanner class handling space-time multi-agent path planning and conflict resolution."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from simulator.astar import AStarPlanner, PathResult
from simulator.grid import Grid
from simulator.position import Position
from simulator.robot import Robot
from simulator.reservation_table import ReservationTable


@dataclass
class PlanningRequest:
    """Dataclass holding a single robot's path planning goal and priority."""

    robot: Robot
    goal_position: Position
    priority: int = 0
    start_position: Optional[Position] = None


class MultiRobotPlanner:
    """Multi-Robot Path Planner implementing Prioritized Multi-Robot Path Finding (PRMAPF)."""

    def __init__(
        self,
        astar_planner: Optional[AStarPlanner] = None,
        reservation_table: Optional[ReservationTable] = None,
    ) -> None:
        self.astar_planner: AStarPlanner = astar_planner or AStarPlanner()
        self.reservation_table: ReservationTable = reservation_table or ReservationTable()

    def plan_joint_paths(
        self,
        grid: Grid,
        requests: List[PlanningRequest],
        start_timestep: int = 0,
        clear_reservations: bool = True,
    ) -> Dict[str, PathResult]:
        """Plans collision-free joint paths for a list of robot requests ordered by priority.

        Args:
            grid: Warehouse spatial grid.
            requests: List of PlanningRequest objects.
            start_timestep: Initial timestep.
            clear_reservations: If True, clears reservation table before planning.

        Returns:
            Dictionary mapping robot_id -> PathResult.
        """
        if clear_reservations:
            self.reservation_table.clear()

        # Sort requests by priority (descending: highest priority planned first)
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)

        results: Dict[str, PathResult] = {}

        for req in sorted_requests:
            robot_id = req.robot.robot_id
            start_pos = req.start_position if req.start_position is not None else req.robot.position
            goal_pos = req.goal_position

            path_result = self.astar_planner.plan(
                grid=grid,
                start=start_pos,
                goal=goal_pos,
                start_timestep=start_timestep,
                reservation_table=self.reservation_table,
                robot_id=robot_id,
            )

            if path_result.success:
                # Reserve optimal path in time-expanded reservation table
                reserved = self.reservation_table.reserve_path(
                    robot_id=robot_id,
                    path=path_result.path,
                    start_timestep=start_timestep,
                )
                if not reserved:
                    path_result.success = False
                    path_result.error_message = f"Failed to lock reservations for robot '{robot_id}'."

            results[robot_id] = path_result

        return results

    def replan_robot(
        self,
        grid: Grid,
        robot: Robot,
        goal_position: Position,
        start_timestep: int,
        old_path: Optional[List[Position]] = None,
    ) -> PathResult:
        """Replans a single robot's path around existing reservations."""
        if old_path:
            self.reservation_table.release_path(
                robot_id=robot.robot_id, path=old_path, start_timestep=start_timestep
            )

        new_result = self.astar_planner.plan(
            grid=grid,
            start=robot.position,
            goal=goal_position,
            start_timestep=start_timestep,
            reservation_table=self.reservation_table,
            robot_id=robot.robot_id,
        )

        if new_result.success:
            self.reservation_table.reserve_path(
                robot_id=robot.robot_id,
                path=new_result.path,
                start_timestep=start_timestep,
            )

        return new_result
