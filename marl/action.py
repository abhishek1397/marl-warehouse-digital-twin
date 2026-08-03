"""ActionMapper class translating discrete Gymnasium actions into simulator state mutations."""

from dataclasses import dataclass
from typing import Dict, Optional

from marl.config import EnvConfig
from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.constants import Direction
from simulator.position import Position
from simulator.robot import Robot, RobotState
from simulator.shelf import Shelf
from simulator.task import Task, TaskStatus
from simulator.warehouse import Warehouse


@dataclass
class ActionResult:
    """Dataclass encapsulating execution diagnostics for an applied action."""

    action: int
    is_valid: bool = True
    is_collision: bool = False
    picked_package: bool = False
    dropped_package: bool = False
    docked_charging: bool = False
    message: str = ""


class ActionMapper:
    """Translates 8 discrete Gymnasium actions into warehouse simulator commands."""

    ACTION_MAP = {
        0: Direction.NORTH,
        1: Direction.SOUTH,
        2: Direction.WEST,
        3: Direction.EAST,
        4: Direction.STAY,
    }

    def execute_action(
        self,
        action: int,
        robot: Robot,
        warehouse: Warehouse,
        task: Optional[Task] = None,
        charging_stations: Optional[Dict[str, ChargingStation]] = None,
    ) -> ActionResult:
        """Applies a discrete action index to the robot and environment state.

        Args:
            action: Integer action index (0 to 7).
            robot: Target Robot instance.
            warehouse: Warehouse digital twin environment.
            task: Assigned Task instance, if any.
            charging_stations: Map of ChargingStation instances.

        Returns:
            ActionResult object containing execution outcome flags.
        """
        if not (0 <= action <= 7):
            return ActionResult(
                action=action,
                is_valid=False,
                is_collision=False,
                message=f"Action index {action} out of bounds (0-7).",
            )

        # 1. Movement & Wait Actions (0 - 4)
        if action in self.ACTION_MAP:
            direction = self.ACTION_MAP[action]
            if direction == Direction.STAY:
                robot.increment_idle_time()
                return ActionResult(action=action, is_valid=True, message="Robot waited.")

            target_pos = robot.position.get_neighbor(direction)

            if not warehouse.is_in_bounds(target_pos):
                return ActionResult(
                    action=action,
                    is_valid=False,
                    is_collision=True,
                    message=f"Target position {target_pos} is out of bounds.",
                )

            cell = warehouse.get_cell(target_pos)
            if not cell.cell_type.is_traversable:
                return ActionResult(
                    action=action,
                    is_valid=False,
                    is_collision=True,
                    message=f"Target cell at {target_pos} is non-traversable ({cell.cell_type.name}).",
                )

            # Move robot to new position
            robot.position = target_pos
            robot._total_distance_travelled += 1
            return ActionResult(action=action, is_valid=True, message=f"Moved to {target_pos}.")

        # 2. Pick Package Action (5)
        elif action == 5:
            if robot.carrying_package is not None:
                return ActionResult(
                    action=action,
                    is_valid=False,
                    message="Already carrying a package.",
                )

            if task is not None and task.package is not None:
                if robot.position == task.pickup_position or robot.position.manhattan_distance(task.pickup_position) <= 1:
                    robot.pick_up_package(task.package)
                    task.status = TaskStatus.IN_PROGRESS
                    return ActionResult(
                        action=action,
                        is_valid=True,
                        picked_package=True,
                        message=f"Picked up package '{task.package.package_id}'.",
                    )

            return ActionResult(
                action=action,
                is_valid=False,
                message="No valid package to pick up at current position.",
            )

        # 3. Drop Package Action (6)
        elif action == 6:
            if robot.carrying_package is None:
                return ActionResult(
                    action=action,
                    is_valid=False,
                    message="Not carrying any package to drop.",
                )

            if task is not None:
                if robot.position == task.drop_position or robot.position.manhattan_distance(task.drop_position) <= 1:
                    robot.drop_package()
                    task.status = TaskStatus.COMPLETED
                    robot.increment_tasks_completed()
                    return ActionResult(
                        action=action,
                        is_valid=True,
                        dropped_package=True,
                        message=f"Delivered package '{task.package.package_id}' to destination.",
                    )

            return ActionResult(
                action=action,
                is_valid=False,
                message="Current position is not the drop destination for carried package.",
            )

        # 4. Go Charge Action (7)
        elif action == 7:
            if robot.carrying_package is not None:
                return ActionResult(
                    action=action,
                    is_valid=False,
                    message="Cannot charge while carrying a package.",
                )

            if charging_stations:
                for station in charging_stations.values():
                    if station.position == robot.position:
                        if station.is_docked(robot.robot_id):
                            robot.state = RobotState.CHARGING
                            return ActionResult(
                                action=action,
                                is_valid=True,
                                docked_charging=True,
                                message=f"Already docked at charging station '{station.station_id}'.",
                            )
                        elif station.is_available():
                            station.dock_robot(robot.robot_id)
                            robot.state = RobotState.CHARGING
                            return ActionResult(
                                action=action,
                                is_valid=True,
                                docked_charging=True,
                                message=f"Docked at charging station '{station.station_id}'.",
                            )

            return ActionResult(
                action=action,
                is_valid=False,
                message="No available charging station at current position.",
            )

        return ActionResult(action=action, is_valid=False, message="Unhandled action.")
