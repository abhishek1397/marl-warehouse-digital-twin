"""ObservationEncoder extracting structured NumPy observation arrays from simulator state."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from marl.config import EnvConfig
from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.package import Package, PackageStatus
from simulator.position import Position
from simulator.robot import Robot
from simulator.task import Task, TaskStatus
from simulator.warehouse import Warehouse


class ObservationEncoder:
    """Encodes simulator state into structured NumPy observation dictionaries for Gymnasium."""

    def __init__(self, config: EnvConfig) -> None:
        self.config: EnvConfig = config

    def encode(
        self,
        robot: Robot,
        warehouse: Warehouse,
        fleet: Dict[str, Robot],
        task: Optional[Task] = None,
        charging_stations: Optional[Dict[str, ChargingStation]] = None,
    ) -> Dict[str, np.ndarray]:
        """Encodes state for a given robot into Gymnasium Dict observation.

        Returns:
            Dict containing robot_position, goal_position, battery_level, package_status,
            local_occupancy, charging_station_distance, task_status.
        """
        # 1. Robot position
        robot_pos_arr = np.array([robot.position.x, robot.position.y], dtype=np.int32)

        # 2. Goal position
        goal_pos = Position(0, 0)
        if task is not None:
            if robot.carrying_package is not None:
                goal_pos = task.drop_position
            else:
                goal_pos = task.pickup_position
        goal_pos_arr = np.array([goal_pos.x, goal_pos.y], dtype=np.int32)

        # 3. Battery level
        battery_arr = np.array([float(robot.battery_level)], dtype=np.float32)

        # 4. Package status (0: None, 1: Pickup, 2: Carrying, 3: Delivered)
        pkg_val = 0
        if robot.carrying_package is not None:
            pkg_val = 2
        elif task is not None and task.package is not None:
            if task.package.status == PackageStatus.DELIVERED:
                pkg_val = 3
            else:
                pkg_val = 1
        pkg_status_arr = np.array([pkg_val], dtype=np.int32)

        # 5. Local egocentric occupancy window
        local_occ = self._extract_local_occupancy(robot.position, warehouse, fleet)

        # 6. Charging station distance
        min_charging_dist = float("inf")
        if charging_stations:
            for station in charging_stations.values():
                dist = robot.position.manhattan_distance(station.position)
                if dist < min_charging_dist:
                    min_charging_dist = float(dist)
        if min_charging_dist == float("inf"):
            min_charging_dist = 0.0
        charging_dist_arr = np.array([min_charging_dist], dtype=np.float32)

        # 7. Task status (0: None, 1: Created, 2: Assigned, 3: In Progress, 4: Completed)
        task_val = 0
        if task is not None:
            if task.status == TaskStatus.CREATED:
                task_val = 1
            elif task.status == TaskStatus.ASSIGNED:
                task_val = 2
            elif task.status == TaskStatus.IN_PROGRESS:
                task_val = 3
            elif task.status == TaskStatus.COMPLETED:
                task_val = 4
        task_status_arr = np.array([task_val], dtype=np.int32)

        return {
            "robot_position": robot_pos_arr,
            "goal_position": goal_pos_arr,
            "battery_level": battery_arr,
            "package_status": pkg_status_arr,
            "local_occupancy": local_occ,
            "charging_station_distance": charging_dist_arr,
            "task_status": task_status_arr,
        }

    def _extract_local_occupancy(
        self, center_pos: Position, warehouse: Warehouse, fleet: Dict[str, Robot]
    ) -> np.ndarray:
        """Extracts a (2*R+1, 2*R+1) egocentric local occupancy grid centered at center_pos.

        Grid Cell Values:
            0: Empty / Walkable
            1: Obstacle
            2: Shelf
            3: Charging Station
            4: Other Robot
            5: Out of Bounds / Self Center
        """
        R = self.config.observation_radius
        win_size = R * 2 + 1
        window = np.zeros((win_size, win_size), dtype=np.int32)

        other_robot_positions = {
            r.position for r_id, r in fleet.items() if r.position != center_pos
        }

        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                target_pos = Position(center_pos.x + dx, center_pos.y + dy)
                grid_y = dy + R
                grid_x = dx + R

                if not warehouse.is_in_bounds(target_pos):
                    window[grid_y, grid_x] = 5
                elif target_pos == center_pos:
                    window[grid_y, grid_x] = 0  # Self center walkable
                elif target_pos in other_robot_positions:
                    window[grid_y, grid_x] = 4
                else:
                    cell = warehouse.get_cell(target_pos)
                    if cell.cell_type == CellType.OBSTACLE:
                        window[grid_y, grid_x] = 1
                    elif cell.cell_type == CellType.SHELF:
                        window[grid_y, grid_x] = 2
                    elif cell.cell_type == CellType.CHARGING_STATION:
                        window[grid_y, grid_x] = 3
                    else:
                        window[grid_y, grid_x] = 0

        return window
