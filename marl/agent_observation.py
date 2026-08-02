"""AgentObservationEncoder for local, global, and hybrid multi-agent observations."""

from typing import Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as GymDict

from marl.communication import CommunicationManager
from marl.multi_agent_config import MultiAgentEnvConfig
from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.package import PackageStatus
from simulator.position import Position
from simulator.robot import Robot
from simulator.task import Task, TaskStatus
from simulator.warehouse import Warehouse


class AgentObservationEncoder:
    """Encodes per-agent observations supporting local, global, and hybrid observation spaces."""

    def __init__(self, config: MultiAgentEnvConfig) -> None:
        self.config: MultiAgentEnvConfig = config

    def get_observation_space(self, agent_id: str) -> GymDict:
        """Returns Gymnasium Dict observation space for an individual agent."""
        max_dim = max(self.config.grid_width, self.config.grid_height)
        R = self.config.observation_radius
        win_size = R * 2 + 1

        spaces_dict = {
            "robot_position": Box(low=0, high=max_dim, shape=(2,), dtype=np.int32),
            "goal_position": Box(low=0, high=max_dim, shape=(2,), dtype=np.int32),
            "battery_level": Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "package_status": Box(low=0, high=3, shape=(1,), dtype=np.int32),
            "local_occupancy": Box(low=0, high=5, shape=(win_size, win_size), dtype=np.int32),
            "charging_station_distance": Box(low=0.0, high=float(max_dim * 2), shape=(1,), dtype=np.float32),
            "comm_message": Box(low=-100.0, high=100.0, shape=(self.config.comm_msg_dim,), dtype=np.float32),
        }

        if self.config.observation_mode in {"global", "hybrid"}:
            spaces_dict["global_fleet_summary"] = Box(
                low=0.0, high=float(max_dim), shape=(self.config.num_robots * 3,), dtype=np.float32
            )

        return GymDict(spaces_dict)

    def encode(
        self,
        robot: Robot,
        warehouse: Warehouse,
        fleet: Dict[str, Robot],
        comm_manager: CommunicationManager,
        task: Optional[Task] = None,
        charging_stations: Optional[Dict[str, ChargingStation]] = None,
    ) -> Dict[str, np.ndarray]:
        """Encodes state observation dictionary for a specific robot agent.

        Returns:
            Dict containing NumPy arrays matching the agent's observation space.
        """
        robot_pos_arr = np.array([robot.position.x, robot.position.y], dtype=np.int32)

        goal_pos = Position(0, 0)
        if task is not None:
            if robot.carrying_package is not None:
                goal_pos = task.drop_position
            else:
                goal_pos = task.pickup_position
        goal_pos_arr = np.array([goal_pos.x, goal_pos.y], dtype=np.int32)

        battery_arr = np.array([float(robot.battery_level)], dtype=np.float32)

        pkg_val = 0
        if robot.carrying_package is not None:
            pkg_val = 2
        elif task is not None and task.package is not None:
            pkg_val = 3 if task.package.status == PackageStatus.DELIVERED else 1
        pkg_status_arr = np.array([pkg_val], dtype=np.int32)

        # Local egocentric occupancy window
        local_occ = self._extract_local_occupancy(robot.position, warehouse, fleet)

        # Charging station distance
        min_charging_dist = float("inf")
        if charging_stations:
            for station in charging_stations.values():
                dist = robot.position.manhattan_distance(station.position)
                if dist < min_charging_dist:
                    min_charging_dist = float(dist)
        if min_charging_dist == float("inf"):
            min_charging_dist = 0.0
        charging_dist_arr = np.array([min_charging_dist], dtype=np.float32)

        # Communication message received from other agents
        comm_msg = comm_manager.get_received_messages(robot.robot_id, robot.position, fleet)

        obs = {
            "robot_position": robot_pos_arr,
            "goal_position": goal_pos_arr,
            "battery_level": battery_arr,
            "package_status": pkg_status_arr,
            "local_occupancy": local_occ,
            "charging_station_distance": charging_dist_arr,
            "comm_message": comm_msg,
        }

        # Global or Hybrid summary features
        if self.config.observation_mode in {"global", "hybrid"}:
            fleet_summary: List[float] = []
            for r_id in [f"robot_{i}" for i in range(self.config.num_robots)]:
                r = fleet.get(r_id)
                if r is not None:
                    fleet_summary.extend([float(r.position.x), float(r.position.y), float(r.battery_level)])
                else:
                    fleet_summary.extend([0.0, 0.0, 0.0])
            obs["global_fleet_summary"] = np.array(fleet_summary, dtype=np.float32)

        return obs

    def extract_global_state(
        self, warehouse: Warehouse, fleet: Dict[str, Robot]
    ) -> np.ndarray:
        """Extracts full environment global state vector for centralized critics in MAPPO/QMIX."""
        grid = warehouse.grid
        state_grid = np.zeros((grid.height, grid.width), dtype=np.float32)

        for y in range(grid.height):
            for x in range(grid.width):
                pos = Position(x, y)
                cell = grid.get_cell(pos)
                if cell.cell_type == CellType.OBSTACLE:
                    state_grid[y, x] = 1.0
                elif cell.cell_type == CellType.SHELF:
                    state_grid[y, x] = 2.0
                elif cell.cell_type == CellType.CHARGING_STATION:
                    state_grid[y, x] = 3.0

        for r_id, r in fleet.items():
            if 0 <= r.position.y < grid.height and 0 <= r.position.x < grid.width:
                state_grid[r.position.y, r.position.x] = 4.0

        return state_grid

    def _extract_local_occupancy(
        self, center_pos: Position, warehouse: Warehouse, fleet: Dict[str, Robot]
    ) -> np.ndarray:
        R = self.config.observation_radius
        win_size = R * 2 + 1
        window = np.zeros((win_size, win_size), dtype=np.int32)

        other_robot_positions = {
            r.position for r in fleet.values() if r.position != center_pos
        }

        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                target_pos = Position(center_pos.x + dx, center_pos.y + dy)
                grid_y = dy + R
                grid_x = dx + R

                if not warehouse.is_in_bounds(target_pos):
                    window[grid_y, grid_x] = 5
                elif target_pos == center_pos:
                    window[grid_y, grid_x] = 0
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
