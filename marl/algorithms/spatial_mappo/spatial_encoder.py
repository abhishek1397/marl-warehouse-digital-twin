"""WarehouseSpatialEncoder encoding warehouse state into a 5-channel 2D spatial grid tensor S."""

from typing import Dict, Optional

import numpy as np
import torch

from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.position import Position
from simulator.robot import Robot
from simulator.shelf import Shelf
from simulator.warehouse import Warehouse


class WarehouseSpatialEncoder:
    """Encodes warehouse entities into a 5-channel 2D spatial grid tensor S in R^(5 x H x W)."""

    def __init__(self, in_channels: int = 5) -> None:
        self.in_channels: int = in_channels

    def encode_spatial_state(
        self,
        warehouse: Warehouse,
        fleet: Dict[str, Robot],
        charging_stations: Optional[Dict[str, ChargingStation]] = None,
        shelves: Optional[Dict[str, Shelf]] = None,
    ) -> np.ndarray:
        """Constructs 5-channel 2D spatial grid array of shape (5, height, width)."""
        h, w = warehouse.height, warehouse.width
        spatial_tensor = np.zeros((5, h, w), dtype=np.float32)

        # Channel 0: Fleet Robots (1.0 for position, battery level scaling)
        for robot in fleet.values():
            pos = robot.position
            if 0 <= pos.x < w and 0 <= pos.y < h:
                battery_factor = float(robot.battery_level / robot.max_battery)
                spatial_tensor[0, pos.y, pos.x] = battery_factor

        # Channel 1: Shelves
        if shelves:
            for shelf in shelves.values():
                pos = shelf.position
                if 0 <= pos.x < w and 0 <= pos.y < h:
                    spatial_tensor[1, pos.y, pos.x] = 1.0

        # Channel 2: Obstacles & Walls
        for y in range(h):
            for x in range(w):
                cell = warehouse.grid.get_cell(Position(x, y))
                if cell and cell.cell_type == CellType.OBSTACLE:
                    spatial_tensor[2, y, x] = 1.0

        # Channel 3: Charging Stations
        if charging_stations:
            for station in charging_stations.values():
                pos = station.position
                if 0 <= pos.x < w and 0 <= pos.y < h:
                    spatial_tensor[3, pos.y, pos.x] = 1.0

        # Channel 4: Packages / Destinations
        for robot in fleet.values():
            if robot.assigned_task:
                pkg_pos = robot.assigned_task.pickup_position
                drop_pos = robot.assigned_task.drop_position
                if 0 <= pkg_pos.x < w and 0 <= pkg_pos.y < h:
                    spatial_tensor[4, pkg_pos.y, pkg_pos.x] = 1.0
                if 0 <= drop_pos.x < w and 0 <= drop_pos.y < h:
                    spatial_tensor[4, drop_pos.y, drop_pos.x] = 0.5

        return spatial_tensor
