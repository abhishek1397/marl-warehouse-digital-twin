"""Multi-modal rendering engine supporting human ASCII mode and rgb_array NumPy image export."""

from typing import Dict, Optional

import numpy as np

from simulator.cell import CellType
from simulator.position import Position
from simulator.robot import Robot
from simulator.visualization import render_ascii_grid
from simulator.warehouse import Warehouse


class EnvironmentRenderer:
    """Renders warehouse environment into ASCII text or RGB image arrays."""

    # RGB Color Palette [R, G, B]
    COLOR_EMPTY = [245, 245, 245]        # Light Gray/White
    COLOR_OBSTACLE = [50, 50, 50]        # Dark Gray
    COLOR_SHELF = [210, 105, 30]         # Chocolate Brown
    COLOR_CHARGER = [34, 139, 34]        # Forest Green
    COLOR_PICKUP = [147, 112, 219]       # Medium Purple
    COLOR_DROP = [220, 20, 60]           # Crimson Red
    COLOR_ROBOT = [30, 144, 255]         # Dodger Blue

    def __init__(self, cell_pixel_size: int = 16) -> None:
        self.cell_pixel_size: int = cell_pixel_size

    def render(
        self,
        mode: str,
        warehouse: Warehouse,
        fleet: Dict[str, Robot],
    ) -> Optional[np.ndarray | str]:
        """Renders environment based on mode ('human' or 'rgb_array').

        Returns:
            Formatted ASCII string for 'human' mode, or uint8 RGB NumPy image array for 'rgb_array' mode.
        """
        if mode == "human":
            return render_ascii_grid(warehouse, fleet=fleet)
        elif mode == "rgb_array":
            return self._render_rgb_array(warehouse, fleet)
        return None

    def _render_rgb_array(
        self, warehouse: Warehouse, fleet: Dict[str, Robot]
    ) -> np.ndarray:
        """Renders grid matrix into a (H * cell_size, W * cell_size, 3) uint8 RGB array."""
        grid = warehouse.grid
        width = grid.width
        height = grid.height
        c_size = self.cell_pixel_size

        img_height = height * c_size
        img_width = width * c_size
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        robot_positions = {r.position for r in fleet.values()}

        for y in range(height):
            for x in range(width):
                pos = Position(x, y)
                cell = grid.get_cell(pos)

                if pos in robot_positions:
                    color = self.COLOR_ROBOT
                elif cell.cell_type == CellType.OBSTACLE:
                    color = self.COLOR_OBSTACLE
                elif cell.cell_type == CellType.SHELF:
                    color = self.COLOR_SHELF
                elif cell.cell_type == CellType.CHARGING_STATION:
                    color = self.COLOR_CHARGER
                elif cell.cell_type == CellType.PICKUP_ZONE:
                    color = self.COLOR_PICKUP
                elif cell.cell_type == CellType.DROP_ZONE:
                    color = self.COLOR_DROP
                else:
                    color = self.COLOR_EMPTY

                y_start, y_end = y * c_size, (y + 1) * c_size
                x_start, x_end = x * c_size, (x + 1) * c_size
                img[y_start:y_end, x_start:x_end] = color

        return img
