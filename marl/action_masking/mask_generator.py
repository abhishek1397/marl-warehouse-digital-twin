"""ActionMaskGenerator module generating boolean action validity masks for warehouse robots."""

from typing import Dict, Optional

import numpy as np
import torch

from marl.action_masking.action_mask import ActionMask
from marl.action_masking.config import ActionMaskConfig
from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.position import Position
from simulator.robot import Robot
from simulator.task import Task
from simulator.warehouse import Warehouse


class ActionMaskGenerator:
    """Generates dynamic boolean action validity masks based on robot state and warehouse layout."""

    def __init__(self, config: Optional[ActionMaskConfig] = None) -> None:
        self.config: ActionMaskConfig = config or ActionMaskConfig()

    def generate_mask(
        self,
        robot: Robot,
        warehouse: Warehouse,
        task: Optional[Task] = None,
        charging_stations: Optional[Dict[str, ChargingStation]] = None,
        low_battery_threshold: float = 20.0,
    ) -> ActionMask:
        """Computes boolean validity mask for 8 discrete actions:
        0: Move Up (y - 1)
        1: Move Down (y + 1)
        2: Move Left (x - 1)
        3: Move Right (x + 1)
        4: Wait
        5: Pick Package
        6: Drop Package
        7: Go Charge
        """
        mask = np.ones(8, dtype=bool)

        if not self.config.enable_action_masking:
            return ActionMask(mask_array=mask, mask_tensor=torch.tensor(mask, dtype=torch.bool))

        grid = warehouse.grid
        curr_pos = robot.position

        # 1. Movement Actions (0-3)
        if self.config.mask_invalid_moves:
            move_offsets = [
                (0, -1),  # Up
                (0, 1),   # Down
                (-1, 0),  # Left
                (1, 0),   # Right
            ]

            for act_idx, (dx, dy) in enumerate(move_offsets):
                nx, ny = curr_pos.x + dx, curr_pos.y + dy
                target_pos = Position(nx, ny)

                if not grid.is_in_bounds(target_pos):
                    mask[act_idx] = False
                else:
                    cell = grid.get_cell(target_pos)
                    if not cell.cell_type.is_traversable or cell.is_occupied:
                        mask[act_idx] = False

        # 2. Action 4: Wait (Always Valid)
        mask[4] = True

        # 3. Action 5: Pick Package
        if self.config.mask_invalid_pick:
            is_pick_valid = False
            if robot.carrying_package is None and task is not None:
                # Valid if sitting directly on or adjacent to pickup position
                dist_to_pickup = abs(curr_pos.x - task.pickup_position.x) + abs(curr_pos.y - task.pickup_position.y)
                if dist_to_pickup <= 1:
                    is_pick_valid = True
            mask[5] = is_pick_valid

        # 4. Action 6: Drop Package
        if self.config.mask_invalid_drop:
            is_drop_valid = False
            if robot.carrying_package is not None and task is not None:
                if curr_pos == task.drop_position:
                    is_drop_valid = True
            mask[6] = is_drop_valid

        # 5. Action 7: Go Charge
        if self.config.mask_invalid_charge:
            is_charge_valid = False
            if charging_stations:
                for cs in charging_stations.values():
                    if cs.position == curr_pos and robot.battery_level < 100.0:
                        is_charge_valid = True
                        break
            mask[7] = is_charge_valid

        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        return ActionMask(mask_array=mask, mask_tensor=mask_tensor)
