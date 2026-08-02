"""ASCII visualization utility for terminal debugging and visual state inspection."""

from typing import Dict, List, Optional, Set

from simulator.cell import CellType
from simulator.position import Position
from simulator.robot import Robot
from simulator.warehouse import Warehouse


def render_ascii_grid(
    warehouse: Warehouse,
    fleet: Optional[Dict[str, Robot]] = None,
    planned_paths: Optional[Dict[str, List[Position]]] = None,
    show_legend: bool = True,
) -> str:
    """Renders an ASCII text grid matrix of the warehouse environment.

    Args:
        warehouse: Warehouse grid instance.
        fleet: Map of active robots to display.
        planned_paths: Optional map of robot_id -> list of path Positions to overlay.
        show_legend: If True, appends a legend key at the bottom.

    Returns:
        Formatted ASCII string representation.
    """
    grid = warehouse.grid
    width = grid.width
    height = grid.height

    # Build robot lookup: Position -> display symbol
    robot_map: Dict[Position, str] = {}
    if fleet:
        for idx, (r_id, robot) in enumerate(fleet.items(), start=1):
            # Display first letter or numeric index
            symbol = f"{idx % 10}"
            robot_map[robot.position] = symbol

    # Build path overlay lookup: Position -> '*'
    path_positions: Set[Position] = set()
    if planned_paths:
        for path in planned_paths.values():
            path_positions.update(path)

    lines: List[str] = []
    lines.append("+" + "-" * (width * 2 + 1) + "+")

    for y in range(height):
        row_str = ["| "]
        for x in range(width):
            pos = Position(x, y)
            cell = grid.get_cell(pos)

            if pos in robot_map:
                row_str.append(robot_map[pos] + " ")
            elif pos in path_positions:
                row_str.append("* ")
            elif cell.cell_type == CellType.OBSTACLE:
                row_str.append("# ")
            elif cell.cell_type == CellType.SHELF:
                row_str.append("S ")
            elif cell.cell_type == CellType.CHARGING_STATION:
                row_str.append("C ")
            elif cell.cell_type == CellType.PICKUP_ZONE:
                row_str.append("P ")
            elif cell.cell_type == CellType.DROP_ZONE:
                row_str.append("D ")
            else:
                row_str.append(". ")

        row_str.append("|")
        lines.append("".join(row_str))

    lines.append("+" + "-" * (width * 2 + 1) + "+")

    if show_legend:
        lines.append(
            "Legend: [. Empty] [# Obstacle] [S Shelf] [C Charger] [P Pickup] [D Drop] [* Path] [0-9 Robot]"
        )

    return "\n".join(lines)
