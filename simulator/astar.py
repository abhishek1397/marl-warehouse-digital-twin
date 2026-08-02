"""Space-Time A* Path Planner for single and multi-robot navigation."""

import heapq
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from simulator.constants import Direction
from simulator.grid import Grid
from simulator.position import Position
from simulator.reservation_table import ReservationTable


def manhattan_heuristic(pos: Position, goal: Position) -> float:
    """Calculates Manhattan distance (L1 norm) heuristic between pos and goal."""
    return float(pos.manhattan_distance(goal))


def euclidean_heuristic(pos: Position, goal: Position) -> float:
    """Calculates Euclidean distance (L2 norm) heuristic between pos and goal."""
    return float(pos.euclidean_distance(goal))


@dataclass
class PathResult:
    """Dataclass encapsulating A* search results and performance metrics."""

    path: List[Position] = field(default_factory=list)
    total_cost: float = 0.0
    expanded_nodes: int = 0
    planning_time_ms: float = 0.0
    success: bool = False
    error_message: str = ""


class AStarPlanner:
    """Space-Time A* Path Planner with 4-cardinal movement, wait actions, and reservation table support."""

    def __init__(
        self,
        heuristic: Optional[Callable[[Position, Position], float]] = None,
        move_cost: float = 1.0,
        wait_cost: float = 1.0,
    ) -> None:
        self.heuristic: Callable[[Position, Position], float] = (
            heuristic if heuristic is not None else manhattan_heuristic
        )
        self.move_cost: float = move_cost
        self.wait_cost: float = wait_cost

    def plan(
        self,
        grid: Grid,
        start: Position,
        goal: Position,
        start_timestep: int = 0,
        reservation_table: Optional[ReservationTable] = None,
        max_steps: int = 1000,
        robot_id: Optional[str] = None,
    ) -> PathResult:
        """Executes Space-Time A* path planning from start to goal.

        Args:
            grid: Warehouse grid environment.
            start: Origin position.
            goal: Target destination position.
            start_timestep: Simulation step when path begins.
            reservation_table: Optional space-time reservation table to avoid collisions.
            max_steps: Maximum search depth/expansions.
            robot_id: Optional ID of the planning robot.

        Returns:
            PathResult object containing path, cost, expanded nodes, and planning time.
        """
        start_time = time.perf_counter()

        if not grid.is_in_bounds(start):
            return PathResult(
                success=False,
                error_message=f"Start position {start} is out of grid bounds.",
            )

        if not grid.is_in_bounds(goal):
            return PathResult(
                success=False,
                error_message=f"Goal position {goal} is out of grid bounds.",
            )

        start_cell = grid.get_cell(start)
        goal_cell = grid.get_cell(goal)

        if not start_cell.cell_type.is_traversable:
            return PathResult(
                success=False,
                error_message=f"Start cell at {start} is non-traversable ({start_cell.cell_type.name}).",
            )

        if not goal_cell.cell_type.is_traversable:
            return PathResult(
                success=False,
                error_message=f"Goal cell at {goal} is non-traversable ({goal_cell.cell_type.name}).",
            )

        if start == goal:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PathResult(
                path=[start],
                total_cost=0.0,
                expanded_nodes=0,
                planning_time_ms=elapsed_ms,
                success=True,
            )

        # Priority queue entries: (f_score, g_score, position, timestep)
        # Node identity in space-time is (position, timestep)
        open_heap: List[Tuple[float, float, Position, int]] = []
        heapq.heappush(open_heap, (self.heuristic(start, goal), 0.0, start, start_timestep))

        # Stores g_score: (Position, timestep) -> float
        g_scores: Dict[Tuple[Position, int], float] = {(start, start_timestep): 0.0}

        # Parent pointer map: (Position, timestep) -> (Position, timestep)
        came_from: Dict[Tuple[Position, int], Tuple[Position, int]] = {}

        closed_set: Set[Tuple[Position, int]] = set()
        expanded_nodes = 0

        cardinal_dirs = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

        while open_heap and expanded_nodes < max_steps:
            f, current_g, current_pos, current_t = heapq.heappop(open_heap)

            state = (current_pos, current_t)
            if state in closed_set:
                continue

            closed_set.add(state)
            expanded_nodes += 1

            # Check if goal is reached
            if current_pos == goal:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                path = self._reconstruct_path(came_from, current_pos, current_t, start_timestep)
                return PathResult(
                    path=path,
                    total_cost=current_g,
                    expanded_nodes=expanded_nodes,
                    planning_time_ms=elapsed_ms,
                    success=True,
                )

            next_t = current_t + 1

            # Generate candidate actions: 4 cardinal movements + STAY (wait)
            candidates: List[Tuple[Position, float]] = []
            for direction in cardinal_dirs:
                neighbor_pos = current_pos.get_neighbor(direction)
                if grid.is_in_bounds(neighbor_pos):
                    cell = grid.get_cell(neighbor_pos)
                    if cell.cell_type.is_traversable:
                        candidates.append((neighbor_pos, self.move_cost))

            # Add STAY (wait in current cell)
            candidates.append((current_pos, self.wait_cost))

            for next_pos, step_cost in candidates:
                next_state = (next_pos, next_t)
                if next_state in closed_set:
                    continue

                # Check space-time reservation table
                if reservation_table is not None:
                    if reservation_table.is_vertex_reserved(next_pos, next_t, ignore_robot_id=robot_id):
                        continue
                    if reservation_table.is_edge_reserved(current_pos, next_pos, next_t, ignore_robot_id=robot_id):
                        continue

                tentative_g = current_g + step_cost

                if next_state not in g_scores or tentative_g < g_scores[next_state]:
                    g_scores[next_state] = tentative_g
                    came_from[next_state] = state
                    f_score = tentative_g + self.heuristic(next_pos, goal)
                    heapq.heappush(open_heap, (f_score, tentative_g, next_pos, next_t))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return PathResult(
            path=[],
            total_cost=0.0,
            expanded_nodes=expanded_nodes,
            planning_time_ms=elapsed_ms,
            success=False,
            error_message="No valid path found within step limit.",
        )

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[Position, int], Tuple[Position, int]],
        curr_pos: Position,
        curr_t: int,
        start_t: int,
    ) -> List[Position]:
        path_states: List[Tuple[Position, int]] = [(curr_pos, curr_t)]
        state = (curr_pos, curr_t)

        while state in came_from:
            state = came_from[state]
            path_states.append(state)

        path_states.reverse()
        return [pos for pos, _ in path_states]
