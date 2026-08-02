"""Unit tests for Space-Time A* Path Planner."""

from simulator.astar import AStarPlanner, euclidean_heuristic, manhattan_heuristic
from simulator.cell import CellType
from simulator.grid import Grid
from simulator.position import Position
from simulator.reservation_table import ReservationTable


def test_astar_simple_path() -> None:
    grid = Grid(10, 10)
    planner = AStarPlanner()
    start = Position(0, 0)
    goal = Position(3, 0)

    res = planner.plan(grid, start, goal)
    assert res.success is True
    assert res.path[0] == start
    assert res.path[-1] == goal
    assert len(res.path) == 4
    assert res.total_cost == 3.0


def test_astar_around_obstacle() -> None:
    grid = Grid(5, 5)
    # Block cell (1, 0)
    grid.set_cell_type(Position(1, 0), CellType.OBSTACLE)

    planner = AStarPlanner(heuristic=manhattan_heuristic)
    res = planner.plan(grid, Position(0, 0), Position(2, 0))

    assert res.success is True
    assert Position(1, 0) not in res.path
    assert res.path[0] == Position(0, 0)
    assert res.path[-1] == Position(2, 0)


def test_astar_unreachable_goal() -> None:
    grid = Grid(5, 5)
    goal = Position(2, 2)
    grid.set_cell_type(goal, CellType.OBSTACLE)

    planner = AStarPlanner()
    res = planner.plan(grid, Position(0, 0), goal)
    assert res.success is False
    assert "non-traversable" in res.error_message


def test_astar_with_reservation_table_wait_action() -> None:
    grid = Grid(5, 5)
    rt = ReservationTable()

    # Reserve cell (1, 0) at timestep 1 for robot r_other
    rt.reserve_vertex("r_other", Position(1, 0), timestep=1)

    planner = AStarPlanner()
    res = planner.plan(
        grid, start=Position(0, 0), goal=Position(2, 0), start_timestep=0, reservation_table=rt, robot_id="r1"
    )

    assert res.success is True
    # Should wait at (0, 0) at timestep 1, then proceed to (1, 0) at timestep 2
    assert res.path[0] == Position(0, 0)
    assert res.path[1] == Position(0, 0)  # Wait step
    assert res.path[2] == Position(1, 0)
    assert res.path[3] == Position(2, 0)
