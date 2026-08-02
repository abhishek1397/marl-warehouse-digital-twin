"""Performance and scalability test for random warehouse generation and classical multi-robot planning."""

import random
from simulator.cell import CellType
from simulator.grid import Grid
from simulator.planner import MultiRobotPlanner, PlanningRequest
from simulator.position import Position
from simulator.robot import Robot
from simulator.warehouse import Warehouse


def generate_random_warehouse(
    width: int = 20, height: int = 20, obstacle_ratio: float = 0.1
) -> Warehouse:
    """Generates a random warehouse layout for benchmarking."""
    wh = Warehouse(width, height, name="BenchmarkWarehouse")
    total_cells = width * height
    obstacle_count = int(total_cells * obstacle_ratio)

    random.seed(42)
    for _ in range(obstacle_count):
        rx = random.randint(0, width - 1)
        ry = random.randint(0, height - 1)
        pos = Position(rx, ry)
        if wh.is_in_bounds(pos):
            wh.set_cell_type(pos, CellType.OBSTACLE)

    return wh


def test_random_warehouse_multi_robot_scaling() -> None:
    wh = generate_random_warehouse(width=15, height=15, obstacle_ratio=0.1)
    planner = MultiRobotPlanner()

    # Create 5 robots with distinct start and goal positions
    num_robots = 5
    requests = []

    valid_positions = [
        c.position for c in wh.grid if c.cell_type == CellType.EMPTY
    ]
    random.seed(123)
    random.shuffle(valid_positions)

    for i in range(num_robots):
        start_pos = valid_positions[i * 2]
        goal_pos = valid_positions[i * 2 + 1]

        robot = Robot(f"benchmark_robot_{i}", start_pos)
        req = PlanningRequest(robot=robot, goal_position=goal_pos, priority=num_robots - i)
        requests.append(req)

    results = planner.plan_joint_paths(wh.grid, requests)

    assert len(results) == num_robots
    successful_plans = sum(1 for res in results.values() if res.success)
    # Most or all robots should successfully find collision-free paths
    assert successful_plans >= num_robots - 1
