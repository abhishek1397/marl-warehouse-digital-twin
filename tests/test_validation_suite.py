"""Validation suite verifying simulation integrity, robot behavior, and classical planner correctness."""

import pytest
from simulator.astar import AStarPlanner
from simulator.battery_manager import BatteryManager
from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.collision_detector import CollisionDetector
from simulator.exceptions import OutOfBoundsError
from simulator.grid import Grid
from simulator.metrics import MetricsCollector
from simulator.obstacle import Obstacle
from simulator.package import Package, PackageStatus
from simulator.planner import MultiRobotPlanner, PlanningRequest
from simulator.position import Position
from simulator.reservation_table import ReservationTable
from simulator.robot import Robot, RobotState
from simulator.shelf import Shelf
from simulator.simulation import SimulationEngine, SimulationState
from simulator.task import Task, TaskPriority, TaskStatus, TaskType
from simulator.task_manager import TaskManager
from simulator.warehouse import Warehouse


# =========================================================
# PART 1: SIMULATION VALIDATION
# =========================================================

def test_part1_warehouse_and_entity_placement_bounds() -> None:
    wh = Warehouse(20, 20, name="ValidationWarehouse")
    assert wh.width == 20
    assert wh.height == 20

    # Place entities and verify bounds
    shelf = Shelf("s1", Position(5, 5))
    obs = Obstacle("o1", Position(10, 10))
    station = ChargingStation("c1", Position(0, 19))

    cell_s = wh.place_object(shelf.position, shelf, cell_type=CellType.SHELF)
    cell_o = wh.place_object(obs.position, obs, cell_type=CellType.OBSTACLE)
    cell_c = wh.place_object(station.position, station, cell_type=CellType.CHARGING_STATION)

    assert wh.is_in_bounds(shelf.position) is True
    assert wh.is_in_bounds(obs.position) is True
    assert wh.is_in_bounds(station.position) is True

    # Out of bounds placement should raise exception
    with pytest.raises(OutOfBoundsError):
        wh.place_object(Position(25, 25), shelf)


# =========================================================
# PART 2: ROBOT VALIDATION
# =========================================================

def test_part2_robot_lifecycle_and_battery() -> None:
    wh = Warehouse(10, 10)
    engine = SimulationEngine(wh)

    robot = Robot("r1", Position(0, 0), max_battery=100.0)
    engine.add_robot(robot)

    pkg = Package("p1", Position(2, 0), Position(4, 0))
    engine.add_package(pkg)
    task = engine.task_manager.create_task(
        "t1", TaskType.PICKUP_AND_DELIVER, Position(2, 0), Position(4, 0), package=pkg
    )

    engine.initialize()

    # Step 1-2: Robot moves towards pickup
    engine.step()
    assert robot.battery_level < 100.0
    assert robot.total_distance_travelled > 0

    # Execute simulation until task completion
    engine.run(max_steps=50)

    assert pkg.is_delivered() is True
    assert task.status == TaskStatus.COMPLETED
    assert robot.tasks_completed == 1
    assert robot.is_idle() is True


def test_part2_robot_battery_recharge_flow() -> None:
    wh = Warehouse(10, 10)
    engine = SimulationEngine(wh)

    robot = Robot("r1", Position(0, 0), max_battery=100.0)
    robot.battery_level = 15.0  # Low battery < 20%
    engine.add_robot(robot)

    station = ChargingStation("c1", Position(1, 0), charge_rate=25.0)
    wh.place_object(station.position, station, cell_type=CellType.CHARGING_STATION)
    engine.add_charging_station(station)

    engine.initialize()
    engine.step()

    assert robot.state in {RobotState.MOVING_TO_CHARGE, RobotState.CHARGING}


# =========================================================
# PART 3: PLANNER VALIDATION
# =========================================================

def test_part3_astar_and_reservation_table_correctness() -> None:
    grid = Grid(10, 10)
    rt = ReservationTable()
    planner = AStarPlanner()

    # 1. Shortest path verification
    res1 = planner.plan(grid, Position(0, 0), Position(3, 0))
    assert res1.success is True
    assert len(res1.path) == 4

    # 2. Blocked path
    grid.set_cell_type(Position(1, 0), CellType.OBSTACLE)
    res2 = planner.plan(grid, Position(0, 0), Position(2, 0))
    assert res2.success is True
    assert Position(1, 0) not in res2.path

    # 3. Dynamic obstacle avoidance via reservation table
    rt.reserve_vertex("r_other", Position(0, 1), timestep=1)
    res3 = planner.plan(
        grid, Position(0, 0), Position(0, 2), start_timestep=0, reservation_table=rt, robot_id="r1"
    )
    assert res3.success is True
    assert (Position(0, 1), 1) not in [(pos, t) for t, pos in enumerate(res3.path)]


def test_part3_multi_robot_collision_prevention() -> None:
    wh = Warehouse(10, 10)
    planner = MultiRobotPlanner()
    cd = CollisionDetector()

    r1 = Robot("r1", Position(0, 0))
    r2 = Robot("r2", Position(4, 0))

    req1 = PlanningRequest(robot=r1, goal_position=Position(4, 0), priority=10)
    req2 = PlanningRequest(robot=r2, goal_position=Position(0, 0), priority=5)

    results = planner.plan_joint_paths(wh.grid, [req1, req2])

    assert results["r1"].success is True
    assert results["r2"].success is True

    joint_paths = {r_id: res.path for r_id, res in results.items()}
    diagnostics = cd.detect_collisions(wh, joint_paths)
    assert len(diagnostics) == 0
