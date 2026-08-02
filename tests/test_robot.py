"""Unit tests for Robot lifecycle and deterministic movement."""

import pytest
from simulator.constants import Direction
from simulator.exceptions import WarehouseError
from simulator.package import Package, PackageStatus
from simulator.position import Position
from simulator.robot import Robot, RobotState
from simulator.task import Task, TaskPriority, TaskType


def test_robot_initialization() -> None:
    robot = Robot("r1", Position(2, 3), max_battery=100.0)
    assert robot.robot_id == "r1"
    assert robot.position == Position(2, 3)
    assert robot.battery_level == 100.0
    assert robot.battery_percentage == 100.0
    assert robot.state == RobotState.IDLE
    assert robot.is_idle() is True
    assert robot.carrying_package is None
    assert robot.assigned_task is None


def test_robot_invalid_battery() -> None:
    with pytest.raises(WarehouseError):
        Robot("r1", Position(0, 0), max_battery=-10.0)


def test_robot_deterministic_movement() -> None:
    robot = Robot("r1", Position(0, 0))
    target = Position(2, 1)

    # Step 1: moves EAST towards (1, 0)
    dir1 = robot.step_towards(target)
    assert dir1 == Direction.EAST
    assert robot.position == Position(1, 0)

    # Step 2: moves EAST towards (2, 0)
    dir2 = robot.step_towards(target)
    assert dir2 == Direction.EAST
    assert robot.position == Position(2, 0)

    # Step 3: moves SOUTH towards (2, 1)
    dir3 = robot.step_towards(target)
    assert dir3 == Direction.SOUTH
    assert robot.position == Position(2, 1)

    # Step 4: already at target, returns STAY
    dir4 = robot.step_towards(target)
    assert dir4 == Direction.STAY
    assert robot.position == Position(2, 1)
    assert robot.total_distance_travelled == 3


def test_robot_task_assignment_and_pickup_drop() -> None:
    robot = Robot("r1", Position(0, 0))
    pkg = Package("p1", Position(1, 1), Position(3, 3))
    task = Task("t1", TaskType.PICKUP_AND_DELIVER, Position(1, 1), Position(3, 3), package=pkg)

    robot.assign_task(task)
    assert robot.assigned_task == task
    assert robot.state == RobotState.MOVING_TO_PICKUP

    # Cannot assign task when robot already has active task
    task2 = Task("t2", TaskType.PICKUP_AND_DELIVER, Position(0, 0), Position(1, 1))
    with pytest.raises(WarehouseError):
        robot.assign_task(task2)

    robot.pick_up_package(pkg)
    assert robot.carrying_package == pkg
    assert pkg.status == PackageStatus.IN_TRANSIT
    assert robot.state == RobotState.MOVING_TO_DROP

    dropped_pkg = robot.drop_package()
    assert dropped_pkg == pkg
    assert pkg.status == PackageStatus.DELIVERED
    assert robot.carrying_package is None
