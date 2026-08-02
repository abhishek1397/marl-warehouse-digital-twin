"""Unit tests for TaskManager task queueing, prioritization, and lifecycle."""

import pytest
from simulator.exceptions import WarehouseError
from simulator.package import Package
from simulator.position import Position
from simulator.robot import Robot
from simulator.task import TaskPriority, TaskStatus, TaskType
from simulator.task_manager import TaskManager


def test_task_manager_creation_and_prioritization() -> None:
    tm = TaskManager()
    t1 = tm.create_task(
        "t1", TaskType.PICKUP_AND_DELIVER, Position(0, 0), Position(5, 5), priority=TaskPriority.LOW
    )
    t2 = tm.create_task(
        "t2", TaskType.PICKUP_AND_DELIVER, Position(1, 1), Position(6, 6), priority=TaskPriority.HIGH
    )
    t3 = tm.create_task(
        "t3", TaskType.PICKUP_AND_DELIVER, Position(2, 2), Position(7, 7), priority=TaskPriority.CRITICAL
    )

    assert tm.total_tasks == 3
    assert tm.unassigned_count == 3

    # Priority queue order should be t3 (CRITICAL), t2 (HIGH), t1 (LOW)
    pending = tm.get_pending_tasks()
    assert pending[0] == t3
    assert pending[1] == t2
    assert pending[2] == t1


def test_task_manager_assignment_to_robot() -> None:
    tm = TaskManager()
    robot = Robot("r1", Position(0, 0))
    t1 = tm.create_task(
        "t1", TaskType.PICKUP_AND_DELIVER, Position(0, 0), Position(5, 5), priority=TaskPriority.MEDIUM
    )

    assigned = tm.assign_next_task(robot)
    assert assigned == t1
    assert robot.assigned_task == t1
    assert t1.assigned_robot_id == "r1"
    assert t1.status == TaskStatus.ASSIGNED
    assert tm.unassigned_count == 0


def test_task_manager_completion_and_cancellation() -> None:
    tm = TaskManager()
    pkg = Package("p1", Position(0, 0), Position(2, 2))
    t1 = tm.create_task(
        "t1", TaskType.PICKUP_AND_DELIVER, Position(0, 0), Position(2, 2), package=pkg
    )

    tm.complete_task("t1", step=10)
    assert t1.status == TaskStatus.COMPLETED
    assert t1.completed_at_step == 10
    assert pkg.is_delivered() is True
    assert tm.all_tasks_completed() is True

    t2 = tm.create_task("t2", TaskType.IDLE_PARK, Position(0, 0), Position(0, 0))
    assert tm.all_tasks_completed() is False

    tm.cancel_task("t2")
    assert t2.status == TaskStatus.CANCELLED
    assert tm.all_tasks_completed() is True


def test_duplicate_task_error() -> None:
    tm = TaskManager()
    tm.create_task("t1", TaskType.IDLE_PARK, Position(0, 0), Position(0, 0))
    with pytest.raises(WarehouseError):
        tm.create_task("t1", TaskType.IDLE_PARK, Position(0, 0), Position(0, 0))
