"""Unit tests for TaskScheduler strategies."""

from simulator.position import Position
from simulator.robot import Robot
from simulator.scheduler import (
    FIFOSchedulerStrategy,
    NearestRobotSchedulerStrategy,
    PrioritySchedulerStrategy,
    TaskScheduler,
)
from simulator.task import Task, TaskPriority, TaskType


def test_fifo_scheduler_strategy() -> None:
    t1 = Task("t1", TaskType.PICKUP_AND_DELIVER, Position(0, 0), Position(1, 1), created_at_step=1)
    t2 = Task("t2", TaskType.PICKUP_AND_DELIVER, Position(5, 5), Position(6, 6), created_at_step=2)

    r1 = Robot("r1", Position(0, 0))

    scheduler = TaskScheduler(strategy=FIFOSchedulerStrategy())
    assignments = scheduler.schedule([t1, t2], [r1])

    assert len(assignments) == 1
    assert assignments[0][0] == t1
    assert assignments[0][1] == r1


def test_priority_scheduler_strategy() -> None:
    t1 = Task("t1", TaskType.PICKUP_AND_DELIVER, Position(0, 0), Position(1, 1), priority=TaskPriority.LOW)
    t2 = Task("t2", TaskType.PICKUP_AND_DELIVER, Position(5, 5), Position(6, 6), priority=TaskPriority.HIGH)

    r1 = Robot("r1", Position(0, 0))

    scheduler = TaskScheduler(strategy=PrioritySchedulerStrategy())
    assignments = scheduler.schedule([t1, t2], [r1])

    assert len(assignments) == 1
    assert assignments[0][0] == t2


def test_nearest_robot_scheduler_strategy() -> None:
    t1 = Task("t1", TaskType.PICKUP_AND_DELIVER, Position(10, 10), Position(11, 11))

    r1 = Robot("r1", Position(0, 0))
    r2 = Robot("r2", Position(9, 9))

    scheduler = TaskScheduler(strategy=NearestRobotSchedulerStrategy())
    assignments = scheduler.schedule([t1], [r1, r2])

    assert len(assignments) == 1
    assert assignments[0][0] == t1
    assert assignments[0][1] == r2  # r2 at (9,9) is closest to pickup at (10,10)
