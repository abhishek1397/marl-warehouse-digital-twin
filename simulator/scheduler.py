"""TaskScheduler and Strategy pattern implementations for warehouse task allocation."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from simulator.position import Position
from simulator.robot import Robot
from simulator.task import Task, TaskPriority


class TaskSchedulerStrategy(ABC):
    """Abstract base class defining the strategy interface for task scheduling algorithms."""

    @abstractmethod
    def select_next_assignment(
        self, unassigned_tasks: List[Task], idle_robots: List[Robot]
    ) -> Optional[Tuple[Task, Robot]]:
        """Selects the next (Task, Robot) pair to assign.

        Returns:
            Tuple of (Task, Robot) or None if no valid assignment can be made.
        """
        pass


class FIFOSchedulerStrategy(TaskSchedulerStrategy):
    """First-In, First-Out (FIFO) Task Scheduler strategy.

    Assigns the oldest pending task to the first available idle robot.
    """

    def select_next_assignment(
        self, unassigned_tasks: List[Task], idle_robots: List[Robot]
    ) -> Optional[Tuple[Task, Robot]]:
        if not unassigned_tasks or not idle_robots:
            return None
        # Tasks are ordered by creation step, so first in list is oldest
        return (unassigned_tasks[0], idle_robots[0])


class PrioritySchedulerStrategy(TaskSchedulerStrategy):
    """Priority-Based Task Scheduler strategy.

    Assigns the task with the highest TaskPriority to the first available idle robot.
    """

    def select_next_assignment(
        self, unassigned_tasks: List[Task], idle_robots: List[Robot]
    ) -> Optional[Tuple[Task, Robot]]:
        if not unassigned_tasks or not idle_robots:
            return None
        # Highest priority task
        sorted_tasks = sorted(unassigned_tasks, key=lambda t: t.priority, reverse=True)
        return (sorted_tasks[0], idle_robots[0])


class NearestRobotSchedulerStrategy(TaskSchedulerStrategy):
    """Nearest-Robot Task Scheduler strategy.

    Pairs tasks with the closest available idle robot based on Manhattan distance to pickup location.
    """

    def select_next_assignment(
        self, unassigned_tasks: List[Task], idle_robots: List[Robot]
    ) -> Optional[Tuple[Task, Robot]]:
        if not unassigned_tasks or not idle_robots:
            return None

        best_pair: Optional[Tuple[Task, Robot]] = None
        min_distance: float = float("inf")

        for task in unassigned_tasks:
            for robot in idle_robots:
                dist = robot.position.manhattan_distance(task.pickup_position)
                if dist < min_distance:
                    min_distance = dist
                    best_pair = (task, robot)

        return best_pair


class TaskScheduler:
    """Context class managing task scheduling execution via interchangeable strategies."""

    def __init__(self, strategy: Optional[TaskSchedulerStrategy] = None) -> None:
        self.strategy: TaskSchedulerStrategy = strategy or PrioritySchedulerStrategy()

    def set_strategy(self, strategy: TaskSchedulerStrategy) -> None:
        """Dynamically updates the active task scheduling strategy."""
        self.strategy = strategy

    def schedule(
        self, unassigned_tasks: List[Task], idle_robots: List[Robot]
    ) -> List[Tuple[Task, Robot]]:
        """Schedules as many tasks as possible using the active strategy.

        Returns:
            List of assigned (Task, Robot) pairs.
        """
        assignments: List[Tuple[Task, Robot]] = []
        pending_tasks = list(unassigned_tasks)
        available_robots = list(idle_robots)

        while pending_tasks and available_robots:
            pair = self.strategy.select_next_assignment(pending_tasks, available_robots)
            if pair is None:
                break
            task, robot = pair
            assignments.append((task, robot))
            pending_tasks.remove(task)
            available_robots.remove(robot)

        return assignments
