"""TaskManager class handling task lifecycle, priority queueing, and robot assignment."""

import heapq
from typing import Dict, List, Optional

from simulator.exceptions import WarehouseError
from simulator.package import Package, PackageStatus
from simulator.position import Position
from simulator.robot import Robot, RobotState
from simulator.task import Task, TaskPriority, TaskStatus, TaskType


class TaskManager:
    """Manages creation, prioritization, allocation, and tracking of warehouse tasks."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._unassigned_queue: List[Task] = []

    @property
    def total_tasks(self) -> int:
        """Returns total number of tasks created."""
        return len(self._tasks)

    @property
    def unassigned_count(self) -> int:
        """Returns count of tasks currently waiting in queue."""
        return len(self._unassigned_queue)

    def create_task(
        self,
        task_id: str,
        task_type: TaskType,
        pickup_position: Position,
        drop_position: Position,
        package: Optional[Package] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        created_at_step: int = 0,
    ) -> Task:
        """Creates a new task and adds it to the unassigned task queue.

        Raises:
            WarehouseError: If task_id already exists.
        """
        if task_id in self._tasks:
            raise WarehouseError(f"Task with ID '{task_id}' already exists.")

        task = Task(
            task_id=task_id,
            task_type=task_type,
            pickup_position=pickup_position,
            drop_position=drop_position,
            package=package,
            priority=priority,
            created_at_step=created_at_step,
        )

        if package is not None:
            package.status = PackageStatus.ASSIGNED

        self._tasks[task_id] = task
        self._unassigned_queue.append(task)
        self._unassigned_queue.sort()  # Highest priority first
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieves a task by ID."""
        return self._tasks.get(task_id)

    def assign_next_task(self, robot: Robot) -> Optional[Task]:
        """Assigns the highest priority pending task to an idle robot.

        Returns:
            The assigned Task, or None if no pending tasks exist.
        """
        if not robot.is_idle() and robot.state != RobotState.IDLE:
            return None

        if not self._unassigned_queue:
            return None

        task = self._unassigned_queue.pop(0)
        robot.assign_task(task)
        return task

    def complete_task(self, task_id: str, step: int = 0) -> Task:
        """Marks a task as completed and updates associated robot and package.

        Raises:
            WarehouseError: If task_id does not exist.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise WarehouseError(f"Task '{task_id}' not found.")

        task.status = TaskStatus.COMPLETED
        task.completed_at_step = step

        if task.package is not None:
            task.package.status = PackageStatus.DELIVERED

        return task

    def cancel_task(self, task_id: str) -> Task:
        """Cancels a task and removes it from unassigned queue if present.

        Raises:
            WarehouseError: If task_id does not exist.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise WarehouseError(f"Task '{task_id}' not found.")

        task.status = TaskStatus.CANCELLED
        if task in self._unassigned_queue:
            self._unassigned_queue.remove(task)

        return task

    def get_pending_tasks(self) -> List[Task]:
        """Returns list of all unassigned pending tasks."""
        return list(self._unassigned_queue)

    def get_completed_tasks(self) -> List[Task]:
        """Returns list of all completed tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.COMPLETED]

    def all_tasks_completed(self) -> bool:
        """Returns True if every created task has reached a terminal status (COMPLETED or CANCELLED)."""
        if not self._tasks:
            return True
        return all(t.is_terminal() for t in self._tasks.values())
