"""Task class and enums representing work units assigned to robots."""

from enum import Enum, auto
from typing import Optional

from simulator.package import Package
from simulator.position import Position


class TaskType(Enum):
    """Enumeration of warehouse task types."""

    PICKUP_AND_DELIVER = auto()
    RECHARGE_BATTERY = auto()
    IDLE_PARK = auto()


class TaskStatus(Enum):
    """Enumeration of task lifecycle states."""

    CREATED = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    """Enumeration of task execution priorities."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: "TaskPriority") -> bool:
        if isinstance(other, TaskPriority):
            return self.value < other.value
        return NotImplemented


class Task:
    """Represents a work directive assigned to a robot within the warehouse."""

    def __init__(
        self,
        task_id: str,
        task_type: TaskType,
        pickup_position: Position,
        drop_position: Position,
        package: Optional[Package] = None,
        assigned_robot_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        status: TaskStatus = TaskStatus.CREATED,
        created_at_step: int = 0,
    ) -> None:
        self._task_id: str = task_id
        self._task_type: TaskType = task_type
        self._pickup_position: Position = pickup_position
        self._drop_position: Position = drop_position
        self._package: Optional[Package] = package
        self._assigned_robot_id: Optional[str] = assigned_robot_id
        self._priority: TaskPriority = priority
        self._status: TaskStatus = status
        self._created_at_step: int = created_at_step
        self._completed_at_step: Optional[int] = None

    @property
    def task_id(self) -> str:
        """Returns unique task identifier."""
        return self._task_id

    @property
    def task_type(self) -> TaskType:
        """Returns functional task type."""
        return self._task_type

    @property
    def pickup_position(self) -> Position:
        """Returns pickup origin position."""
        return self._pickup_position

    @property
    def drop_position(self) -> Position:
        """Returns destination drop position."""
        return self._drop_position

    @property
    def package(self) -> Optional[Package]:
        """Returns associated package object, if any."""
        return self._package

    @property
    def assigned_robot_id(self) -> Optional[str]:
        """Returns ID of assigned robot, if any."""
        return self._assigned_robot_id

    @assigned_robot_id.setter
    def assigned_robot_id(self, robot_id: Optional[str]) -> None:
        """Assigns task to a robot."""
        self._assigned_robot_id = robot_id

    @property
    def priority(self) -> TaskPriority:
        """Returns task priority."""
        return self._priority

    @property
    def status(self) -> TaskStatus:
        """Returns current task status."""
        return self._status

    @status.setter
    def status(self, new_status: TaskStatus) -> None:
        """Updates task status."""
        self._status = new_status

    @property
    def created_at_step(self) -> int:
        """Returns simulation step when task was created."""
        return self._created_at_step

    @property
    def completed_at_step(self) -> Optional[int]:
        """Returns simulation step when task was completed."""
        return self._completed_at_step

    @completed_at_step.setter
    def completed_at_step(self, step: int) -> None:
        """Sets completion step timestamp."""
        self._completed_at_step = step

    def is_terminal(self) -> bool:
        """Returns True if task is in a terminal state (COMPLETED or CANCELLED)."""
        return self._status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}

    def __lt__(self, other: "Task") -> bool:
        """Enables sorting tasks by priority (higher priority first)."""
        if not isinstance(other, Task):
            return NotImplemented
        return self._priority.value > other._priority.value

    def __repr__(self) -> str:
        return (
            f"Task(id='{self._task_id}', type={self._task_type.name}, "
            f"priority={self._priority.name}, status={self._status.name})"
        )

    def __str__(self) -> str:
        return f"Task '{self._task_id}' ({self._task_type.name}) Priority={self._priority.name} Status={self._status.name}"
