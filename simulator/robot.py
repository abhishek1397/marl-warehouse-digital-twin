"""Robot class representing an Autonomous Mobile Robot (AMR) in the warehouse digital twin."""

from enum import Enum, auto
from typing import Optional

from simulator.constants import Direction
from simulator.exceptions import WarehouseError
from simulator.package import Package, PackageStatus
from simulator.position import Position
from simulator.task import Task, TaskStatus, TaskType


class RobotState(Enum):
    """Enumeration of operational states for a warehouse robot."""

    IDLE = auto()
    MOVING_TO_PICKUP = auto()
    PICKING_UP = auto()
    MOVING_TO_DROP = auto()
    DROPPING_OFF = auto()
    MOVING_TO_CHARGE = auto()
    CHARGING = auto()


class Robot:
    """Represents an Autonomous Mobile Robot (AMR) executing tasks deterministically."""

    def __init__(
        self,
        robot_id: str,
        initial_position: Position,
        max_battery: float = 100.0,
    ) -> None:
        if max_battery <= 0:
            raise WarehouseError(f"Max battery must be positive, got {max_battery}.")

        self._robot_id: str = robot_id
        self._position: Position = initial_position
        self._max_battery: float = max_battery
        self._battery_level: float = max_battery
        self._state: RobotState = RobotState.IDLE
        self._assigned_task: Optional[Task] = None
        self._carrying_package: Optional[Package] = None

        # Statistics telemetry
        self._total_distance_travelled: int = 0
        self._tasks_completed: int = 0
        self._idle_steps: int = 0
        self._charging_steps: int = 0

    @property
    def robot_id(self) -> str:
        """Returns unique robot identifier."""
        return self._robot_id

    @property
    def position(self) -> Position:
        """Returns current grid position of the robot."""
        return self._position

    @position.setter
    def position(self, new_pos: Position) -> None:
        """Directly sets robot position."""
        self._position = new_pos

    @property
    def battery_level(self) -> float:
        """Returns current battery level percentage (0.0 to max_battery)."""
        return self._battery_level

    @battery_level.setter
    def battery_level(self, val: float) -> None:
        """Sets battery level bounded between 0 and max_battery."""
        self._battery_level = max(0.0, min(self._max_battery, val))

    @property
    def max_battery(self) -> float:
        """Returns maximum battery capacity."""
        return self._max_battery

    @property
    def battery_percentage(self) -> float:
        """Returns current battery percentage (0.0 - 100.0)."""
        return (self._battery_level / self._max_battery) * 100.0

    @property
    def state(self) -> RobotState:
        """Returns current operational state."""
        return self._state

    @state.setter
    def state(self, new_state: RobotState) -> None:
        """Updates robot state."""
        self._state = new_state

    @property
    def assigned_task(self) -> Optional[Task]:
        """Returns currently assigned task, if any."""
        return self._assigned_task

    @property
    def carrying_package(self) -> Optional[Package]:
        """Returns package currently being carried by the robot, if any."""
        return self._carrying_package

    @property
    def total_distance_travelled(self) -> int:
        """Returns total distance travelled in grid steps."""
        return self._total_distance_travelled

    @property
    def tasks_completed(self) -> int:
        """Returns total number of completed tasks."""
        return self._tasks_completed

    @property
    def idle_steps(self) -> int:
        """Returns cumulative steps spent in IDLE state."""
        return self._idle_steps

    @property
    def charging_steps(self) -> int:
        """Returns cumulative steps spent in CHARGING state."""
        return self._charging_steps

    def is_idle(self) -> bool:
        """Returns True if robot is in IDLE state with no task assigned."""
        return self._state == RobotState.IDLE and self._assigned_task is None

    def is_charging(self) -> bool:
        """Returns True if robot is actively charging."""
        return self._state == RobotState.CHARGING

    def assign_task(self, task: Task) -> None:
        """Assigns a new task to the robot.

        Raises:
            WarehouseError: If robot already has an active task.
        """
        if self._assigned_task is not None and not self._assigned_task.is_terminal():
            raise WarehouseError(
                f"Robot '{self._robot_id}' already has active task '{self._assigned_task.task_id}'."
            )

        self._assigned_task = task
        task.assigned_robot_id = self._robot_id
        task.status = TaskStatus.ASSIGNED

        if task.task_type == TaskType.RECHARGE_BATTERY:
            self._state = RobotState.MOVING_TO_CHARGE
        else:
            self._state = RobotState.MOVING_TO_PICKUP

    def clear_task(self) -> Optional[Task]:
        """Clears current task and resets robot to IDLE state."""
        t = self._assigned_task
        self._assigned_task = None
        self._state = RobotState.IDLE
        return t

    def pick_up_package(self, package: Package) -> None:
        """Pick up a package at the current location.

        Raises:
            WarehouseError: If already carrying a package.
        """
        if self._carrying_package is not None:
            raise WarehouseError(
                f"Robot '{self._robot_id}' is already carrying package '{self._carrying_package.package_id}'."
            )

        self._carrying_package = package
        package.status = PackageStatus.IN_TRANSIT
        self._state = RobotState.MOVING_TO_DROP

    def drop_package(self) -> Package:
        """Drop and return the carried package.

        Raises:
            WarehouseError: If robot is not carrying a package.
        """
        if self._carrying_package is None:
            raise WarehouseError(f"Robot '{self._robot_id}' is not carrying any package to drop.")

        pkg = self._carrying_package
        self._carrying_package = None
        pkg.status = PackageStatus.DELIVERED
        return pkg

    def step_towards(self, target: Position) -> Direction:
        """Moves one step deterministically towards target position (X-axis first, then Y-axis).

        Returns:
            The Direction taken (STAY if already at target).
        """
        if self._position == target:
            return Direction.STAY

        dx = target.x - self._position.x
        dy = target.y - self._position.y

        if dx != 0:
            direction = Direction.EAST if dx > 0 else Direction.WEST
        else:
            direction = Direction.NORTH if dy < 0 else Direction.SOUTH

        self._position = self._position.get_neighbor(direction)
        self._total_distance_travelled += 1
        return direction

    def consume_battery(self, amount: float) -> None:
        """Drains battery by given amount."""
        self.battery_level -= amount

    def charge_battery(self, amount: float) -> None:
        """Restores battery by given amount."""
        self.battery_level += amount

    def increment_idle_time(self) -> None:
        """Increments idle step counter."""
        self._idle_steps += 1

    def increment_charging_time(self) -> None:
        """Increments charging step counter."""
        self._charging_steps += 1

    def increment_tasks_completed(self) -> None:
        """Increments completed tasks counter."""
        self._tasks_completed += 1

    def __repr__(self) -> str:
        return (
            f"Robot(id='{self._robot_id}', pos={self._position}, battery={self.battery_percentage:.1f}%, "
            f"state={self._state.name})"
        )

    def __str__(self) -> str:
        pkg_str = f" carrying {self._carrying_package.package_id}" if self._carrying_package else ""
        return f"Robot '{self._robot_id}' at {self._position} [{self._state.name}]{pkg_str} Battery={self.battery_percentage:.1f}%"
