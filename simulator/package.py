"""Package class and PackageStatus enum representing items stored and transported in the warehouse."""

from enum import Enum, auto
from typing import Optional

from simulator.position import Position


class PackageStatus(Enum):
    """Enumeration of lifecycle statuses for a package."""

    UNASSIGNED = auto()
    ASSIGNED = auto()
    IN_TRANSIT = auto()
    DELIVERED = auto()


class Package:
    """Represents a physical package that can be stored on shelves and transported by robots."""

    def __init__(
        self,
        package_id: str,
        source_position: Position,
        destination_position: Position,
        weight: float = 1.0,
        current_shelf_id: Optional[str] = None,
        status: PackageStatus = PackageStatus.UNASSIGNED,
    ) -> None:
        self._package_id: str = package_id
        self._source_position: Position = source_position
        self._destination_position: Position = destination_position
        self._weight: float = weight
        self.current_shelf_id: Optional[str] = current_shelf_id
        self._status: PackageStatus = status

    @property
    def package_id(self) -> str:
        """Returns the unique identifier of the package."""
        return self._package_id

    @property
    def source_position(self) -> Position:
        """Returns the origin pickup position of the package."""
        return self._source_position

    @property
    def destination_position(self) -> Position:
        """Returns the drop-off destination position of the package."""
        return self._destination_position

    @property
    def weight(self) -> float:
        """Returns package weight in kg."""
        return self._weight

    @property
    def status(self) -> PackageStatus:
        """Returns current package lifecycle status."""
        return self._status

    @status.setter
    def status(self, new_status: PackageStatus) -> None:
        """Updates package status."""
        self._status = new_status

    def is_delivered(self) -> bool:
        """Returns True if the package has reached its destination and is marked DELIVERED."""
        return self._status == PackageStatus.DELIVERED

    def __repr__(self) -> str:
        return (
            f"Package(id='{self._package_id}', src={self._source_position}, "
            f"dst={self._destination_position}, status={self._status.name})"
        )

    def __str__(self) -> str:
        return f"Package '{self._package_id}' [{self._status.name}] from {self._source_position} to {self._destination_position}"
