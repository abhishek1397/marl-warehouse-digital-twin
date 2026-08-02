"""Shelf class representing inventory storage racks within the warehouse grid."""

from typing import Dict, List, Optional

from simulator.exceptions import InvalidPlacementError, WarehouseError
from simulator.position import Position


class Shelf:
    """Represents a physical storage shelf capable of holding packages up to its capacity."""

    def __init__(
        self,
        shelf_id: str,
        position: Position,
        capacity: int = 10,
    ) -> None:
        if capacity <= 0:
            raise InvalidPlacementError(f"Shelf capacity must be positive, got {capacity}.")

        self._shelf_id: str = shelf_id
        self._position: Position = position
        self._capacity: int = capacity
        self._packages: Dict[str, "Package"] = {}

    @property
    def shelf_id(self) -> str:
        """Returns the unique identifier of the shelf."""
        return self._shelf_id

    @property
    def position(self) -> Position:
        """Returns the spatial position of the shelf."""
        return self._position

    @property
    def capacity(self) -> int:
        """Returns the maximum package storage capacity of the shelf."""
        return self._capacity

    @property
    def current_load(self) -> int:
        """Returns the number of packages currently stored on the shelf."""
        return len(self._packages)

    @property
    def packages(self) -> List["Package"]:
        """Returns a list of all packages stored on this shelf."""
        return list(self._packages.values())

    def is_full(self) -> bool:
        """Returns True if the shelf has reached maximum package capacity."""
        return len(self._packages) >= self._capacity

    def is_empty(self) -> bool:
        """Returns True if no packages are stored on this shelf."""
        return len(self._packages) == 0

    def add_package(self, package: "Package") -> None:
        """Stores a package on the shelf.

        Raises:
            WarehouseError: If shelf is full or package is already stored.
        """
        if self.is_full():
            raise WarehouseError(f"Shelf '{self._shelf_id}' is full (capacity {self._capacity}).")
        if package.package_id in self._packages:
            raise WarehouseError(
                f"Package '{package.package_id}' is already stored on shelf '{self._shelf_id}'."
            )

        self._packages[package.package_id] = package
        package.current_shelf_id = self._shelf_id

    def remove_package(self, package_id: str) -> "Package":
        """Removes and returns a package from the shelf by ID.

        Raises:
            WarehouseError: If package is not stored on this shelf.
        """
        if package_id not in self._packages:
            raise WarehouseError(
                f"Package '{package_id}' not found on shelf '{self._shelf_id}'."
            )

        pkg = self._packages.pop(package_id)
        pkg.current_shelf_id = None
        return pkg

    def get_package(self, package_id: str) -> Optional["Package"]:
        """Retrieves a package reference by ID without removing it."""
        return self._packages.get(package_id)

    def __repr__(self) -> str:
        return (
            f"Shelf(id='{self._shelf_id}', position={self._position}, "
            f"load={self.current_load}/{self._capacity})"
        )

    def __str__(self) -> str:
        return f"Shelf '{self._shelf_id}' at {self._position} [{self.current_load}/{self._capacity} packages]"
