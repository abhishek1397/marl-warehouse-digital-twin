"""ChargingStation class managing battery replenishment hubs for warehouse robots."""

from typing import List, Set

from simulator.exceptions import WarehouseError
from simulator.position import Position


class ChargingStation:
    """Represents a dedicated charging dock where robots can replenish battery power."""

    def __init__(
        self,
        station_id: str,
        position: Position,
        charge_rate: float = 10.0,
        capacity: int = 1,
    ) -> None:
        if charge_rate <= 0:
            raise WarehouseError(f"Charging rate must be positive, got {charge_rate}.")
        if capacity <= 0:
            raise WarehouseError(f"Charging station capacity must be positive, got {capacity}.")

        self._station_id: str = station_id
        self._position: Position = position
        self._charge_rate: float = charge_rate
        self._capacity: int = capacity
        self._docked_robot_ids: Set[str] = set()

    @property
    def station_id(self) -> str:
        """Returns unique identifier of the charging station."""
        return self._station_id

    @property
    def position(self) -> Position:
        """Returns spatial position of the charging station."""
        return self._position

    @property
    def charge_rate(self) -> float:
        """Returns battery percentage restored per step."""
        return self._charge_rate

    @property
    def capacity(self) -> int:
        """Returns maximum simultaneous docked robots."""
        return self._capacity

    @property
    def current_occupancy(self) -> int:
        """Returns number of robots currently docked."""
        return len(self._docked_robot_ids)

    @property
    def docked_robot_ids(self) -> List[str]:
        """Returns list of IDs of currently docked robots."""
        return list(self._docked_robot_ids)

    def is_available(self) -> bool:
        """Returns True if the charging station has open docking capacity."""
        return len(self._docked_robot_ids) < self._capacity

    def is_docked(self, robot_id: str) -> bool:
        """Returns True if the given robot is currently docked at this station."""
        return robot_id in self._docked_robot_ids

    def dock_robot(self, robot_id: str) -> None:
        """Docks a robot at this charging station.

        Raises:
            WarehouseError: If capacity is full or robot is already docked.
        """
        if not self.is_available():
            raise WarehouseError(
                f"ChargingStation '{self._station_id}' is full ({self.current_occupancy}/{self._capacity})."
            )
        if robot_id in self._docked_robot_ids:
            raise WarehouseError(
                f"Robot '{robot_id}' is already docked at station '{self._station_id}'."
            )

        self._docked_robot_ids.add(robot_id)

    def undock_robot(self, robot_id: str) -> None:
        """Undocks a robot from this charging station.

        Raises:
            WarehouseError: If robot is not currently docked at this station.
        """
        if robot_id not in self._docked_robot_ids:
            raise WarehouseError(
                f"Robot '{robot_id}' is not docked at station '{self._station_id}'."
            )

        self._docked_robot_ids.remove(robot_id)

    def __repr__(self) -> str:
        return (
            f"ChargingStation(id='{self._station_id}', position={self._position}, "
            f"occupancy={self.current_occupancy}/{self._capacity})"
        )

    def __str__(self) -> str:
        return f"ChargingStation '{self._station_id}' at {self._position} [{self.current_occupancy}/{self._capacity} docked]"
