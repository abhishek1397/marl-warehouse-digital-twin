"""Cell class representing a single discrete location within the warehouse grid."""

from typing import Any, Optional

from simulator.constants import CellType
from simulator.exceptions import CellOccupiedError
from simulator.position import Position


class Cell:
    """Represents an individual spatial cell within the warehouse grid matrix."""

    def __init__(
        self,
        position: Position,
        cell_type: CellType = CellType.EMPTY,
        is_occupied: bool = False,
        occupied_by: Optional[Any] = None,
    ) -> None:
        self._position: Position = position
        self._cell_type: CellType = cell_type
        self._is_occupied: bool = is_occupied
        self._occupied_by: Optional[Any] = occupied_by

    @property
    def position(self) -> Position:
        """Returns the immutable position of this cell."""
        return self._position

    @property
    def cell_type(self) -> CellType:
        """Returns the functional cell type."""
        return self._cell_type

    @property
    def is_occupied(self) -> bool:
        """Returns True if the cell is currently occupied by an entity."""
        return self._is_occupied

    @property
    def occupied_by(self) -> Optional[Any]:
        """Returns the entity currently occupying this cell, if any."""
        return self._occupied_by

    @property
    def is_walkable(self) -> bool:
        """Returns True if the cell type is traversable and currently unoccupied."""
        return self._cell_type.is_traversable and not self._is_occupied

    def occupy(self, entity: Any) -> None:
        """Occupies the cell with an entity object.

        Raises:
            CellOccupiedError: If the cell is already occupied.
        """
        if self._is_occupied:
            raise CellOccupiedError(
                self._position.x, self._position.y, existing_entity=self._occupied_by
            )
        self._is_occupied = True
        self._occupied_by = entity

    def vacate(self) -> Optional[Any]:
        """Vacates the cell and clears entity occupation.

        Returns:
            The entity object that previously occupied the cell, if any.
        """
        former_entity = self._occupied_by
        self._is_occupied = False
        self._occupied_by = None
        return former_entity

    def set_cell_type(self, new_type: CellType) -> None:
        """Updates the functional cell type."""
        self._cell_type = new_type

    def __repr__(self) -> str:
        return (
            f"Cell(position={self._position}, cell_type={self._cell_type.name}, "
            f"is_occupied={self._is_occupied}, occupied_by={self._occupied_by})"
        )

    def __str__(self) -> str:
        occ_str = f" [Occupied by {self._occupied_by}]" if self._is_occupied else ""
        return f"Cell{self._position} ({self._cell_type.name}){occ_str}"
