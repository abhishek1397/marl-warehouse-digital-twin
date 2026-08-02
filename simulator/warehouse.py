"""Warehouse class serving as the top-level container for the simulation world."""

from typing import Any, List, Optional

from simulator.cell import Cell
from simulator.constants import (
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_WIDTH,
    CellType,
)
from simulator.exceptions import (
    CellOccupiedError,
    InvalidPlacementError,
    OutOfBoundsError,
    WarehouseConfigurationError,
)
from simulator.grid import Grid
from simulator.position import Position


class Warehouse:
    """Encapsulates the warehouse digital twin world, managing grid cells and placed entities."""

    def __init__(
        self,
        width: int = DEFAULT_GRID_WIDTH,
        height: int = DEFAULT_GRID_HEIGHT,
        name: str = "DigitalTwinWarehouse",
    ) -> None:
        if not name or not isinstance(name, str):
            raise WarehouseConfigurationError("Warehouse name must be a non-empty string.")

        self._name: str = name
        self._grid: Grid = Grid(width=width, height=height)

    @property
    def name(self) -> str:
        """Returns the name identifier of the warehouse."""
        return self._name

    @property
    def grid(self) -> Grid:
        """Returns the internal grid instance."""
        return self._grid

    @property
    def width(self) -> int:
        """Returns warehouse grid width."""
        return self._grid.width

    @property
    def height(self) -> int:
        """Returns warehouse grid height."""
        return self._grid.height

    def is_in_bounds(self, position: Position) -> bool:
        """Delegates boundary checking to the internal grid."""
        return self._grid.is_in_bounds(position)

    def validate_placement(
        self, position: Position, require_walkable: bool = False
    ) -> bool:
        """Validates whether an entity or structure can be placed at a position.

        Returns True if the position is within bounds and not currently occupied.
        If require_walkable is True, also checks that cell_type is traversable.
        """
        if not self._grid.is_in_bounds(position):
            return False

        cell = self._grid.get_cell(position)
        if cell.is_occupied:
            return False

        if require_walkable and not cell.cell_type.is_traversable:
            return False

        return True

    def place_object(
        self,
        position: Position,
        obj: Any,
        cell_type: Optional[CellType] = None,
    ) -> Cell:
        """Places an object into a specified warehouse grid cell.

        Args:
            position: Target grid position.
            obj: Entity or object to place in the cell.
            cell_type: Optional CellType update for the cell.

        Returns:
            The modified Cell object.

        Raises:
            OutOfBoundsError: If position is outside grid boundaries.
            CellOccupiedError: If the target cell is already occupied.
            InvalidPlacementError: If object is None.
        """
        if obj is None:
            raise InvalidPlacementError("Cannot place None object into warehouse cell.")

        cell = self._grid.get_cell(position)
        cell.occupy(obj)

        if cell_type is not None:
            cell.set_cell_type(cell_type)

        return cell

    def remove_object(self, position: Position) -> Any:
        """Removes and returns an object from a warehouse grid cell.

        Raises:
            OutOfBoundsError: If position is outside grid boundaries.
            InvalidPlacementError: If the target cell is not occupied.
        """
        cell = self._grid.get_cell(position)
        if not cell.is_occupied:
            raise InvalidPlacementError(
                f"Cannot remove object: cell at {position} is not occupied."
            )
        return cell.vacate()

    def query_object(self, position: Position) -> Optional[Any]:
        """Queries the entity occupying a position, returning None if unoccupied.

        Raises:
            OutOfBoundsError: If position is outside grid boundaries.
        """
        return self._grid.get_cell(position).occupied_by

    def get_cell(self, position: Position) -> Cell:
        """Retrieves cell at the given position."""
        return self._grid.get_cell(position)

    def get_neighbors(
        self, position: Position, include_diagonals: bool = False
    ) -> List[Cell]:
        """Retrieves adjacent cells surrounding a position."""
        return self._grid.get_neighbors(position, include_diagonals=include_diagonals)

    def get_cells_by_type(self, cell_type: CellType) -> List[Cell]:
        """Returns all cells matching a specific CellType."""
        return [cell for cell in self._grid if cell.cell_type == cell_type]

    def set_cell_type(self, position: Position, cell_type: CellType) -> None:
        """Sets the functional cell type at a given position."""
        self._grid.set_cell_type(position, cell_type)

    def __repr__(self) -> str:
        return f"Warehouse(name='{self._name}', dimensions={self.width}x{self.height})"
