"""Grid class managing the 2D spatial cell array of the warehouse digital twin."""

from typing import Iterator, List, Optional

from simulator.cell import Cell
from simulator.constants import (
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_WIDTH,
    MAX_GRID_DIMENSION,
    MIN_GRID_DIMENSION,
    CellType,
)
from simulator.exceptions import OutOfBoundsError, WarehouseConfigurationError
from simulator.position import Position


class Grid:
    """Manages a 2D matrix of Cell instances representing the warehouse floor."""

    def __init__(
        self,
        width: int = DEFAULT_GRID_WIDTH,
        height: int = DEFAULT_GRID_HEIGHT,
        default_cell_type: CellType = CellType.EMPTY,
    ) -> None:
        if not (MIN_GRID_DIMENSION <= width <= MAX_GRID_DIMENSION):
            raise WarehouseConfigurationError(
                f"Width {width} must be between {MIN_GRID_DIMENSION} and {MAX_GRID_DIMENSION}."
            )
        if not (MIN_GRID_DIMENSION <= height <= MAX_GRID_DIMENSION):
            raise WarehouseConfigurationError(
                f"Height {height} must be between {MIN_GRID_DIMENSION} and {MAX_GRID_DIMENSION}."
            )

        self._width: int = width
        self._height: int = height
        self._cells: List[List[Cell]] = [
            [
                Cell(position=Position(x=x, y=y), cell_type=default_cell_type)
                for x in range(width)
            ]
            for y in range(height)
        ]

    @property
    def width(self) -> int:
        """Returns the grid width in cells."""
        return self._width

    @property
    def height(self) -> int:
        """Returns the grid height in cells."""
        return self._height

    def is_in_bounds(self, position: Position) -> bool:
        """Checks if a position lies within grid boundaries."""
        return 0 <= position.x < self._width and 0 <= position.y < self._height

    def is_in_bounds_coords(self, x: int, y: int) -> bool:
        """Checks if raw integer coordinates (x, y) lie within grid boundaries."""
        return 0 <= x < self._width and 0 <= y < self._height

    def get_cell(self, position: Position) -> Cell:
        """Retrieves the Cell object at the given Position.

        Raises:
            OutOfBoundsError: If coordinates are outside grid boundaries.
        """
        if not self.is_in_bounds(position):
            raise OutOfBoundsError(position.x, position.y, self._width, self._height)
        return self._cells[position.y][position.x]

    def get_cell_by_coords(self, x: int, y: int) -> Cell:
        """Retrieves the Cell object at raw integer coordinates (x, y).

        Raises:
            OutOfBoundsError: If coordinates are outside grid boundaries.
        """
        return self.get_cell(Position(x, y))

    def get_neighbors(
        self, position: Position, include_diagonals: bool = False
    ) -> List[Cell]:
        """Returns all adjacent in-bounds Cell objects around a position."""
        if not self.is_in_bounds(position):
            raise OutOfBoundsError(position.x, position.y, self._width, self._height)

        neighbor_positions = position.get_all_neighbors(include_diagonals=include_diagonals)
        return [
            self.get_cell(pos)
            for pos in neighbor_positions
            if self.is_in_bounds(pos)
        ]

    def get_walkable_neighbors(
        self, position: Position, include_diagonals: bool = False
    ) -> List[Cell]:
        """Returns all adjacent in-bounds Cell objects that are walkable."""
        return [
            cell
            for cell in self.get_neighbors(position, include_diagonals=include_diagonals)
            if cell.is_walkable
        ]

    def is_cell_occupied(self, position: Position) -> bool:
        """Checks if the cell at the given position is occupied.

        Raises:
            OutOfBoundsError: If position is outside grid boundaries.
        """
        return self.get_cell(position).is_occupied

    def set_cell_type(self, position: Position, cell_type: CellType) -> None:
        """Updates cell type at the specified position.

        Raises:
            OutOfBoundsError: If position is outside grid boundaries.
        """
        self.get_cell(position).set_cell_type(cell_type)

    def __iter__(self) -> Iterator[Cell]:
        """Iterates through all cells in row-major order."""
        for row in self._cells:
            for cell in row:
                yield cell

    def __len__(self) -> int:
        """Returns total number of cells in the grid."""
        return self._width * self._height

    def __repr__(self) -> str:
        return f"Grid(width={self._width}, height={self._height})"
