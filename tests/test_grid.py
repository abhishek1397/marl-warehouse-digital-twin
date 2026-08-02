"""Unit tests for Grid class."""

import pytest

from simulator.constants import CellType
from simulator.exceptions import OutOfBoundsError, WarehouseConfigurationError
from simulator.grid import Grid
from simulator.position import Position


def test_grid_initialization_valid() -> None:
    """Test valid grid creation and dimensions."""
    grid = Grid(width=10, height=20, default_cell_type=CellType.EMPTY)
    assert grid.width == 10
    assert grid.height == 20
    assert len(grid) == 200
    assert repr(grid) == "Grid(width=10, height=20)"


def test_grid_initialization_invalid_dimensions() -> None:
    """Test grid creation with invalid dimensions raises error."""
    with pytest.raises(WarehouseConfigurationError):
        Grid(width=1, height=10)

    with pytest.raises(WarehouseConfigurationError):
        Grid(width=10, height=10000)


def test_bounds_checking() -> None:
    """Test coordinate boundary checking methods."""
    grid = Grid(width=10, height=10)

    assert grid.is_in_bounds(Position(0, 0)) is True
    assert grid.is_in_bounds(Position(9, 9)) is True
    assert grid.is_in_bounds(Position(10, 5)) is False
    assert grid.is_in_bounds(Position(-1, 5)) is False

    assert grid.is_in_bounds_coords(5, 5) is True
    assert grid.is_in_bounds_coords(5, 10) is False


def test_get_cell_valid_and_out_of_bounds() -> None:
    """Test getting cell by Position and raw coordinates."""
    grid = Grid(width=5, height=5)

    cell = grid.get_cell(Position(2, 3))
    assert cell.position == Position(2, 3)

    cell_coords = grid.get_cell_by_coords(1, 4)
    assert cell_coords.position == Position(1, 4)

    with pytest.raises(OutOfBoundsError):
        grid.get_cell(Position(5, 5))


def test_get_neighbors() -> None:
    """Test cardinal and diagonal neighbor lookups at corners and center."""
    grid = Grid(width=5, height=5)

    # Top-left corner (0, 0)
    top_left_neighbors = grid.get_neighbors(Position(0, 0))
    assert len(top_left_neighbors) == 2
    neighbor_positions = [c.position for c in top_left_neighbors]
    assert Position(1, 0) in neighbor_positions
    assert Position(0, 1) in neighbor_positions

    # Center position (2, 2)
    center_neighbors = grid.get_neighbors(Position(2, 2))
    assert len(center_neighbors) == 4

    # Center position with diagonals (2, 2)
    diag_neighbors = grid.get_neighbors(Position(2, 2), include_diagonals=True)
    assert len(diag_neighbors) == 8


def test_get_walkable_neighbors() -> None:
    """Test retrieving only walkable neighbors."""
    grid = Grid(width=5, height=5)
    # Block Position(2, 1) with an obstacle
    grid.set_cell_type(Position(2, 1), CellType.OBSTACLE)

    walkable = grid.get_walkable_neighbors(Position(2, 2))
    assert len(walkable) == 3
    walkable_positions = [c.position for c in walkable]
    assert Position(2, 1) not in walkable_positions


def test_grid_iteration() -> None:
    """Test iterating over grid cells."""
    grid = Grid(width=3, height=3)
    cells = list(grid)
    assert len(cells) == 9
    assert cells[0].position == Position(0, 0)
    assert cells[-1].position == Position(2, 2)
