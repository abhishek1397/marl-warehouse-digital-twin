"""Unit tests for Cell class."""

import pytest

from simulator.cell import Cell
from simulator.constants import CellType
from simulator.exceptions import CellOccupiedError
from simulator.position import Position


def test_cell_initialization_defaults() -> None:
    """Test Cell initialization with default values."""
    pos = Position(1, 1)
    cell = Cell(position=pos)

    assert cell.position == pos
    assert cell.cell_type == CellType.EMPTY
    assert cell.is_occupied is False
    assert cell.occupied_by is None
    assert cell.is_walkable is True


def test_cell_occupy_and_vacate() -> None:
    """Test occupying and vacating a cell."""
    cell = Cell(position=Position(0, 0))
    dummy_robot = "Robot_1"

    cell.occupy(dummy_robot)
    assert cell.is_occupied is True
    assert cell.occupied_by == dummy_robot
    assert cell.is_walkable is False

    former = cell.vacate()
    assert former == dummy_robot
    assert cell.is_occupied is False
    assert cell.occupied_by is None
    assert cell.is_walkable is True


def test_cell_occupy_raises_cell_occupied_error() -> None:
    """Test that occupying an occupied cell raises CellOccupiedError."""
    cell = Cell(position=Position(2, 2))
    cell.occupy("Obj1")

    with pytest.raises(CellOccupiedError) as exc_info:
        cell.occupy("Obj2")

    assert exc_info.value.x == 2
    assert exc_info.value.y == 2
    assert exc_info.value.existing_entity == "Obj1"


def test_cell_type_modification() -> None:
    """Test updating cell type and its effect on walkability."""
    cell = Cell(position=Position(3, 3), cell_type=CellType.EMPTY)
    assert cell.is_walkable is True

    cell.set_cell_type(CellType.OBSTACLE)
    assert cell.cell_type == CellType.OBSTACLE
    assert cell.is_walkable is False


def test_cell_string_representations() -> None:
    """Test __repr__ and __str__ output formatting."""
    cell = Cell(position=Position(1, 2), cell_type=CellType.SHELF)
    assert "Cell(position=(1, 2), cell_type=SHELF" in repr(cell)
    assert "Cell(1, 2) (SHELF)" in str(cell)

    cell.occupy("Shelf_A")
    assert "[Occupied by Shelf_A]" in str(cell)

