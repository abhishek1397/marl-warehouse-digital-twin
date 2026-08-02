"""Unit tests for Warehouse class."""

import pytest

from simulator.constants import CellType
from simulator.exceptions import (
    CellOccupiedError,
    InvalidPlacementError,
    OutOfBoundsError,
    WarehouseConfigurationError,
)
from simulator.position import Position
from simulator.warehouse import Warehouse


def test_warehouse_initialization() -> None:
    """Test warehouse instantiation and attributes."""
    wh = Warehouse(width=20, height=30, name="TestWarehouse")
    assert wh.name == "TestWarehouse"
    assert wh.width == 20
    assert wh.height == 30
    assert repr(wh) == "Warehouse(name='TestWarehouse', dimensions=20x30)"


def test_warehouse_invalid_name() -> None:
    """Test warehouse instantiation with invalid name."""
    with pytest.raises(WarehouseConfigurationError):
        Warehouse(name="")  # type: ignore[arg-type]


def test_validate_placement() -> None:
    """Test placement validation logic."""
    wh = Warehouse(width=10, height=10)
    pos = Position(5, 5)

    assert wh.validate_placement(pos) is True

    # Place an object
    wh.place_object(pos, "SampleEntity")
    assert wh.validate_placement(pos) is False

    # Out of bounds position
    assert wh.validate_placement(Position(15, 15)) is False


def test_validate_placement_require_walkable() -> None:
    """Test placement validation when requiring traversable cell types."""
    wh = Warehouse(width=10, height=10)
    pos = Position(3, 3)
    wh.set_cell_type(pos, CellType.OBSTACLE)

    assert wh.validate_placement(pos, require_walkable=False) is True
    assert wh.validate_placement(pos, require_walkable=True) is False


def test_place_and_query_object() -> None:
    """Test placing and querying objects in warehouse cells."""
    wh = Warehouse(width=10, height=10)
    pos = Position(2, 4)
    dummy_item = {"id": "Item_100"}

    cell = wh.place_object(pos, dummy_item, cell_type=CellType.SHELF)
    assert cell.is_occupied is True
    assert cell.cell_type == CellType.SHELF
    assert wh.query_object(pos) == dummy_item


def test_place_object_none_raises_error() -> None:
    """Test placing None raises InvalidPlacementError."""
    wh = Warehouse(width=10, height=10)
    with pytest.raises(InvalidPlacementError):
        wh.place_object(Position(1, 1), None)


def test_remove_object() -> None:
    """Test removing an object from a warehouse cell."""
    wh = Warehouse(width=10, height=10)
    pos = Position(4, 4)
    wh.place_object(pos, "Robot_X")

    removed = wh.remove_object(pos)
    assert removed == "Robot_X"
    assert wh.query_object(pos) is None
    assert wh.get_cell(pos).is_occupied is False


def test_remove_object_unoccupied_raises_error() -> None:
    """Test removing an object from an empty cell raises InvalidPlacementError."""
    wh = Warehouse(width=10, height=10)
    with pytest.raises(InvalidPlacementError):
        wh.remove_object(Position(0, 0))


def test_get_cells_by_type() -> None:
    """Test retrieving all cells matching a specified CellType."""
    wh = Warehouse(width=5, height=5)
    wh.set_cell_type(Position(0, 0), CellType.PICKUP_ZONE)
    wh.set_cell_type(Position(4, 4), CellType.PICKUP_ZONE)

    pickups = wh.get_cells_by_type(CellType.PICKUP_ZONE)
    assert len(pickups) == 2
    pickup_positions = [c.position for c in pickups]
    assert Position(0, 0) in pickup_positions
    assert Position(4, 4) in pickup_positions


def test_get_neighbors_delegation() -> None:
    """Test neighbor query delegation from warehouse to grid."""
    wh = Warehouse(width=5, height=5)
    neighbors = wh.get_neighbors(Position(2, 2))
    assert len(neighbors) == 4
