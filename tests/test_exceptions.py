"""Unit tests for simulator custom exception classes."""

import pytest

from simulator.exceptions import (
    CellOccupiedError,
    InvalidPlacementError,
    OutOfBoundsError,
    WarehouseConfigurationError,
    WarehouseError,
)


def test_warehouse_error_base() -> None:
    """Test base exception class inheritance."""
    err = WarehouseError("General simulation failure")
    assert isinstance(err, Exception)
    assert str(err) == "General simulation failure"


def test_out_of_bounds_error() -> None:
    """Test OutOfBoundsError formatting and attributes."""
    err = OutOfBoundsError(x=10, y=15, width=10, height=10)
    assert err.x == 10
    assert err.y == 15
    assert err.width == 10
    assert err.height == 10
    assert "Coordinates (10, 15) are out of bounds for grid size 10x10" in str(err)
    assert isinstance(err, WarehouseError)


def test_cell_occupied_error() -> None:
    """Test CellOccupiedError formatting with and without existing entity."""
    err1 = CellOccupiedError(x=2, y=3)
    assert err1.x == 2
    assert err1.y == 3
    assert "Cell at (2, 3) is already occupied" in str(err1)

    err2 = CellOccupiedError(x=4, y=5, existing_entity="Robot_A")
    assert "Occupying entity: Robot_A" in str(err2)


def test_invalid_placement_error() -> None:
    """Test InvalidPlacementError exception behavior."""
    err = InvalidPlacementError("Invalid location for shelf")
    assert isinstance(err, WarehouseError)
    assert "Invalid location for shelf" in str(err)


def test_warehouse_configuration_error() -> None:
    """Test WarehouseConfigurationError exception behavior."""
    err = WarehouseConfigurationError("Negative grid size")
    assert isinstance(err, WarehouseError)
    assert "Negative grid size" in str(err)
