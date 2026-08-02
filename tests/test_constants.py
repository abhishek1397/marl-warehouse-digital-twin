"""Unit tests for constants and enumerations."""

import pytest

from simulator.constants import CellType, Direction


def test_cell_type_traversability() -> None:
    """Test traversability property of CellType enum members."""
    assert CellType.EMPTY.is_traversable is True
    assert CellType.PICKUP_ZONE.is_traversable is True
    assert CellType.DROP_ZONE.is_traversable is True
    assert CellType.CHARGING_STATION.is_traversable is True

    assert CellType.OBSTACLE.is_traversable is False
    assert CellType.SHELF.is_traversable is False


def test_direction_vectors() -> None:
    """Test delta coordinate offsets for cardinal directions."""
    assert Direction.NORTH.delta == (0, -1)
    assert Direction.SOUTH.delta == (0, 1)
    assert Direction.EAST.delta == (1, 0)
    assert Direction.WEST.delta == (-1, 0)
    assert Direction.STAY.delta == (0, 0)

    assert Direction.NORTH.dx == 0
    assert Direction.NORTH.dy == -1
    assert Direction.EAST.dx == 1
    assert Direction.EAST.dy == 0
