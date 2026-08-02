"""Unit tests for Position dataclass."""

import pytest

from simulator.constants import Direction
from simulator.position import Position


def test_position_creation_and_attributes() -> None:
    """Test Position instantiation and properties."""
    pos = Position(3, 5)
    assert pos.x == 3
    assert pos.y == 5
    assert pos.to_tuple() == (3, 5)
    assert str(pos) == "(3, 5)"
    assert repr(pos) == "Position(x=3, y=5)"


def test_position_immutability() -> None:
    """Test that Position instances are frozen and immutable."""
    pos = Position(1, 2)
    with pytest.raises(AttributeError):
        pos.x = 10  # type: ignore[misc]


def test_position_equality_and_hashing() -> None:
    """Test equality comparison and hashing behavior for set/dict usage."""
    p1 = Position(4, 4)
    p2 = Position(4, 4)
    p3 = Position(5, 4)

    assert p1 == p2
    assert p1 != p3
    assert hash(p1) == hash(p2)
    assert len({p1, p2, p3}) == 2


def test_manhattan_distance() -> None:
    """Test Manhattan distance calculation between positions."""
    p1 = Position(0, 0)
    p2 = Position(3, 4)
    assert p1.manhattan_distance(p2) == 7
    assert p2.manhattan_distance(p1) == 7


def test_euclidean_distance() -> None:
    """Test Euclidean distance calculation between positions."""
    p1 = Position(0, 0)
    p2 = Position(3, 4)
    assert p1.euclidean_distance(p2) == pytest.approx(5.0)


def test_get_neighbor() -> None:
    """Test single direction neighbor derivation."""
    origin = Position(5, 5)
    assert origin.get_neighbor(Direction.NORTH) == Position(5, 4)
    assert origin.get_neighbor(Direction.SOUTH) == Position(5, 6)
    assert origin.get_neighbor(Direction.EAST) == Position(6, 5)
    assert origin.get_neighbor(Direction.WEST) == Position(4, 5)
    assert origin.get_neighbor(Direction.STAY) == Position(5, 5)


def test_get_all_neighbors_cardinal_and_diagonal() -> None:
    """Test retrieval of 4 cardinal and 8 cardinal+diagonal neighbors."""
    center = Position(2, 2)

    cardinal = center.get_all_neighbors(include_diagonals=False)
    assert len(cardinal) == 4
    assert Position(2, 1) in cardinal
    assert Position(3, 2) in cardinal
    assert Position(2, 3) in cardinal
    assert Position(1, 2) in cardinal

    all_neighbors = center.get_all_neighbors(include_diagonals=True)
    assert len(all_neighbors) == 8
    assert Position(1, 1) in all_neighbors
    assert Position(3, 3) in all_neighbors
