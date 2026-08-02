"""Immutable Position dataclass representing discrete 2D spatial coordinates."""

import math
from dataclasses import dataclass
from typing import List

from simulator.constants import Direction


@dataclass(frozen=True, order=True)
class Position:
    """Immutable discrete 2D coordinate representation on a grid."""

    x: int
    y: int

    def manhattan_distance(self, other: "Position") -> int:
        """Calculates the Manhattan (L1) distance to another Position."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def euclidean_distance(self, other: "Position") -> float:
        """Calculates the Euclidean (L2) distance to another Position."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def get_neighbor(self, direction: Direction) -> "Position":
        """Returns the adjacent Position in the given cardinal direction."""
        return Position(self.x + direction.dx, self.y + direction.dy)

    def get_all_neighbors(self, include_diagonals: bool = False) -> List["Position"]:
        """Returns all adjacent positions (4 cardinal, or 8 with diagonals)."""
        cardinal_dirs = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        neighbors = [self.get_neighbor(d) for d in cardinal_dirs]

        if include_diagonals:
            diagonal_offsets = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            for dx, dy in diagonal_offsets:
                neighbors.append(Position(self.x + dx, self.y + dy))

        return neighbors

    def to_tuple(self) -> tuple[int, int]:
        """Returns coordinates as an (x, y) integer tuple."""
        return (self.x, self.y)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Position(x={self.x}, y={self.y})"
