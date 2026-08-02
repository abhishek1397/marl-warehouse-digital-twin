"""Distance metrics for reward shaping potential functions."""

import math
from typing import Union

from simulator.position import Position


def manhattan_distance(
    pos1: Union[Position, tuple, list], pos2: Union[Position, tuple, list]
) -> float:
    """Computes L1 Manhattan distance between two positions."""
    x1, y1 = (pos1.x, pos1.y) if isinstance(pos1, Position) else (pos1[0], pos1[1])
    x2, y2 = (pos2.x, pos2.y) if isinstance(pos2, Position) else (pos2[0], pos2[1])
    return float(abs(x1 - x2) + abs(y1 - y2))


def euclidean_distance(
    pos1: Union[Position, tuple, list], pos2: Union[Position, tuple, list]
) -> float:
    """Computes L2 Euclidean distance between two positions."""
    x1, y1 = (pos1.x, pos1.y) if isinstance(pos1, Position) else (pos1[0], pos1[1])
    x2, y2 = (pos2.x, pos2.y) if isinstance(pos2, Position) else (pos2[0], pos2[1])
    return float(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def chebyshev_distance(
    pos1: Union[Position, tuple, list], pos2: Union[Position, tuple, list]
) -> float:
    """Computes L_infinity Chebyshev distance between two positions."""
    x1, y1 = (pos1.x, pos1.y) if isinstance(pos1, Position) else (pos1[0], pos1[1])
    x2, y2 = (pos2.x, pos2.y) if isinstance(pos2, Position) else (pos2[0], pos2[1])
    return float(max(abs(x1 - x2), abs(y1 - y2)))
