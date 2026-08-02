"""Constants and Enumerations for the Warehouse Digital Twin simulator."""

from enum import Enum, auto
from typing import Tuple


class CellType(Enum):
    """Enumeration of functional cell types within the warehouse grid layout."""

    EMPTY = auto()
    OBSTACLE = auto()
    SHELF = auto()
    PICKUP_ZONE = auto()
    DROP_ZONE = auto()
    CHARGING_STATION = auto()

    @property
    def is_traversable(self) -> bool:
        """Returns True if the cell type is inherently walkable by default."""
        return self in {
            CellType.EMPTY,
            CellType.PICKUP_ZONE,
            CellType.DROP_ZONE,
            CellType.CHARGING_STATION,
        }


class Direction(Enum):
    """Enumeration of cardinal movement directions and stay action with delta vectors (dx, dy)."""

    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)
    STAY = (0, 0)

    @property
    def delta(self) -> Tuple[int, int]:
        """Returns the coordinate offset tuple (dx, dy) for the direction."""
        return self.value

    @property
    def dx(self) -> int:
        """Returns the horizontal coordinate offset."""
        return self.value[0]

    @property
    def dy(self) -> int:
        """Returns the vertical coordinate offset."""
        return self.value[1]


# Default Warehouse Grid Dimensions
DEFAULT_GRID_WIDTH: int = 50
DEFAULT_GRID_HEIGHT: int = 50
MIN_GRID_DIMENSION: int = 2
MAX_GRID_DIMENSION: int = 1000
