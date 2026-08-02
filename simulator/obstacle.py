"""Obstacle class representing static physical barriers within the warehouse grid."""

from simulator.position import Position


class Obstacle:
    """Represents a static non-traversable obstacle placed on the warehouse grid."""

    def __init__(
        self,
        obstacle_id: str,
        position: Position,
        name: str = "Obstacle",
        description: str = "",
    ) -> None:
        self._obstacle_id: str = obstacle_id
        self._position: Position = position
        self._name: str = name
        self._description: str = description

    @property
    def obstacle_id(self) -> str:
        """Returns the unique identifier of the obstacle."""
        return self._obstacle_id

    @property
    def position(self) -> Position:
        """Returns the spatial position of the obstacle."""
        return self._position

    @property
    def name(self) -> str:
        """Returns the human-readable name of the obstacle."""
        return self._name

    @property
    def description(self) -> str:
        """Returns the optional description of the obstacle."""
        return self._description

    def __repr__(self) -> str:
        return f"Obstacle(id='{self._obstacle_id}', position={self._position})"

    def __str__(self) -> str:
        return f"Obstacle '{self._obstacle_id}' at {self._position}"
