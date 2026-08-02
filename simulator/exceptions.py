"""Custom exception classes for the Warehouse Digital Twin simulator."""

class WarehouseError(Exception):
    """Base exception class for all warehouse simulation errors."""

    pass


class OutOfBoundsError(WarehouseError):
    """Raised when an operation targets coordinates outside the grid boundary."""

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        super().__init__(
            f"Coordinates ({x}, {y}) are out of bounds for grid size {width}x{height}."
        )


class CellOccupiedError(WarehouseError):
    """Raised when attempting to place an entity into an already occupied cell."""

    def __init__(self, x: int, y: int, existing_entity: object = None) -> None:
        self.x = x
        self.y = y
        self.existing_entity = existing_entity
        msg = f"Cell at ({x}, {y}) is already occupied."
        if existing_entity is not None:
            msg += f" Occupying entity: {existing_entity}."
        super().__init__(msg)


class InvalidPlacementError(WarehouseError):
    """Raised when an entity placement violates warehouse topology or cell constraints."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class WarehouseConfigurationError(WarehouseError):
    """Raised when warehouse initialization receives invalid parameters."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
