"""Custom exceptions and FastAPI error handlers."""

from fastapi import Request
from fastapi.responses import JSONResponse


class SimulationNotFoundError(Exception):
    """Raised when requested simulation instance does not exist."""
    def __init__(self, detail: str = "Simulation not initialized.") -> None:
        self.detail = detail
        super().__init__(detail)


class AlgorithmNotFoundError(Exception):
    """Raised when requested algorithm metadata is unknown."""
    def __init__(self, algorithm_name: str) -> None:
        self.detail = f"Algorithm '{algorithm_name}' not found."
        super().__init__(self.detail)


class ExperimentNotFoundError(Exception):
    """Raised when requested experiment ID is unknown."""
    def __init__(self, experiment_id: str) -> None:
        self.detail = f"Experiment '{experiment_id}' not found."
        super().__init__(self.detail)


class InvalidSimulationStateError(Exception):
    """Raised when performing invalid operations on simulation state."""
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


async def simulation_not_found_handler(request: Request, exc: SimulationNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "Simulation Error", "detail": exc.detail})


async def algorithm_not_found_handler(request: Request, exc: AlgorithmNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "Algorithm Error", "detail": exc.detail})


async def experiment_not_found_handler(request: Request, exc: ExperimentNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "Experiment Error", "detail": exc.detail})


async def invalid_simulation_state_handler(request: Request, exc: InvalidSimulationStateError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "Invalid State", "detail": exc.detail})
