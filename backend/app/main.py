"""FastAPI application entry point connecting all API routers and error handlers."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.algorithms import router as algorithms_router
from backend.app.api.routes.experiments import router as experiments_router
from backend.app.api.routes.simulation import router as simulation_router
from backend.app.api.routes.system import router as system_router
from backend.app.core.config import settings
from backend.app.core.exceptions import (
    AlgorithmNotFoundError,
    ExperimentNotFoundError,
    InvalidSimulationStateError,
    SimulationNotFoundError,
    algorithm_not_found_handler,
    experiment_not_found_handler,
    invalid_simulation_state_handler,
    simulation_not_found_handler,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production REST API backend serving Warehouse Digital Twin simulator and MARL algorithms.",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local and cloud frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handlers
app.add_exception_handler(SimulationNotFoundError, simulation_not_found_handler)
app.add_exception_handler(AlgorithmNotFoundError, algorithm_not_found_handler)
app.add_exception_handler(ExperimentNotFoundError, experiment_not_found_handler)
app.add_exception_handler(InvalidSimulationStateError, invalid_simulation_state_handler)

# Include API Routers under /api prefix and system endpoints
app.include_router(system_router, prefix="/api")
app.include_router(simulation_router, prefix=settings.API_PREFIX)
app.include_router(algorithms_router, prefix=settings.API_PREFIX)
app.include_router(experiments_router, prefix=settings.API_PREFIX)


@app.get("/")
def read_root():
    """Root endpoint returning basic platform information."""
    return {
        "project": settings.PROJECT_NAME,
        "status": "running",
        "version": settings.VERSION,
        "docs": "/docs",
    }
