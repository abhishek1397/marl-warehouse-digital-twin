"""Simulation control REST API endpoints."""

from fastapi import APIRouter
from backend.app.schemas.simulation import (
    SimulationCreateRequest,
    SimulationStateResponse,
    SimulationStepRequest,
)
from backend.app.services.simulation_service import SimulationService

router = APIRouter(prefix="/simulation", tags=["Simulation"])


@router.post("/create", response_model=SimulationStateResponse, summary="Create Simulation")
def create_simulation(req: SimulationCreateRequest):
    """Initializes a new Warehouse Digital Twin simulator instance."""
    service = SimulationService.get_instance()
    return service.create_simulation(
        grid_width=req.grid_width,
        grid_height=req.grid_height,
        num_robots=req.num_robots,
        enable_pbrs=req.enable_pbrs,
        enable_dam=req.enable_dam,
    )


@router.post("/start", response_model=SimulationStateResponse, summary="Start Simulation")
def start_simulation():
    """Starts simulation execution."""
    service = SimulationService.get_instance()
    return service.start()


@router.post("/pause", response_model=SimulationStateResponse, summary="Pause Simulation")
def pause_simulation():
    """Pauses simulation execution."""
    service = SimulationService.get_instance()
    return service.pause()


@router.post("/reset", response_model=SimulationStateResponse, summary="Reset Simulation")
def reset_simulation():
    """Resets simulation environment to initial state."""
    service = SimulationService.get_instance()
    return service.reset()


@router.get("/state", response_model=SimulationStateResponse, summary="Get Simulation State")
def get_simulation_state():
    """Retrieves current 2D grid entity positions and live metrics."""
    service = SimulationService.get_instance()
    return service.get_state()


@router.post("/step", response_model=SimulationStateResponse, summary="Step Simulation")
def step_simulation(req: SimulationStepRequest):
    """Steps simulation environment forward by specified number of timesteps."""
    service = SimulationService.get_instance()
    return service.step(steps=req.steps)
