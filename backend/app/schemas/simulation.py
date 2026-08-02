"""Pydantic schemas for simulation API requests and responses."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SimulationCreateRequest(BaseModel):
    grid_width: int = Field(default=8, ge=4, le=50, description="Warehouse grid width")
    grid_height: int = Field(default=8, ge=4, le=50, description="Warehouse grid height")
    num_robots: int = Field(default=2, ge=1, le=32, description="Number of autonomous robots")
    enable_pbrs: bool = Field(default=True, description="Enable Potential-Based Reward Shaping")
    enable_dam: bool = Field(default=True, description="Enable Dynamic Action Masking")


class SimulationStepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=100, description="Number of simulation steps to advance")


class RobotStateSchema(BaseModel):
    id: str
    position: List[int]
    battery_level: float
    state: str
    assigned_task: Optional[str] = None


class GridEntitySchema(BaseModel):
    id: str
    position: List[int]
    type: str


class LiveMetricsSchema(BaseModel):
    episode: int
    step: int
    reward: float
    throughput: float
    collisions: int
    idle_robots: int
    battery_avg: float
    packages_delivered: int
    fps: int = 60
    policy_entropy: float = 1.0


class SimulationStateResponse(BaseModel):
    is_initialized: bool
    is_running: bool
    is_paused: bool
    step_count: int
    grid_size: List[int]
    robots: List[RobotStateSchema]
    entities: List[GridEntitySchema]
    metrics: LiveMetricsSchema
