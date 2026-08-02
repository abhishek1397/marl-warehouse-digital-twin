"""Pydantic schemas for experiment API endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ExperimentSchema(BaseModel):
    id: str
    name: str
    algorithm: str
    status: str
    mean_reward: float
    success_rate: float
    collisions: int
    created_at: str


class ExperimentDetailResponse(BaseModel):
    experiment: ExperimentSchema
    metrics_summary: Dict[str, Any]


class ExperimentListResponse(BaseModel):
    experiments: List[ExperimentSchema]
