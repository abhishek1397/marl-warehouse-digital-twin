"""Experiment benchmark REST API endpoints."""

from fastapi import APIRouter
from backend.app.schemas.experiment import (
    ExperimentDetailResponse,
    ExperimentListResponse,
)
from backend.app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("", response_model=ExperimentListResponse, summary="List Experiments")
def list_experiments():
    """Lists registered training experiments and multi-seed benchmarks."""
    return ExperimentListResponse(experiments=ExperimentService.get_all_experiments())


@router.get("/{id}", response_model=ExperimentDetailResponse, summary="Get Experiment Detail")
def get_experiment_detail(id: str):
    """Retrieves metrics summary for specific experiment ID."""
    return ExperimentService.get_experiment_detail(id)
