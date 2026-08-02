"""Algorithm metadata and selection REST API endpoints."""

from fastapi import APIRouter
from backend.app.schemas.algorithm import (
    AlgorithmListResponse,
    AlgorithmMetadataSchema,
    AlgorithmSelectRequest,
)
from backend.app.services.algorithm_service import AlgorithmService

router = APIRouter(prefix="/algorithms", tags=["Algorithms"])


@router.get("", response_model=AlgorithmListResponse, summary="List Algorithms")
def list_algorithms():
    """Lists metadata for all supported MARL algorithms."""
    return AlgorithmListResponse(algorithms=AlgorithmService.get_all_algorithms())


@router.get("/{name}", response_model=AlgorithmMetadataSchema, summary="Get Algorithm Details")
def get_algorithm(name: str):
    """Retrieves metadata for specific MARL algorithm by name."""
    return AlgorithmService.get_algorithm(name)


@router.post("/select", summary="Select Active Algorithm")
def select_algorithm(req: AlgorithmSelectRequest):
    """Validates and selects active MARL algorithm for simulation execution."""
    meta = AlgorithmService.get_algorithm(req.algorithm_name)
    return {"message": f"Algorithm '{meta.name}' selected successfully.", "algorithm": meta}
