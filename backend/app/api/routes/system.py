"""System endpoints for health check and platform version info."""

from fastapi import APIRouter
from backend.app.core.config import settings

router = APIRouter(tags=["System"])


@router.get("/health", summary="Health Check")
def health_check():
    """Returns backend system status."""
    return {"status": "ok", "service": settings.PROJECT_NAME}


@router.get("/version", summary="Platform Version")
def version_check():
    """Returns platform version details."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": "production",
    }
