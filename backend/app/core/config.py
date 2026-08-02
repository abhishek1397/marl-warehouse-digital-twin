"""Application Configuration Settings using Pydantic BaseModel."""

from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    """Global configuration settings for backend API application."""

    PROJECT_NAME: str = "Warehouse Digital Twin MARL Platform API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    CORS_ORIGINS: List[str] = ["*"]


settings = Settings()
