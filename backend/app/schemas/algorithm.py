"""Pydantic schemas for MARL algorithm API endpoints."""

from typing import List, Optional
from pydantic import BaseModel


class AlgorithmMetadataSchema(BaseModel):
    name: str
    category: str
    paradigm: str
    actor_architecture: str
    critic_architecture: str
    reward_shaping: bool
    action_masking: bool
    description: str


class AlgorithmSelectRequest(BaseModel):
    algorithm_name: str


class AlgorithmListResponse(BaseModel):
    algorithms: List[AlgorithmMetadataSchema]
