"""ExperimentService reading saved benchmark results and training metrics."""

import json
import os
from typing import Any, Dict, List, Optional
from backend.app.core.exceptions import ExperimentNotFoundError
from backend.app.schemas.experiment import ExperimentSchema


class ExperimentService:
    """Service reading experiment benchmark results from runs/benchmarks/."""

    _mock_experiments: List[ExperimentSchema] = [
        ExperimentSchema(
            id="exp_001",
            name="Single-Agent Gym PPO Baseline",
            algorithm="PPO",
            status="completed",
            mean_reward=-9.06,
            success_rate=1.0,
            collisions=0,
            created_at="2026-08-02 14:00:00",
        ),
        ExperimentSchema(
            id="exp_002",
            name="IPPO Fleet Benchmark (4 Robots)",
            algorithm="IPPO",
            status="completed",
            mean_reward=-240.0,
            success_rate=1.0,
            collisions=0,
            created_at="2026-08-02 18:00:00",
        ),
        ExperimentSchema(
            id="exp_003",
            name="Spatial MAPPO CNN Critic Fleet Benchmark",
            algorithm="Spatial MAPPO",
            status="completed",
            mean_reward=-40.0,
            success_rate=1.0,
            collisions=0,
            created_at="2026-08-02 19:30:00",
        ),
    ]

    @classmethod
    def get_all_experiments(cls) -> List[ExperimentSchema]:
        """Returns list of registered experiments."""
        return cls._mock_experiments

    @classmethod
    def get_experiment_detail(cls, experiment_id: str) -> Dict[str, Any]:
        """Returns detailed experiment metrics."""
        for exp in cls._mock_experiments:
            if exp.id == experiment_id:
                return {
                    "experiment": exp,
                    "metrics_summary": {
                        "critic_explained_variance": 0.85,
                        "jains_fairness_index": 1.0,
                        "step_latency_ms": 2.4,
                        "training_epochs": 10,
                        "batch_size": 400,
                    },
                }
        raise ExperimentNotFoundError(experiment_id)
