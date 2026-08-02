"""Comprehensive test suite for MAPPO diagnostic verifier modules."""

import pytest
import torch

from research.ablation_runner import AblationRunner
from research.coordination_metrics import CoordinationMetrics
from research.critic_analysis import CriticAnalyzer
from research.ctde_validator import CTDEValidator
from research.failure_analyzer import MAPPOFailureClassifier
from research.joint_state_analysis import JointStateAnalyzer
from research.learning_curve_analysis import LearningCurveAnalyzer
from research.scalability_profiler import ScalabilityProfiler


def test_ctde_validator() -> None:
    res = CTDEValidator.validate_ctde_separation()
    assert res["status"] == "PASSED"
    assert res["passed"] is True


def test_critic_analyzer() -> None:
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.1, 1.9, 3.1])
    ev = CriticAnalyzer.compute_explained_variance(y_true, y_pred)
    assert ev > 0.9

    res = CriticAnalyzer.analyze_critic_trajectory([1.0, 0.5], [1.0, 2.0], [0.9, 1.9])
    assert res["status"] == "PASSED"


def test_joint_state_analyzer() -> None:
    res = JointStateAnalyzer.analyze_scaling()
    assert res["status"] == "COMPLETED"
    assert "2_robots" in res["fleet_scaling"]


def test_coordination_metrics() -> None:
    res = CoordinationMetrics.compute_coordination_summary(collisions=0, deliveries=10, steps=100, fleet_size=2)
    assert res["collision_avoidance_rate"] == 1.0
    assert res["throughput"] == 0.1


def test_scalability_profiler() -> None:
    res = ScalabilityProfiler.profile_fleet_scalability(fleet_sizes=[1])
    assert res["status"] == "COMPLETED"
    assert "1_robots" in res["fleet_profiles"]


def test_ablation_runner() -> None:
    res = AblationRunner.run_ablations(num_timesteps=50)
    assert res["status"] == "COMPLETED"
    assert "baseline" in res["ablations"]


def test_learning_curve_analyzer() -> None:
    res = LearningCurveAnalyzer.analyze_learning_curve([-10.0, -5.0, -2.0])
    assert res["final_reward"] == -2.0


def test_failure_classifier() -> None:
    res = MAPPOFailureClassifier.classify_failure_mode(critic_loss=50.0, explained_variance=-0.2, collisions=0, n_robots=8)
    assert res["has_failure"] is True
    assert "CRITIC_EXPLAINED_VARIANCE_COLLAPSE" in res["failure_modes"]
