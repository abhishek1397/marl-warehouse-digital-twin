from enum import Enum
from typing import Any, Dict, List


class FailureMode(Enum):
    SUCCESS = "success"
    EXCESSIVE_COLLISIONS = "excessive_collisions"
    NEVER_FOUND_PACKAGE = "never_found_package"
    CRITIC_COLLAPSE = "critic_collapse"
    BATTERY_DEPLETION = "battery_depletion"
    CORRIDOR_DEADLOCK = "corridor_deadlock"


class FailureClassifier:
    """Classifies trajectory failure modes for policy analysis."""

    @staticmethod
    def classify_episode(trajectory: Any) -> FailureMode:
        if getattr(trajectory, "is_success", False):
            return FailureMode.SUCCESS
        if getattr(trajectory, "total_collisions", 0) > 0:
            return FailureMode.EXCESSIVE_COLLISIONS
        return FailureMode.NEVER_FOUND_PACKAGE

    @staticmethod
    def classify_trajectory(trajectory: List[Dict[str, Any]]) -> FailureMode:
        if not trajectory:
            return FailureMode.SUCCESS
        last_step = trajectory[-1]
        if last_step.get("is_collision", False):
            return FailureMode.CORRIDOR_DEADLOCK
        return FailureMode.SUCCESS


class MAPPOFailureClassifier:
    """Automatically classifies MAPPO failure modes from training metrics."""

    @staticmethod
    def classify_failure_mode(
        critic_loss: float,
        explained_variance: float,
        collisions: int,
        n_robots: int,
    ) -> Dict[str, Any]:
        """Classifies failure modes: Critic Collapse, Value Saturation, Coordination Deadlock, State Explosion."""
        failure_modes = []

        if explained_variance < 0.0:
            failure_modes.append("CRITIC_EXPLAINED_VARIANCE_COLLAPSE")
        if critic_loss > 10000.0:
            failure_modes.append("CRITIC_VALUE_EXPLOSION")
        if collisions > 100:
            failure_modes.append("COORDINATION_DEADLOCK")
        if n_robots >= 8:
            failure_modes.append("CENTRALIZED_STATE_DIMENSION_EXPLOSION")

        return {
            "has_failure": len(failure_modes) > 0,
            "failure_modes": failure_modes,
            "primary_failure": failure_modes[0] if failure_modes else "NONE",
        }
