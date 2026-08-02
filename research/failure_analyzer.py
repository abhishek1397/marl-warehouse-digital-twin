"""FailureClassifier module categorizing episode outcome failure modes."""

from collections import Counter
from typing import Dict, List

from research.trajectory_recorder import EpisodeTrajectory


class FailureMode:
    """Enumeration constants for episode failure modes."""

    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    EXCESSIVE_COLLISIONS = "EXCESSIVE_COLLISIONS"
    BATTERY_DEPLETED = "BATTERY_DEPLETED"
    NEVER_FOUND_PACKAGE = "NEVER_FOUND_PACKAGE"
    PICKED_BUT_NEVER_DELIVERED = "PICKED_BUT_NEVER_DELIVERED"
    PASSIVE_WAIT = "PASSIVE_WAIT"
    OSCILLATION = "OSCILLATION"


class FailureClassifier:
    """Classifies episode execution outcomes into explicit failure taxonomy categories."""

    @staticmethod
    def classify_episode(trajectory: EpisodeTrajectory) -> str:
        """Determines primary failure mode for a recorded trajectory."""
        if trajectory.is_success:
            return FailureMode.SUCCESS

        # 1. Excessive Collisions (>= 5 collisions)
        if trajectory.total_collisions >= 5:
            return FailureMode.EXCESSIVE_COLLISIONS

        steps = trajectory.steps
        if not steps:
            return FailureMode.TIMEOUT

        # 3. Passive Wait Policy (> 50% steps spent waiting)
        wait_count = sum(1 for s in steps if s.action == 4)
        if wait_count / len(steps) > 0.5:
            return FailureMode.PASSIVE_WAIT

        # 4. Oscillation (back-and-forth between same 2 positions >= 5 times)
        if len(steps) >= 6:
            positions = [s.position for s in steps]
            osc_count = 0
            for i in range(len(positions) - 2):
                if positions[i] == positions[i + 2] and positions[i] != positions[i + 1]:
                    osc_count += 1
            if osc_count >= 5:
                return FailureMode.OSCILLATION

        # 5. Picked but never delivered
        if trajectory.total_pickups > 0 and trajectory.total_deliveries == 0:
            return FailureMode.PICKED_BUT_NEVER_DELIVERED

        # 6. Never found package
        if trajectory.total_pickups == 0:
            return FailureMode.NEVER_FOUND_PACKAGE

        return FailureMode.TIMEOUT

    @staticmethod
    def summarize_failure_distribution(trajectories: List[EpisodeTrajectory]) -> Dict[str, int]:
        """Summarizes failure mode distribution counts across all trajectories."""
        counts = Counter()
        for t in trajectories:
            mode = FailureClassifier.classify_episode(t)
            counts[mode] += 1
        return dict(counts)
