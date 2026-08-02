"""Unit tests for TrafficController congestion and deadlock detection."""

from simulator.position import Position
from simulator.robot import Robot, RobotState
from simulator.traffic_controller import TrafficController


def test_traffic_controller_congestion_detection() -> None:
    tc = TrafficController()
    fleet = {
        "r1": Robot("r1", Position(5, 5)),
        "r2": Robot("r2", Position(5, 6)),
        "r3": Robot("r3", Position(6, 5)),
        "r4": Robot("r4", Position(0, 0)),
    }

    congested = tc.detect_congestion(fleet, radius=2, density_threshold=3)
    assert len(congested) > 0
    assert Position(5, 5) in congested


def test_traffic_controller_deadlock_detection() -> None:
    tc = TrafficController()
    r1 = Robot("r1", Position(2, 2))
    r1.state = RobotState.MOVING_TO_PICKUP
    fleet = {"r1": r1}

    # Step 1-3 at same position
    assert len(tc.detect_deadlocks(fleet, stall_threshold=3)) == 0
    assert len(tc.detect_deadlocks(fleet, stall_threshold=3)) == 0
    stalled = tc.detect_deadlocks(fleet, stall_threshold=3)

    assert len(stalled) == 1
    assert stalled[0] == "r1"
