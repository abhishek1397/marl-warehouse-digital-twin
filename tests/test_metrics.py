"""Unit tests for MetricsCollector calculation logic."""

from simulator.metrics import MetricsCollector
from simulator.position import Position
from simulator.robot import Robot, RobotState


def test_metrics_calculation() -> None:
    mc = MetricsCollector()
    fleet = {
        "r1": Robot("r1", Position(0, 0)),
        "r2": Robot("r2", Position(5, 5)),
    }
    fleet["r1"].step_towards(Position(1, 0))  # dist=1
    fleet["r2"].increment_idle_time()

    summary = mc.compute_summary(fleet, completed_deliveries=3, total_steps=10)

    assert summary["simulation_steps"] == 10
    assert summary["completed_deliveries"] == 3
    assert summary["robot_count"] == 2
    assert summary["total_distance_travelled"] == 1
    assert summary["total_idle_time"] == 1
    assert summary["average_battery_level"] == 100.0
    # 20 total robot-steps, 1 idle step -> 19 active -> 95% utilization
    assert summary["robot_utilization_pct"] == 95.0
    # 3 deliveries / 10 steps * 100 = 30.0
    assert summary["throughput_per_100_steps"] == 30.0
