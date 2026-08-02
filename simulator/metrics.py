"""MetricsCollector class gathering, calculating, and formatting simulation performance telemetry."""

from typing import Any, Dict, List, Optional

from simulator.robot import Robot


class MetricsCollector:
    """Collects step-by-step performance metrics and generates system summary statistics."""

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []

        # Planning engine telemetry counters
        self.total_planning_time_ms: float = 0.0
        self.total_plans_executed: int = 0
        self.total_replans: int = 0
        self.total_collisions_prevented: int = 0
        self.total_deadlocks_prevented: int = 0
        self.total_robot_waiting_steps: int = 0

    def record_planning_event(
        self,
        planning_time_ms: float,
        is_replan: bool = False,
        collisions_prevented: int = 0,
        deadlocks_prevented: int = 0,
    ) -> None:
        """Records a path planning calculation event."""
        self.total_planning_time_ms += planning_time_ms
        self.total_plans_executed += 1
        if is_replan:
            self.total_replans += 1
        self.total_collisions_prevented += collisions_prevented
        self.total_deadlocks_prevented += deadlocks_prevented

    def record_step(
        self,
        step_index: int,
        fleet: Dict[str, Robot],
        completed_deliveries: int,
    ) -> Dict[str, Any]:
        """Records telemetry snapshot for a single simulation step."""
        robot_count = len(fleet)
        if robot_count == 0:
            avg_battery = 0.0
            total_distance = 0
            total_idle = 0
        else:
            avg_battery = sum(r.battery_percentage for r in fleet.values()) / robot_count
            total_distance = sum(r.total_distance_travelled for r in fleet.values())
            total_idle = sum(r.idle_steps for r in fleet.values())

        snapshot = {
            "step": step_index,
            "completed_deliveries": completed_deliveries,
            "average_battery": avg_battery,
            "total_distance": total_distance,
            "total_idle_steps": total_idle,
        }
        self._history.append(snapshot)
        return snapshot

    def compute_summary(
        self,
        fleet: Dict[str, Robot],
        completed_deliveries: int,
        total_steps: int,
    ) -> Dict[str, Any]:
        """Computes aggregated performance metrics for the simulation run."""
        robot_count = len(fleet)
        total_distance = sum(r.total_distance_travelled for r in fleet.values()) if robot_count > 0 else 0
        total_idle = sum(r.idle_steps for r in fleet.values()) if robot_count > 0 else 0
        avg_battery = (
            sum(r.battery_percentage for r in fleet.values()) / robot_count if robot_count > 0 else 0.0
        )

        total_robot_steps = total_steps * robot_count
        active_steps = total_robot_steps - total_idle
        robot_utilization = (active_steps / total_robot_steps * 100.0) if total_robot_steps > 0 else 0.0
        throughput = (completed_deliveries / total_steps * 100.0) if total_steps > 0 else 0.0
        avg_plan_time = (
            self.total_planning_time_ms / self.total_plans_executed
            if self.total_plans_executed > 0
            else 0.0
        )

        return {
            "simulation_steps": total_steps,
            "completed_deliveries": completed_deliveries,
            "robot_count": robot_count,
            "total_distance_travelled": total_distance,
            "total_idle_time": total_idle,
            "average_battery_level": round(avg_battery, 2),
            "robot_utilization_pct": round(robot_utilization, 2),
            "throughput_per_100_steps": round(throughput, 2),
            "total_planning_time_ms": round(self.total_planning_time_ms, 2),
            "avg_planning_time_ms": round(avg_plan_time, 2),
            "plans_executed": self.total_plans_executed,
            "replans_count": self.total_replans,
            "collisions_prevented": self.total_collisions_prevented,
            "deadlocks_prevented": self.total_deadlocks_prevented,
            "robot_waiting_steps": self.total_robot_waiting_steps,
        }

    def print_summary(
        self,
        fleet: Dict[str, Robot],
        completed_deliveries: int,
        total_steps: int,
    ) -> None:
        """Prints formatted simulation metrics report to stdout."""
        metrics = self.compute_summary(fleet, completed_deliveries, total_steps)

        print("\n" + "=" * 65)
        print("         WAREHOUSE DIGITAL TWIN - SIMULATION METRICS        ")
        print("=" * 65)
        print(f" Total Simulation Steps Elapsed    : {metrics['simulation_steps']}")
        print(f" Total Completed Deliveries         : {metrics['completed_deliveries']}")
        print(f" Active Robot Count                 : {metrics['robot_count']}")
        print(f" Total Distance Travelled (steps)   : {metrics['total_distance_travelled']}")
        print(f" Total Idle Time (robot-steps)      : {metrics['total_idle_time']}")
        print(f" Fleet Average Battery Level        : {metrics['average_battery_level']}%")
        print(f" Fleet Robot Utilization Rate       : {metrics['robot_utilization_pct']}%")
        print(f" Delivery Throughput (per 100 steps): {metrics['throughput_per_100_steps']}")
        print("-" * 65)
        print(" CLASSICAL PLANNING ENGINE BASELINE METRICS")
        print("-" * 65)
        print(f" Total Planning Time               : {metrics['total_planning_time_ms']} ms")
        print(f" Average Plan Time per Request     : {metrics['avg_planning_time_ms']} ms")
        print(f" Total Path Plans Executed         : {metrics['plans_executed']}")
        print(f" Total Replans Triggered            : {metrics['replans_count']}")
        print(f" Collisions Prevented               : {metrics['collisions_prevented']}")
        print(f" Deadlocks Prevented                : {metrics['deadlocks_prevented']}")
        print("=" * 65 + "\n")
