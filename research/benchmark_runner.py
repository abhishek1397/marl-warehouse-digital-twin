"""Automated benchmark scenario execution engine, CSV metrics export, and plot generator."""

import csv
import os
import random
import sys
import time
from typing import Any, Dict, List

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt

from simulator.battery_manager import BatteryManager
from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.metrics import MetricsCollector
from simulator.obstacle import Obstacle
from simulator.package import Package
from simulator.planner import MultiRobotPlanner
from simulator.position import Position
from simulator.robot import Robot
from simulator.scheduler import PrioritySchedulerStrategy, TaskScheduler
from simulator.shelf import Shelf
from simulator.simulation import SimulationEngine
from simulator.task import TaskPriority, TaskType
from simulator.task_manager import TaskManager
from simulator.warehouse import Warehouse


SCENARIO_CONFIGS = [
    {
        "name": "Small",
        "width": 20,
        "height": 20,
        "num_robots": 5,
        "num_shelves": 10,
        "num_chargers": 2,
        "num_obstacles": 15,
        "num_packages": 10,
        "max_steps": 200,
    },
    {
        "name": "Medium",
        "width": 50,
        "height": 50,
        "num_robots": 20,
        "num_shelves": 30,
        "num_chargers": 5,
        "num_obstacles": 50,
        "num_packages": 40,
        "max_steps": 300,
    },
    {
        "name": "Large",
        "width": 100,
        "height": 100,
        "num_robots": 50,
        "num_shelves": 60,
        "num_chargers": 10,
        "num_obstacles": 150,
        "num_packages": 100,
        "max_steps": 400,
    },
    {
        "name": "Stress",
        "width": 200,
        "height": 200,
        "num_robots": 100,
        "num_shelves": 120,
        "num_chargers": 20,
        "num_obstacles": 300,
        "num_packages": 200,
        "max_steps": 500,
    },
]

SEEDS = [42, 43, 44]


def build_random_benchmark_engine(cfg: Dict[str, Any], seed: int) -> SimulationEngine:
    """Builds a deterministic benchmark simulation environment for a given scenario config and seed."""
    random.seed(seed)
    width, height = cfg["width"], cfg["height"]
    name = f"Benchmark_{cfg['name']}_s{seed}"

    warehouse = Warehouse(width=width, height=height, name=name)
    task_manager = TaskManager()
    battery_manager = BatteryManager()
    metrics_collector = MetricsCollector()
    planner = MultiRobotPlanner()
    task_scheduler = TaskScheduler(strategy=PrioritySchedulerStrategy())

    engine = SimulationEngine(
        warehouse=warehouse,
        task_manager=task_manager,
        battery_manager=battery_manager,
        metrics_collector=metrics_collector,
        planner=planner,
        task_scheduler=task_scheduler,
    )

    # 1. Spawn Obstacles
    placed_coords = set()
    for o_idx in range(cfg["num_obstacles"]):
        rx, ry = random.randint(0, width - 1), random.randint(0, height - 1)
        pos = Position(rx, ry)
        if pos not in placed_coords:
            placed_coords.add(pos)
            obs = Obstacle(f"obs_{o_idx}", pos)
            warehouse.place_object(pos, obs, cell_type=CellType.OBSTACLE)

    # 2. Spawn Shelves
    for s_idx in range(cfg["num_shelves"]):
        rx, ry = random.randint(0, width - 1), random.randint(0, height - 1)
        pos = Position(rx, ry)
        if pos not in placed_coords:
            placed_coords.add(pos)
            shelf = Shelf(f"shelf_{s_idx}", pos, capacity=10)
            warehouse.place_object(pos, shelf, cell_type=CellType.SHELF)
            engine.add_shelf(shelf)

    # 3. Spawn Charging Stations
    for c_idx in range(cfg["num_chargers"]):
        rx, ry = random.randint(0, width - 1), random.randint(0, height - 1)
        pos = Position(rx, ry)
        if pos not in placed_coords:
            placed_coords.add(pos)
            station = ChargingStation(f"charger_{c_idx}", pos, charge_rate=15.0, capacity=2)
            warehouse.place_object(pos, station, cell_type=CellType.CHARGING_STATION)
            engine.add_charging_station(station)

    # 4. Spawn Robots
    for r_idx in range(cfg["num_robots"]):
        rx, ry = random.randint(0, width - 1), random.randint(0, height - 1)
        pos = Position(rx, ry)
        while pos in placed_coords or not warehouse.validate_placement(pos):
            rx, ry = random.randint(0, width - 1), random.randint(0, height - 1)
            pos = Position(rx, ry)

        placed_coords.add(pos)
        robot = Robot(f"robot_{r_idx:03d}", initial_position=pos, max_battery=100.0)
        engine.add_robot(robot)

    # 5. Spawn Packages and Tasks
    for p_idx in range(1, cfg["num_packages"] + 1):
        src_x, src_y = random.randint(0, width - 1), random.randint(0, height - 1)
        dst_x, dst_y = random.randint(0, width - 1), random.randint(0, height - 1)
        src_pos = Position(src_x, src_y)
        dst_pos = Position(dst_x, dst_y)

        pkg = Package(f"pkg_{p_idx:03d}", source_position=src_pos, destination_position=dst_pos)
        engine.add_package(pkg)

        task_manager.create_task(
            task_id=f"task_{p_idx:03d}",
            task_type=TaskType.PICKUP_AND_DELIVER,
            pickup_position=src_pos,
            drop_position=dst_pos,
            package=pkg,
            priority=random.choice([TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH]),
        )

    engine.initialize()
    return engine


def run_benchmark_suite(output_dir: str = "runs/benchmarks") -> List[Dict[str, Any]]:
    """Runs all benchmark scenarios across seeds and exports CSVs and plot charts."""
    os.makedirs(output_dir, exist_ok=True)
    raw_results: List[Dict[str, Any]] = []

    print("=" * 70)
    print("      EXECUTING WAREHOUSE SIMULATOR BENCHMARK SUITE")
    print("=" * 70)

    for cfg in SCENARIO_CONFIGS:
        for seed in SEEDS:
            print(
                f"--> Running Scenario: {cfg['name']} ({cfg['width']}x{cfg['height']}, "
                f"{cfg['num_robots']} robots, {cfg['num_packages']} tasks) [Seed {seed}]..."
            )
            t0 = time.perf_counter()
            engine = build_random_benchmark_engine(cfg, seed)
            summary = engine.run(max_steps=cfg["max_steps"])
            total_time_s = time.perf_counter() - t0

            robot_count = len(engine.fleet)
            avg_travel_dist = (
                summary["total_distance_travelled"] / robot_count if robot_count > 0 else 0.0
            )
            avg_wait_time = summary["total_idle_time"] / robot_count if robot_count > 0 else 0.0

            result_entry = {
                "scenario": cfg["name"],
                "seed": seed,
                "grid_size": f"{cfg['width']}x{cfg['height']}",
                "num_robots": cfg["num_robots"],
                "num_tasks": cfg["num_packages"],
                "simulation_steps": summary["simulation_steps"],
                "completed_deliveries": summary["completed_deliveries"],
                "throughput_per_100_steps": summary["throughput_per_100_steps"],
                "robot_utilization_pct": summary["robot_utilization_pct"],
                "avg_travel_distance": round(avg_travel_dist, 2),
                "total_idle_time": summary["total_idle_time"],
                "avg_wait_time": round(avg_wait_time, 2),
                "total_planning_time_ms": summary["total_planning_time_ms"],
                "avg_planning_time_ms": summary["avg_planning_time_ms"],
                "replans_count": summary["replans_count"],
                "collisions_prevented": summary["collisions_prevented"],
                "deadlocks_prevented": summary["deadlocks_prevented"],
                "average_battery_level": summary["average_battery_level"],
                "wall_runtime_seconds": round(total_time_s, 2),
            }
            raw_results.append(result_entry)

    # 1. Export Raw Benchmark CSV
    csv_raw_path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = list(raw_results[0].keys())
    with open(csv_raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_results)

    print(f"\n[OK] Raw benchmark metrics saved to: {csv_raw_path}")

    # 2. Compute and Export Summary CSV (averaged across seeds)
    summary_rows: List[Dict[str, Any]] = []
    for cfg in SCENARIO_CONFIGS:
        scen_runs = [r for r in raw_results if r["scenario"] == cfg["name"]]
        count = len(scen_runs)
        if count == 0:
            continue

        avg_entry = {
            "scenario": cfg["name"],
            "grid_size": f"{cfg['width']}x{cfg['height']}",
            "num_robots": cfg["num_robots"],
            "num_tasks": cfg["num_packages"],
            "avg_completed_deliveries": round(sum(r["completed_deliveries"] for r in scen_runs) / count, 2),
            "avg_throughput_per_100_steps": round(sum(r["throughput_per_100_steps"] for r in scen_runs) / count, 2),
            "avg_robot_utilization_pct": round(sum(r["robot_utilization_pct"] for r in scen_runs) / count, 2),
            "mean_travel_distance": round(sum(r["avg_travel_distance"] for r in scen_runs) / count, 2),
            "mean_wait_time": round(sum(r["avg_wait_time"] for r in scen_runs) / count, 2),
            "avg_planning_time_ms": round(sum(r["avg_planning_time_ms"] for r in scen_runs) / count, 2),
            "avg_collisions_prevented": round(sum(r["collisions_prevented"] for r in scen_runs) / count, 2),
            "avg_deadlocks_prevented": round(sum(r["deadlocks_prevented"] for r in scen_runs) / count, 2),
        }
        summary_rows.append(avg_entry)

    csv_summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    with open(csv_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Summary benchmark metrics saved to: {csv_summary_path}")

    # 3. Generate Charts using Matplotlib
    _generate_plots(summary_rows, output_dir)

    return raw_results


def _generate_plots(summary_rows: List[Dict[str, Any]], output_dir: str) -> None:
    """Generates and saves performance plot charts."""
    scenarios = [r["scenario"] for r in summary_rows]
    robot_counts = [r["num_robots"] for r in summary_rows]
    throughputs = [r["avg_throughput_per_100_steps"] for r in summary_rows]
    travel_distances = [r["mean_travel_distance"] for r in summary_rows]
    waiting_times = [r["mean_wait_time"] for r in summary_rows]
    planner_runtimes = [r["avg_planning_time_ms"] for r in summary_rows]

    # Chart 1: Throughput vs Robot Count
    plt.figure(figsize=(8, 5))
    plt.plot(robot_counts, throughputs, marker="o", color="blue", linewidth=2)
    plt.title("Delivery Throughput vs. Fleet Robot Count")
    plt.xlabel("Robot Count")
    plt.ylabel("Throughput (deliveries / 100 steps)")
    plt.grid(True)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "throughput_vs_robots.png")
    plt.savefig(plot1_path)
    plt.close()

    # Chart 2: Travel Distance vs Robot Count
    plt.figure(figsize=(8, 5))
    plt.plot(robot_counts, travel_distances, marker="s", color="green", linewidth=2)
    plt.title("Average Robot Travel Distance vs. Fleet Size")
    plt.xlabel("Robot Count")
    plt.ylabel("Average Travel Distance (steps)")
    plt.grid(True)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "travel_distance_vs_robots.png")
    plt.savefig(plot2_path)
    plt.close()

    # Chart 3: Waiting Time vs Robot Count
    plt.figure(figsize=(8, 5))
    plt.plot(robot_counts, waiting_times, marker="^", color="orange", linewidth=2)
    plt.title("Average Robot Waiting Time vs. Fleet Size")
    plt.xlabel("Robot Count")
    plt.ylabel("Average Idle/Waiting Time (steps)")
    plt.grid(True)
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "waiting_time_vs_robots.png")
    plt.savefig(plot3_path)
    plt.close()

    # Chart 4: Planner Runtime vs Robot Count
    plt.figure(figsize=(8, 5))
    plt.plot(robot_counts, planner_runtimes, marker="d", color="red", linewidth=2)
    plt.title("Planner Runtime vs. Fleet Size")
    plt.xlabel("Robot Count")
    plt.ylabel("Average Planning Time (ms)")
    plt.grid(True)
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, "planner_runtime_vs_robots.png")
    plt.savefig(plot4_path)
    plt.close()

    print(f"[OK] Generated plot charts in: {output_dir}")


if __name__ == "__main__":
    run_benchmark_suite()
