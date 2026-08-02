"""CLI entry point for running the deterministic Warehouse Digital Twin simulation."""

import argparse
import json
import os
import sys
from typing import Any, Dict

from simulator.battery_manager import BatteryManager
from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.metrics import MetricsCollector
from simulator.obstacle import Obstacle
from simulator.package import Package
from simulator.position import Position
from simulator.robot import Robot
from simulator.scheduler import (
    FIFOSchedulerStrategy,
    NearestRobotSchedulerStrategy,
    PrioritySchedulerStrategy,
    TaskScheduler,
)
from simulator.shelf import Shelf
from simulator.simulation import SimulationEngine
from simulator.task import TaskPriority, TaskType
from simulator.task_manager import TaskManager
from simulator.visualization import render_ascii_grid
from simulator.warehouse import Warehouse


def load_configuration(config_path: str) -> Dict[str, Any]:
    """Loads warehouse configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_simulation_environment(
    config: Dict[str, Any], scheduler_name: str = "priority"
) -> SimulationEngine:
    """Builds and initializes warehouse entities and simulation engine from configuration."""
    wh_cfg = config.get("warehouse", {})
    width = wh_cfg.get("width", 30)
    height = wh_cfg.get("height", 20)
    name = wh_cfg.get("name", "WarehouseDigitalTwin")

    warehouse = Warehouse(width=width, height=height, name=name)
    task_manager = TaskManager()
    battery_manager = BatteryManager()
    metrics_collector = MetricsCollector()

    # Configure scheduler strategy
    if scheduler_name == "fifo":
        strategy = FIFOSchedulerStrategy()
    elif scheduler_name == "nearest":
        strategy = NearestRobotSchedulerStrategy()
    else:
        strategy = PrioritySchedulerStrategy()

    task_scheduler = TaskScheduler(strategy=strategy)

    engine = SimulationEngine(
        warehouse=warehouse,
        task_manager=task_manager,
        battery_manager=battery_manager,
        metrics_collector=metrics_collector,
        task_scheduler=task_scheduler,
    )

    # 1. Spawn Shelves
    for s_cfg in config.get("shelves", []):
        pos = Position(s_cfg["x"], s_cfg["y"])
        shelf = Shelf(
            shelf_id=s_cfg["shelf_id"],
            position=pos,
            capacity=s_cfg.get("capacity", 10),
        )
        warehouse.place_object(pos, shelf, cell_type=CellType.SHELF)
        engine.add_shelf(shelf)

    # 2. Spawn Obstacles
    for o_cfg in config.get("obstacles", []):
        pos = Position(o_cfg["x"], o_cfg["y"])
        obstacle = Obstacle(
            obstacle_id=o_cfg["obstacle_id"],
            position=pos,
            name=o_cfg.get("name", "Obstacle"),
        )
        warehouse.place_object(pos, obstacle, cell_type=CellType.OBSTACLE)

    # 3. Spawn Charging Stations
    for c_cfg in config.get("charging_stations", []):
        pos = Position(c_cfg["x"], c_cfg["y"])
        station = ChargingStation(
            station_id=c_cfg["station_id"],
            position=pos,
            charge_rate=c_cfg.get("charge_rate", 10.0),
            capacity=c_cfg.get("capacity", 1),
        )
        warehouse.place_object(pos, station, cell_type=CellType.CHARGING_STATION)
        engine.add_charging_station(station)

    # 4. Spawn Robots
    for r_cfg in config.get("robots", []):
        pos = Position(r_cfg["start_x"], r_cfg["start_y"])
        robot = Robot(
            robot_id=r_cfg["robot_id"],
            initial_position=pos,
            max_battery=r_cfg.get("max_battery", 100.0),
        )
        engine.add_robot(robot)

    # 5. Spawn Packages and Tasks
    for idx, p_cfg in enumerate(config.get("packages", []), start=1):
        src_pos = Position(p_cfg["source_x"], p_cfg["source_y"])
        dst_pos = Position(p_cfg["dest_x"], p_cfg["dest_y"])

        pkg = Package(
            package_id=p_cfg["package_id"],
            source_position=src_pos,
            destination_position=dst_pos,
            weight=p_cfg.get("weight", 1.0),
        )
        engine.add_package(pkg)

        task_manager.create_task(
            task_id=f"task_{idx:03d}",
            task_type=TaskType.PICKUP_AND_DELIVER,
            pickup_position=src_pos,
            drop_position=dst_pos,
            package=pkg,
            priority=TaskPriority.MEDIUM,
        )

    engine.initialize()
    return engine


def main() -> None:
    """Main CLI runner entry point."""
    parser = argparse.ArgumentParser(
        description="Deterministic Warehouse Digital Twin Classical Planning Baseline"
    )
    default_cfg = os.path.join(os.path.dirname(__file__), "configs", "warehouse_config.json")
    parser.add_argument(
        "--config",
        type=str,
        default=default_cfg,
        help="Path to JSON warehouse configuration file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum simulation steps to run.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["fifo", "priority", "nearest"],
        default="priority",
        help="Task scheduler strategy to use.",
    )
    parser.add_argument(
        "--ascii-viz",
        action="store_true",
        help="Render ASCII grid view of initial and final simulation state.",
    )

    args = parser.parse_args()

    print(f"Loading warehouse configuration from: {args.config}")
    config = load_configuration(args.config)

    print(f"Initializing Warehouse Digital Twin environment with {args.scheduler.upper()} scheduler...")
    engine = build_simulation_environment(config, scheduler_name=args.scheduler)

    print(
        f"Created Warehouse '{engine.warehouse.name}' ({engine.warehouse.width}x{engine.warehouse.height}) "
        f"with {len(engine.fleet)} robots and {engine.task_manager.total_tasks} tasks."
    )

    if args.ascii_viz:
        print("\nInitial Warehouse Layout:")
        print(render_ascii_grid(engine.warehouse, fleet=engine.fleet))

    print(f"\nExecuting classical multi-robot motion planning simulation for up to {args.steps} steps...")
    engine.run(max_steps=args.steps)

    print(f"\nSimulation finished at step {engine.current_step}. Final state: {engine.state.name}")

    if args.ascii_viz:
        print("\nFinal Warehouse Layout:")
        print(render_ascii_grid(engine.warehouse, fleet=engine.fleet))

    completed_count = len(engine.task_manager.get_completed_tasks())
    engine.metrics_collector.print_summary(
        engine.fleet, completed_count, engine.current_step
    )


if __name__ == "__main__":
    main()
