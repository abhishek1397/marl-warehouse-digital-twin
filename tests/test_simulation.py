"""Integration tests for SimulationEngine lifecycle, steps, and execution."""

import pytest
from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.exceptions import WarehouseError
from simulator.package import Package
from simulator.position import Position
from simulator.robot import Robot
from simulator.shelf import Shelf
from simulator.simulation import SimulationEngine, SimulationState
from simulator.task import TaskPriority, TaskType
from simulator.warehouse import Warehouse


def test_simulation_engine_initialization_and_entities() -> None:
    wh = Warehouse(10, 10)
    engine = SimulationEngine(wh)

    robot = Robot("r1", Position(0, 0))
    engine.add_robot(robot)
    assert "r1" in engine.fleet

    shelf = Shelf("s1", Position(5, 5))
    engine.add_shelf(shelf)

    station = ChargingStation("c1", Position(9, 9))
    engine.add_charging_station(station)

    engine.initialize()
    assert engine.state == SimulationState.READY


def test_simulation_engine_task_execution() -> None:
    wh = Warehouse(10, 10)
    engine = SimulationEngine(wh)

    robot = Robot("r1", Position(0, 0))
    engine.add_robot(robot)

    pkg = Package("p1", Position(2, 0), Position(4, 0))
    engine.add_package(pkg)

    engine.task_manager.create_task(
        "t1", TaskType.PICKUP_AND_DELIVER, Position(2, 0), Position(4, 0), package=pkg
    )

    engine.initialize()
    summary = engine.run(max_steps=50)

    assert engine.state == SimulationState.COMPLETED
    assert summary["completed_deliveries"] == 1
    assert pkg.is_delivered() is True
    assert robot.tasks_completed == 1


def test_simulation_pause_and_reset() -> None:
    wh = Warehouse(10, 10)
    engine = SimulationEngine(wh)

    engine.initialize()
    engine.run(max_steps=5)
    assert engine.current_step == 5

    engine.reset()
    assert engine.current_step == 0
    assert engine.state == SimulationState.UNINITIALIZED
    assert len(engine.fleet) == 0
