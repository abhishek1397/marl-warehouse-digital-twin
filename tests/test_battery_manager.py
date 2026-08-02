"""Unit tests for BatteryManager drain rates, charging, and thresholds."""

from simulator.battery_manager import BatteryManager
from simulator.package import Package
from simulator.position import Position
from simulator.robot import Robot, RobotState


def test_battery_manager_idle_drain() -> None:
    bm = BatteryManager(idle_drain_rate=1.0)
    robot = Robot("r1", Position(0, 0), max_battery=100.0)

    bm.update_robot_battery(robot)
    assert robot.battery_level == 99.0
    assert robot.idle_steps == 1


def test_battery_manager_movement_and_loaded_drain() -> None:
    bm = BatteryManager(move_drain_rate=2.0, loaded_drain_rate=5.0)
    robot = Robot("r1", Position(0, 0), max_battery=100.0)
    robot.state = RobotState.MOVING_TO_PICKUP

    bm.update_robot_battery(robot)
    assert robot.battery_level == 98.0

    pkg = Package("p1", Position(0, 0), Position(2, 2))
    robot.pick_up_package(pkg)
    bm.update_robot_battery(robot)
    assert robot.battery_level == 93.0


def test_battery_manager_charging_and_thresholds() -> None:
    bm = BatteryManager(charging_rate=10.0, low_battery_threshold=20.0, critical_battery_threshold=5.0)
    robot = Robot("r1", Position(0, 0), max_battery=100.0)
    robot.battery_level = 15.0

    assert bm.is_low_battery(robot) is True
    assert bm.is_critical_battery(robot) is False

    robot.battery_level = 3.0
    assert bm.is_critical_battery(robot) is True

    robot.state = RobotState.CHARGING
    bm.update_robot_battery(robot)
    assert robot.battery_level == 13.0
    assert robot.charging_steps == 1
