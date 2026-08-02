"""Comprehensive test suite for marl/reward_shaping package."""

import pytest
import numpy as np

from marl import EnvConfig, WarehouseGymEnv
from marl.action import ActionResult
from marl.reward import RewardEngine
from marl.reward_shaping import (
    ManhattanPotential,
    RewardShapingConfig,
    ShapedRewardEngine,
    ShapedRewardOutput,
    calculate_goal_progress,
    calculate_shaping_reward,
    chebyshev_distance,
    euclidean_distance,
    manhattan_distance,
)
from simulator.charging_station import ChargingStation
from simulator.package import Package
from simulator.position import Position
from simulator.robot import Robot
from simulator.task import Task, TaskType


def test_distance_metrics() -> None:
    p1 = Position(0, 0)
    p2 = Position(3, 4)

    assert manhattan_distance(p1, p2) == 7.0
    assert euclidean_distance(p1, p2) == 5.0
    assert chebyshev_distance(p1, p2) == 4.0

    # Tuple and list inputs
    assert manhattan_distance((0, 0), [3, 4]) == 7.0


def test_reward_shaping_config() -> None:
    cfg = RewardShapingConfig(enable_reward_shaping=True, shaping_scale=2.0)
    assert cfg.enable_reward_shaping is True
    assert cfg.shaping_scale == 2.0
    assert cfg.gamma == 0.99


def test_manhattan_potential_goal_switching() -> None:
    potential = ManhattanPotential(low_battery_threshold=20.0)

    robot = Robot("r0", initial_position=Position(0, 0))
    robot.battery_level = 100.0

    # 1. Idle -> Goal is self position (distance 0 -> potential 0)
    assert potential.compute_potential(robot) == 0.0

    # 2. Unladen with active task -> Goal is pickup position
    task = Task("t0", TaskType.PICKUP_AND_DELIVER, pickup_position=Position(5, 5), drop_position=Position(10, 10))
    goal = potential.get_active_goal(robot, task=task)
    assert goal == Position(5, 5)
    assert potential.compute_potential(robot, task=task) == -10.0

    # 3. Carrying package -> Goal is drop position
    pkg = Package("p0", source_position=Position(5, 5), destination_position=Position(10, 10), weight=1.0)
    robot.pick_up_package(pkg)
    goal_drop = potential.get_active_goal(robot, task=task)
    assert goal_drop == Position(10, 10)
    assert potential.compute_potential(robot, task=task) == -20.0

    # 4. Low battery override -> Goal is nearest charging station
    robot.battery_level = 10.0
    stations = {"cs0": ChargingStation("cs0", position=Position(1, 1))}
    goal_cs = potential.get_active_goal(robot, task=task, charging_stations=stations)
    assert goal_cs == Position(1, 1)
    assert potential.compute_potential(robot, task=task, charging_stations=stations) == -2.0


def test_shaping_utilities() -> None:
    f_shaping = calculate_shaping_reward(phi_current=-10.0, phi_next=-8.0, gamma=0.99, scale=1.0)
    # F = 0.99 * (-8.0) - (-10.0) = -7.92 + 10.0 = 2.08
    assert abs(f_shaping - 2.08) < 1e-5

    progress = calculate_goal_progress(dist_current=10.0, dist_next=8.0)
    assert progress == 2.0


def test_shaped_reward_engine() -> None:
    env_cfg = EnvConfig()
    base_engine = RewardEngine(env_cfg)

    # Disabled shaping
    cfg_disabled = RewardShapingConfig(enable_reward_shaping=False)
    engine_dis = ShapedRewardEngine(base_reward_engine=base_engine, config=cfg_disabled)

    robot = Robot("r0", initial_position=Position(0, 0))
    action_res = ActionResult(action=0, is_valid=True)

    out_dis = engine_dis.calculate_reward(action_res, robot, robot)
    assert isinstance(out_dis, ShapedRewardOutput)
    assert out_dis.potential_reward == 0.0

    # Enabled shaping
    cfg_enabled = RewardShapingConfig(enable_reward_shaping=True, gamma=0.99)
    engine_en = ShapedRewardEngine(base_reward_engine=base_engine, config=cfg_enabled)

    robot_prev = Robot("r0", initial_position=Position(0, 0))
    robot_next = Robot("r0", initial_position=Position(1, 0))
    task = Task("t0", TaskType.PICKUP_AND_DELIVER, pickup_position=Position(5, 0), drop_position=Position(10, 0))

    out_en = engine_en.calculate_reward(action_res, robot_prev, robot_next, task=task)
    assert out_en.potential_reward > 0.0  # Moved closer to (5,0)
    assert out_en.total_reward == out_en.env_reward + out_en.potential_reward


def test_warehouse_gym_env_integration_pbrs() -> None:
    env_cfg = EnvConfig(grid_width=6, grid_height=6, enable_reward_shaping=True, shaping_scale=1.0)
    env = WarehouseGymEnv(config=env_cfg)

    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(reward, float)
    assert not np.isnan(reward)
    env.close()
