"""Comprehensive test suite for WarehouseGymEnv and marl package."""

import gymnasium as gym
import numpy as np
import pytest

from marl.action import ActionMapper, ActionResult
from marl.config import EnvConfig
from marl.environment import WarehouseGymEnv
from marl.episode import EpisodeManager
from marl.observation import ObservationEncoder
from marl.rendering import EnvironmentRenderer
from marl.reward import RewardEngine
from marl.spaces import get_action_space, get_observation_space
from marl.utils import set_seed
from simulator.position import Position
from simulator.robot import Robot, RobotState


def test_gym_registration_and_make() -> None:
    env = gym.make("Warehouse-v0")
    assert isinstance(env.unwrapped, WarehouseGymEnv)
    obs, info = env.reset(seed=42)
    assert "robot_position" in obs
    env.close()


def test_env_reset_and_observation_space() -> None:
    config = EnvConfig(grid_width=10, grid_height=10, seed=42)
    env = WarehouseGymEnv(config=config)

    obs, info = env.reset(seed=42)
    assert env.action_space.n == 8
    assert isinstance(obs["robot_position"], np.ndarray)
    assert obs["robot_position"].shape == (2,)
    assert obs["battery_level"].shape == (1,)
    assert obs["local_occupancy"].shape == (7, 7)  # 2*R+1 = 7 for R=3
    assert info["step"] == 0
    env.close()


def test_deterministic_seeding() -> None:
    config1 = EnvConfig(seed=123)
    env1 = WarehouseGymEnv(config=config1)
    obs1, _ = env1.reset(seed=123)

    config2 = EnvConfig(seed=123)
    env2 = WarehouseGymEnv(config=config2)
    obs2, _ = env2.reset(seed=123)

    np.testing.assert_array_equal(obs1["robot_position"], obs2["robot_position"])
    np.testing.assert_array_equal(obs1["local_occupancy"], obs2["local_occupancy"])
    env1.close()
    env2.close()


def test_action_execution_and_step() -> None:
    config = EnvConfig(grid_width=10, grid_height=10, seed=42)
    env = WarehouseGymEnv(config=config)
    obs, _ = env.reset(seed=42)

    # Action 3: Move East (1, 0)
    obs_next, reward, terminated, truncated, info = env.step(3)
    assert info["action_valid"] is True
    assert reward < 0.0  # Step penalty
    np.testing.assert_array_equal(obs_next["robot_position"], np.array([1, 0]))

    # Action 4: Wait
    obs_wait, reward_wait, _, _, _ = env.step(4)
    assert reward_wait == config.step_time_penalty + config.waiting_penalty
    env.close()


def test_invalid_action_and_collision_penalties() -> None:
    config = EnvConfig(grid_width=10, grid_height=10, seed=42)
    env = WarehouseGymEnv(config=config)
    env.reset(seed=42)

    # Move North from (0,0) -> Out of Bounds collision
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["action_valid"] is False
    assert reward == config.step_time_penalty + config.collision_penalty

    # Invalid Pick action when not carrying / not at pickup
    obs2, reward2, _, _, info2 = env.step(5)
    assert info2["action_valid"] is False
    assert reward2 == config.step_time_penalty + config.invalid_action_penalty

    # Invalid action out of range
    res_out = env.action_mapper.execute_action(99, env.robot, env.warehouse)
    assert res_out.is_valid is False
    env.close()


def test_package_pickup_and_delivery_reward_flow() -> None:
    config = EnvConfig(grid_width=10, grid_height=10, seed=42)
    env = WarehouseGymEnv(config=config)
    env.reset(seed=42)

    # Move robot to pickup position (5, 2)
    env.robot.position = Position(5, 2)

    # Action 5: Pick package
    obs_pick, reward_pick, _, _, info_pick = env.step(5)
    assert info_pick["action_valid"] is True
    assert reward_pick == config.package_pickup_reward + config.step_time_penalty

    # Action 5 again: Already carrying package -> invalid
    _, reward_pick_again, _, _, info_again = env.step(5)
    assert info_again["action_valid"] is False

    # Move robot to drop destination (9, 9)
    env.robot.position = Position(9, 9)

    # Action 6: Drop package
    obs_drop, reward_drop, _, _, info_drop = env.step(6)
    assert info_drop["action_valid"] is True
    assert reward_drop == config.successful_delivery_reward + config.step_time_penalty

    # Action 6 again: Not carrying package -> invalid
    _, reward_drop_again, _, _, _ = env.step(6)
    assert reward_drop_again == config.step_time_penalty + config.invalid_action_penalty
    env.close()


def test_charging_station_docking_flow() -> None:
    config = EnvConfig(grid_width=10, grid_height=10, seed=42)
    env = WarehouseGymEnv(config=config)
    env.reset(seed=42)

    # Charger is placed at (0, 9)
    # Action 7 at (0, 0) -> invalid (no charger at current pos)
    _, reward_no_charger, _, _, info1 = env.step(7)
    assert info1["action_valid"] is False

    # Move robot to charger position (0, 9)
    env.robot.position = Position(0, 9)

    # Action 7: Dock at charging station
    _, reward_charge, _, _, info2 = env.step(7)
    assert info2["action_valid"] is True
    assert reward_charge == config.successful_charging_reward + config.step_time_penalty
    env.close()


def test_rendering_modes() -> None:
    config = EnvConfig(render_mode="human", seed=42)
    env = WarehouseGymEnv(config=config)
    env.reset(seed=42)

    human_render = env.render()
    assert isinstance(human_render, str)
    assert "+" in human_render

    env.render_mode = "rgb_array"
    rgb_render = env.render()
    assert isinstance(rgb_render, np.ndarray)
    assert rgb_render.ndim == 3
    assert rgb_render.shape[2] == 3
    env.close()


def test_episode_truncation_timeout() -> None:
    config = EnvConfig(max_episode_steps=5, seed=42)
    env = WarehouseGymEnv(config=config)
    env.reset(seed=42)

    truncated_flag = False
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(4)  # Wait step
        if truncated:
            truncated_flag = True
            break

    assert truncated_flag is True
    env.close()


def test_battery_empty_penalty() -> None:
    config = EnvConfig(seed=42)
    env = WarehouseGymEnv(config=config)
    env.reset(seed=42)

    # Force robot battery to 0
    env.robot.battery_level = 0.0
    _, reward, terminated, _, info = env.step(4)

    assert terminated is True
    assert info["is_battery_empty"] is True
    assert reward <= config.battery_empty_penalty
    env.close()
