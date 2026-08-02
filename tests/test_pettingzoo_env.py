"""Comprehensive test suite for WarehouseParallelEnv and PettingZoo API compliance."""

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from marl.communication import CommunicationManager
from marl.multi_agent_config import MultiAgentEnvConfig
from marl.parallel_env import WarehouseParallelEnv
from simulator.position import Position


def test_pettingzoo_official_parallel_api_compliance() -> None:
    """Validates full compliance with the official PettingZoo parallel API test suite."""
    env = WarehouseParallelEnv()
    parallel_api_test(env, num_cycles=50)
    env.close()


def test_parallel_env_reset_and_joint_step() -> None:
    config = MultiAgentEnvConfig(num_robots=3, seed=42)
    env = WarehouseParallelEnv(config=config)

    obs_dict, info_dict = env.reset(seed=42)
    assert len(env.agents) == 3
    assert set(env.agents) == {"robot_0", "robot_1", "robot_2"}
    assert "robot_0" in obs_dict
    assert "local_occupancy" in obs_dict["robot_0"]

    # Joint step: robot_0 moves East, robot_1 moves East, robot_2 waits
    actions = {"robot_0": 3, "robot_1": 3, "robot_2": 4}
    obs_n, rew_n, term_n, trunc_n, info_n = env.step(actions)

    assert "robot_0" in obs_n
    assert "robot_0" in rew_n
    assert isinstance(rew_n["robot_0"], float)
    env.close()


def test_observation_modes_local_global_hybrid() -> None:
    for mode in ["local", "global", "hybrid"]:
        config = MultiAgentEnvConfig(num_robots=2, observation_mode=mode, seed=42)
        env = WarehouseParallelEnv(config=config)
        obs_dict, _ = env.reset(seed=42)

        if mode in ["global", "hybrid"]:
            assert "global_fleet_summary" in obs_dict["robot_0"]
        else:
            assert "global_fleet_summary" not in obs_dict["robot_0"]

        env.close()


def test_reward_modes_individual_team_hybrid() -> None:
    for mode in ["individual", "team", "hybrid"]:
        config = MultiAgentEnvConfig(num_robots=2, reward_mode=mode, seed=42)
        env = WarehouseParallelEnv(config=config)
        env.reset(seed=42)

        actions = {"robot_0": 3, "robot_1": 4}
        _, rewards, _, _, _ = env.step(actions)

        assert "robot_0" in rewards
        assert "robot_1" in rewards

        if mode == "team":
            assert rewards["robot_0"] == rewards["robot_1"]

        env.close()


def test_communication_modes_none_broadcast_radius() -> None:
    for mode in ["none", "broadcast", "radius"]:
        config = MultiAgentEnvConfig(num_robots=3, comm_mode=mode, comm_radius=5, seed=42)
        env = WarehouseParallelEnv(config=config)
        env.reset(seed=42)

        # Set communication message for robot_0
        msg = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        env.comm_manager.set_message("robot_0", msg)

        rec1 = env.comm_manager.get_received_messages("robot_1", Position(1, 0), env._fleet)
        assert isinstance(rec1, np.ndarray)
        assert rec1.shape == (4,)

        if mode == "broadcast":
            np.testing.assert_array_almost_equal(rec1, msg)
        elif mode == "none":
            np.testing.assert_array_equal(rec1, np.zeros(4, dtype=np.float32))

        env.close()


def test_inter_agent_collision_detection() -> None:
    config = MultiAgentEnvConfig(num_robots=2, seed=42)
    env = WarehouseParallelEnv(config=config)
    env.reset(seed=42)

    # Position robot_0 at (0, 0) and robot_1 at (1, 0)
    env._fleet["robot_0"].position = Position(0, 0)
    env._fleet["robot_1"].position = Position(1, 0)

    # robot_0 moves East (1, 0) and robot_1 moves West (0, 0) -> Swap Collision!
    actions = {"robot_0": 3, "robot_1": 2}
    _, rewards, _, _, infos = env.step(actions)

    assert infos["robot_0"]["is_collision"] is True
    assert infos["robot_1"]["is_collision"] is True
    assert env._fleet["robot_0"].position == Position(0, 0)  # Restored
    assert env._fleet["robot_1"].position == Position(1, 0)  # Restored
    env.close()


def test_global_state_extraction() -> None:
    config = MultiAgentEnvConfig(num_robots=2, seed=42)
    env = WarehouseParallelEnv(config=config)
    env.reset(seed=42)

    global_state = env.state()
    assert isinstance(global_state, np.ndarray)
    assert global_state.shape == (config.grid_height, config.grid_width)
    env.close()
