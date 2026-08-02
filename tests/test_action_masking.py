"""Comprehensive test suite for marl/action_masking package."""

import pytest
import numpy as np
import torch
from torch.distributions import Categorical

from marl import EnvConfig, WarehouseGymEnv
from marl.action_masking import (
    ActionMask,
    ActionMaskConfig,
    ActionMaskGenerator,
    ActionMaskValidator,
    MaskedPolicyWrapper,
    calculate_mask_entropy,
    compute_mask_utilization,
    format_mask_visualization,
)
from marl.networks.policy_network import PolicyNetwork
from simulator.charging_station import ChargingStation
from simulator.grid import Grid
from simulator.package import Package
from simulator.position import Position
from simulator.robot import Robot
from simulator.task import Task, TaskType
from simulator.warehouse import Warehouse


def test_action_mask_config() -> None:
    cfg = ActionMaskConfig(enable_action_masking=True, strict_masking=True)
    assert cfg.enable_action_masking is True
    assert cfg.strict_masking is True
    assert cfg.mask_invalid_moves is True


def test_action_mask_dataclass() -> None:
    arr = np.array([True, True, False, False, True, False, False, False])
    mask = ActionMask(mask_array=arr, mask_tensor=torch.tensor(arr))

    assert mask.num_valid == 3
    assert mask.valid_indices == [0, 1, 4]
    assert mask.mask_entropy > 0.0


def test_mask_generator_rules() -> None:
    warehouse = Warehouse(width=10, height=10)
    generator = ActionMaskGenerator()

    # 1. Corner Robot (0,0) -> Up (0) and Left (2) are invalid
    robot = Robot("r0", initial_position=Position(0, 0))
    mask = generator.generate_mask(robot=robot, warehouse=warehouse)

    assert mask.mask_array[0] is np.bool_(False)  # Up invalid
    assert mask.mask_array[2] is np.bool_(False)  # Left invalid
    assert mask.mask_array[4] is np.bool_(True)   # Wait valid
    assert mask.mask_array[5] is np.bool_(False)  # Pick invalid (no task)
    assert mask.mask_array[6] is np.bool_(False)  # Drop invalid (no package)
    assert mask.mask_array[7] is np.bool_(False)  # Charge invalid (no station)

    # 2. Pick Action Valid when sitting at pickup position
    task = Task("t0", TaskType.PICKUP_AND_DELIVER, pickup_position=Position(0, 0), drop_position=Position(5, 5))
    mask_pick = generator.generate_mask(robot=robot, warehouse=warehouse, task=task)
    assert mask_pick.mask_array[5] is np.bool_(True)

    # 3. Drop Action Valid when carrying package and at drop position
    robot_drop = Robot("r0", initial_position=Position(5, 5))
    pkg = Package("p0", source_position=Position(0, 0), destination_position=Position(5, 5))
    robot_drop.pick_up_package(pkg)
    mask_drop = generator.generate_mask(robot=robot_drop, warehouse=warehouse, task=task)
    assert mask_drop.mask_array[6] is np.bool_(True)

    # 4. Charge Action Valid when at charging station
    cs = ChargingStation("cs0", position=Position(0, 0))
    robot.battery_level = 50.0
    mask_cs = generator.generate_mask(robot=robot, warehouse=warehouse, charging_stations={"cs0": cs})
    assert mask_cs.mask_array[7] is np.bool_(True)


def test_mask_validator() -> None:
    arr_valid = np.array([True, False, False, False, True, False, False, False])
    mask_valid = ActionMask(mask_array=arr_valid, mask_tensor=torch.tensor(arr_valid))
    assert ActionMaskValidator.validate_mask(mask_valid) is True

    # Empty mask error
    arr_empty = np.zeros(8, dtype=bool)
    mask_empty = ActionMask(mask_array=arr_empty, mask_tensor=torch.tensor(arr_empty))
    with pytest.raises(ValueError, match="No valid actions"):
        ActionMaskValidator.validate_mask(mask_empty)

    # Wait invalid error
    arr_no_wait = np.array([True, True, False, False, False, False, False, False])
    mask_no_wait = ActionMask(mask_array=arr_no_wait, mask_tensor=torch.tensor(arr_no_wait))
    with pytest.raises(ValueError, match="Wait"):
        ActionMaskValidator.validate_mask(mask_no_wait)


def test_masked_policy_wrapper_math() -> None:
    raw_logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    mask = torch.tensor([True, False, True, False, True, False, False, False], dtype=torch.bool)

    masked_logits = MaskedPolicyWrapper.apply_mask(raw_logits, mask)
    assert masked_logits[1].item() == -1e9
    assert masked_logits[3].item() == -1e9
    assert masked_logits[0].item() == 1.0

    dist = MaskedPolicyWrapper.get_masked_distribution(raw_logits, mask)
    probs = dist.probs
    assert probs[1].item() == 0.0
    assert probs[3].item() == 0.0
    assert probs[5].item() == 0.0
    assert probs[0].item() > 0.0


def test_utils_and_visualization() -> None:
    mask = torch.tensor([True, True, False, False, True, False, False, False], dtype=torch.bool)
    ent = calculate_mask_entropy(mask)
    util = compute_mask_utilization(mask)

    assert ent > 0.0
    assert util == 62.5  # 5 out of 8 masked

    raw = torch.randn(8)
    m_logits = MaskedPolicyWrapper.apply_mask(raw, mask)
    viz = format_mask_visualization(raw, m_logits, mask, selected_action=0)
    assert "Action Masking Visualization" in viz


def test_warehouse_gym_env_integration_dam() -> None:
    env_cfg = EnvConfig(grid_width=6, grid_height=6, enable_action_masking=True)
    env = WarehouseGymEnv(config=env_cfg)

    obs, info = env.reset(seed=42)
    assert "action_mask" in info
    assert len(info["action_mask"]) == 8

    obs, reward, terminated, truncated, info = env.step(4)  # Wait
    assert "action_mask" in info
    env.close()
