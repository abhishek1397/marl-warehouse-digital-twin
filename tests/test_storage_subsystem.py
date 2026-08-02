"""Comprehensive test suite for marl/storage trajectory, rollout buffer, and GAE subsystem."""

import os
import pytest
import numpy as np
import torch

from marl.storage import (
    Batch,
    BufferStatistics,
    MiniBatchSampler,
    RolloutBuffer,
    Trajectory,
    Transition,
    compute_bootstrapped_returns,
    compute_discounted_returns,
    compute_gae,
    compute_gae_reference,
    compute_mask,
    compute_mc_returns,
    convert_obs_to_tensor,
    normalize_advantages,
    normalize_observations,
    normalize_rewards,
    stack_observations,
)


def test_transition_and_trajectory() -> None:
    obs = np.array([1.0, 2.0])
    trans1 = Transition(observation=obs, action=0, reward=1.0, timestep=0)
    trans2 = Transition(observation=obs, action=1, reward=2.0, terminated=True, timestep=1)

    assert trans1.done is False
    assert trans2.done is True

    traj = Trajectory(agent_id="robot_0", episode_id=1)
    traj.append(trans1)
    traj.append(trans2)

    assert traj.compute_episode_length() == 2
    assert traj.compute_return() == 3.0

    stats = traj.statistics()
    assert stats["length"] == 2.0
    assert stats["total_reward"] == 3.0

    ser = traj.serialize()
    assert ser["agent_id"] == "robot_0"


def test_gae_vectorized_vs_reference_equivalence() -> None:
    T = 10
    rewards = torch.tensor([1.0] * T, dtype=torch.float32)
    values = torch.tensor([0.5] * T, dtype=torch.float32)
    next_values = torch.tensor([0.5] * T, dtype=torch.float32)
    dones = torch.tensor([False] * T, dtype=torch.bool)
    dones[4] = True
    dones[9] = True

    adv_vec, ret_vec = compute_gae(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95)
    adv_ref, ret_ref = compute_gae_reference(rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.95)

    torch.testing.assert_close(adv_vec, adv_ref)
    torch.testing.assert_close(ret_vec, ret_ref)


def test_return_computations() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    dones = torch.tensor([False, False, True], dtype=torch.bool)

    disc_ret = compute_discounted_returns(rewards, dones, gamma=0.9)
    assert disc_ret[2] == 1.0
    assert abs(disc_ret[1].item() - (1.0 + 0.9 * 1.0)) < 1e-5

    trans1 = Transition(observation=np.zeros(2), action=0, reward=1.0)
    trans2 = Transition(observation=np.zeros(2), action=1, reward=2.0, terminated=True)
    traj = Trajectory(transitions=[trans1, trans2])

    mc_rets = compute_mc_returns([traj], gamma=1.0)
    assert len(mc_rets) == 1
    assert mc_rets[0][0] == 3.0

    boot_ret = compute_bootstrapped_returns(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(boot_ret, torch.tensor([1.5, 2.5]))


def test_normalizations() -> None:
    advs = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    norm_advs = normalize_advantages(advs)
    assert abs(norm_advs.mean().item()) < 1e-5

    # Single element edge case
    single_adv = torch.tensor([1.0])
    assert torch.equal(normalize_advantages(single_adv), single_adv)

    norm_rews = normalize_rewards(torch.tensor([10.0]), running_mean=0.0, running_std=2.0)
    assert norm_rews[0].item() == 5.0

    norm_obs = normalize_observations(torch.tensor([10.0]), mean=torch.tensor([0.0]), std=torch.tensor([2.0]))
    assert norm_obs[0].item() == 5.0


def test_sampler() -> None:
    sampler = MiniBatchSampler(dataset_size=100, mini_batch_size=32, num_epochs=2, shuffle=True)
    indices_list = list(sampler.sample_indices())
    assert len(indices_list) == 8  # 4 batches per epoch * 2 epochs
    assert indices_list[0].shape[0] == 32

    # Empty sampler edge case
    empty_sampler = MiniBatchSampler(dataset_size=0, mini_batch_size=10)
    assert list(empty_sampler.sample_indices()) == []


def test_buffer_utils() -> None:
    obs_arr = np.array([1.0, 2.0])
    t_obs = convert_obs_to_tensor(obs_arr)
    assert isinstance(t_obs, torch.Tensor)

    stacked = stack_observations([t_obs, t_obs])
    assert stacked.shape == (2, 2)

    dict_obs = {"pos": np.array([1.0, 2.0])}
    t_dict_obs = convert_obs_to_tensor(dict_obs)
    stacked_dict = stack_observations([t_dict_obs, t_dict_obs])
    assert stacked_dict["pos"].shape == (2, 2)

    assert compute_mask(terminated=True, truncated=False) == 0.0
    assert compute_mask(terminated=False, truncated=True) == 1.0

    with pytest.raises(TypeError):
        convert_obs_to_tensor("invalid_type")
    with pytest.raises(ValueError):
        stack_observations([])


def test_rollout_buffer_single_and_multi_agent() -> None:
    buffer = RolloutBuffer(capacity=5, device="cpu")
    assert len(buffer) == 0

    obs = np.array([1.0, 2.0, 3.0])
    for i in range(6):  # Overflow capacity 5
        trans = Transition(
            observation=obs,
            action=i % 8,
            reward=float(i),
            value_estimate=0.5,
            log_prob=-0.1,
            agent_id=f"robot_{i % 2}",
            timestep=i,
        )
        buffer.insert(trans)

    assert len(buffer) == 5

    buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    assert buffer.advantages is not None
    assert buffer.returns is not None

    stats = buffer.get_statistics()
    assert stats["num_transitions"] == 5.0

    # Mini-batch generation
    batches = list(buffer.get_generator(mini_batch_size=2, num_epochs=1))
    assert len(batches) == 3
    assert isinstance(batches[0], Batch)
    assert batches[0].observations.shape[0] <= 2

    # Joint insertion
    joint_trans = {
        "robot_0": Transition(observation=obs, action=0, reward=1.0),
        "robot_1": Transition(observation=obs, action=1, reward=2.0),
    }
    buffer.clear()
    buffer.insert_joint(joint_trans)
    assert len(buffer) == 2

    buffer.reset()
    assert len(buffer) == 0


def test_batch_device_and_dict_conversion() -> None:
    obs = torch.randn(4, 2)
    b = Batch(
        observations=obs,
        actions=torch.zeros(4),
        advantages=torch.zeros(4),
        returns=torch.zeros(4),
        values=torch.zeros(4),
        old_log_probs=torch.zeros(4),
        masks=torch.ones(4),
    )

    b_dev = b.to_device(torch.device("cpu"))
    assert b_dev.observations.device.type == "cpu"

    d = b.to_dict()
    assert "observations" in d
