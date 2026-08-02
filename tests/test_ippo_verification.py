"""Comprehensive test suite for research diagnostic verifiers."""

import os
import pytest

from research.buffer_verifier import BufferVerifier
from research.environment_sync_verifier import EnvironmentSyncVerifier
from research.gae_verifier import GAEVerifier
from research.observation_verifier import ObservationVerifier
from research.pettingzoo_api_verifier import PettingZooAPIVerifier
from research.policy_sync_verifier import PolicySyncVerifier
from research.reward_verifier import RewardVerifier
from research.rollout_verifier import RolloutVerifier
from research.trainer_verifier import SingleAgentEquivalenceVerifier


def test_pettingzoo_api_verifier() -> None:
    res = PettingZooAPIVerifier.verify_api_compliance()
    assert res["status"] == "PASSED"
    assert res["passed"] is True


def test_observation_verifier() -> None:
    res = ObservationVerifier.verify_observations()
    assert res["status"] == "PASSED"


def test_reward_verifier() -> None:
    res = RewardVerifier.verify_reward_assignment()
    assert res["status"] == "PASSED"


def test_rollout_verifier() -> None:
    res = RolloutVerifier.verify_rollout_collection()
    assert res["status"] == "PASSED"


def test_gae_verifier() -> None:
    res = GAEVerifier.verify_gae_calculation()
    assert res["status"] == "PASSED"


def test_buffer_verifier() -> None:
    res = BufferVerifier.verify_buffer_isolation()
    assert res["status"] == "PASSED"


def test_policy_sync_verifier() -> None:
    res = PolicySyncVerifier.verify_policy_updates()
    assert res["status"] == "PASSED"


def test_environment_sync_verifier() -> None:
    res = EnvironmentSyncVerifier.compare_trajectories(num_steps=5)
    assert "status" in res


def test_single_agent_equivalence_verifier() -> None:
    res = SingleAgentEquivalenceVerifier.verify_single_agent_equivalence(timesteps=100)
    assert "status" in res
