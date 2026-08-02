"""Comprehensive test suite for research behavior analysis framework."""

import os
import pytest
import numpy as np

from marl import EnvConfig, WarehouseGymEnv
from research.action_statistics import ActionStatisticsAnalyzer
from research.episode_visualizer import EpisodeAnimationExporter
from research.evaluation_report import EvaluationReportGenerator
from research.failure_analyzer import FailureClassifier, FailureMode
from research.policy_evaluator import PolicyEvaluator
from research.reward_breakdown import RewardDecompositionAnalyzer
from research.success_analyzer import SuccessMetricsAnalyzer
from research.trajectory_recorder import EpisodeTrajectory, TrajectoryRecorder, TrajectoryStep
from research.trajectory_visualizer import TrajectoryVisualizer
from simulator.position import Position


def test_trajectory_recorder() -> None:
    recorder = TrajectoryRecorder()
    recorder.start_episode(episode_id=1)

    recorder.record_step(
        timestep=1,
        position=Position(0, 0),
        action=3,
        reward=1.0,
        env_reward=-0.1,
        potential_reward=1.1,
        battery_level=99.0,
        carrying_package=False,
        goal_position=Position(5, 5),
        task_status="ASSIGNED",
        is_collision=False,
        is_pickup=True,
        is_delivery=False,
    )

    traj = recorder.finish_episode()
    assert isinstance(traj, EpisodeTrajectory)
    assert traj.episode_id == 1
    assert traj.total_reward == 1.0
    assert traj.total_pickups == 1
    assert traj.total_collisions == 0


def test_action_statistics_analyzer() -> None:
    recorder = TrajectoryRecorder()
    recorder.start_episode(1)
    recorder.record_step(1, Position(0, 0), action=0, reward=-1.0)
    recorder.record_step(2, Position(0, 1), action=4, reward=-1.0)
    traj = recorder.finish_episode()

    stats = ActionStatisticsAnalyzer.compute_action_frequencies([traj])
    assert "Move Up" in stats
    assert stats["wait_pct"] == 50.0
    assert stats["action_entropy"] > 0.0


def test_reward_decomposition_analyzer() -> None:
    recorder = TrajectoryRecorder()
    recorder.start_episode(1)
    recorder.record_step(1, Position(0, 0), action=0, reward=2.0, env_reward=1.0, potential_reward=1.0)
    traj = recorder.finish_episode()

    breakdown = RewardDecompositionAnalyzer.compute_reward_breakdown([traj])
    assert breakdown["mean_env_reward"] == 1.0
    assert breakdown["mean_potential_reward"] == 1.0
    assert breakdown["mean_total_reward"] == 2.0


def test_failure_classifier() -> None:
    # 1. Success
    t_succ = EpisodeTrajectory(episode_id=1, is_success=True)
    assert FailureClassifier.classify_episode(t_succ) == FailureMode.SUCCESS

    # 2. Collision Failure
    t_coll = EpisodeTrajectory(episode_id=2, total_collisions=5)
    assert FailureClassifier.classify_episode(t_coll) == FailureMode.EXCESSIVE_COLLISIONS

    # 3. Timeout Failure
    t_time = EpisodeTrajectory(episode_id=3)
    t_time.steps.append(TrajectoryStep(1, Position(0, 0), 0, -1.0, -1.0, 0.0, 100.0, False, None, "IDLE", False, False, False))
    assert FailureClassifier.classify_episode(t_time) == FailureMode.NEVER_FOUND_PACKAGE


def test_success_metrics_analyzer() -> None:
    t1 = EpisodeTrajectory(episode_id=1, is_success=True, episode_length=20)
    t2 = EpisodeTrajectory(episode_id=2, is_success=False, episode_length=50)

    metrics = SuccessMetricsAnalyzer.compute_success_metrics([t1, t2])
    assert metrics["success_rate"] == 0.5
    assert metrics["mean_completion_time"] == 20.0


def test_policy_evaluator_and_visualizers(tmp_path) -> None:
    env_cfg = EnvConfig(grid_width=6, grid_height=6)
    env = WarehouseGymEnv(config=env_cfg)

    evaluator = PolicyEvaluator(env=env, policy=None)
    trajs = evaluator.evaluate_policy(num_episodes=2, seed=42)

    assert len(trajs) == 2
    assert isinstance(trajs[0], EpisodeTrajectory)

    # Test visualizers
    plot_path = os.path.join(tmp_path, "traj_test.png")
    TrajectoryVisualizer.plot_trajectory_path(trajs[0], 6, 6, plot_path)
    assert os.path.exists(plot_path)

    frame_path = os.path.join(tmp_path, "frame_test.png")
    EpisodeAnimationExporter.render_frame(trajs[0], 0, 6, 6, frame_path)
    assert os.path.exists(frame_path)

    # Test report generator
    report_path = os.path.join(tmp_path, "report_test.md")
    EvaluationReportGenerator.generate_markdown_report({"comparative_matrix": {}}, report_path)
    assert os.path.exists(report_path)

    env.close()
