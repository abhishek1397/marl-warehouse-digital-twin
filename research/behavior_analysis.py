"""Behavior Analysis Runner executing comprehensive 4-policy comparative evaluation."""

import json
import os
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl import EnvConfig, WarehouseGymEnv
from marl.algorithms.ppo import PPOConfig, PPOTrainer
from research.action_statistics import ActionStatisticsAnalyzer
from research.episode_visualizer import EpisodeAnimationExporter
from research.evaluation_report import EvaluationReportGenerator
from research.failure_analyzer import FailureClassifier
from research.policy_evaluator import PolicyEvaluator
from research.reward_breakdown import RewardDecompositionAnalyzer
from research.success_analyzer import SuccessMetricsAnalyzer
from research.trajectory_visualizer import TrajectoryVisualizer


def run_behavior_analysis_framework() -> Dict[str, Any]:
    """Executes comparative evaluation across 4 policy arms and generates behavioral analytics."""
    print("=" * 75)
    print("     PPO POLICY EVALUATION & BEHAVIOR ANALYSIS FRAMEWORK")
    print("=" * 75)

    seed = 42
    num_episodes = 5
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)
    docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs/POLICY_BEHAVIOR_ANALYSIS.md"))

    # Arm 1: Random Policy
    print("\n[1/4] Evaluating Arm 1: Random Policy Baseline...")
    env1 = WarehouseGymEnv(config=EnvConfig(grid_width=8, grid_height=8, seed=seed, enable_reward_shaping=False, enable_action_masking=False))
    evaluator1 = PolicyEvaluator(env=env1, policy=None)
    trajs1 = evaluator1.evaluate_policy(num_episodes=num_episodes, seed=seed)
    metrics1 = SuccessMetricsAnalyzer.compute_success_metrics(trajs1)
    env1.close()
    print(f"      Arm 1 Mean Reward: {metrics1['mean_completion_time']:.1f} steps | Collisions: {metrics1['mean_collisions']:.1f}")

    # Arm 2: Baseline PPO
    print("\n[2/4] Evaluating Arm 2: Baseline PPO...")
    env2 = WarehouseGymEnv(config=EnvConfig(grid_width=8, grid_height=8, seed=seed, enable_reward_shaping=False, enable_action_masking=False))
    trainer2 = PPOTrainer(env=env2, config=PPOConfig(learning_rate=3e-4, epochs=3, batch_size=300, mini_batch_size=64, seed=seed))
    trainer2.train(total_timesteps=3000)
    evaluator2 = PolicyEvaluator(env=env2, policy=trainer2.policy)
    trajs2 = evaluator2.evaluate_policy(num_episodes=num_episodes, seed=seed)
    metrics2 = SuccessMetricsAnalyzer.compute_success_metrics(trajs2)
    env2.close()
    print(f"      Arm 2 Mean Reward: {metrics2['mean_completion_time']:.1f} steps | Collisions: {metrics2['mean_collisions']:.1f}")

    # Arm 3: PPO + PBRS
    print("\n[3/4] Evaluating Arm 3: PPO + PBRS...")
    env3 = WarehouseGymEnv(config=EnvConfig(grid_width=8, grid_height=8, seed=seed, enable_reward_shaping=True, enable_action_masking=False))
    trainer3 = PPOTrainer(env=env3, config=PPOConfig(learning_rate=3e-4, epochs=3, batch_size=300, mini_batch_size=64, seed=seed))
    trainer3.train(total_timesteps=3000)
    evaluator3 = PolicyEvaluator(env=env3, policy=trainer3.policy)
    trajs3 = evaluator3.evaluate_policy(num_episodes=num_episodes, seed=seed)
    metrics3 = SuccessMetricsAnalyzer.compute_success_metrics(trajs3)
    env3.close()
    print(f"      Arm 3 Mean Reward: {metrics3['mean_completion_time']:.1f} steps | Collisions: {metrics3['mean_collisions']:.1f}")

    # Arm 4: PPO + PBRS + DAM
    print("\n[4/4] Evaluating Arm 4: PPO + PBRS + Dynamic Action Masking...")
    env4 = WarehouseGymEnv(config=EnvConfig(grid_width=8, grid_height=8, seed=seed, enable_reward_shaping=True, enable_action_masking=True))
    trainer4 = PPOTrainer(env=env4, config=PPOConfig(learning_rate=3e-4, epochs=3, batch_size=300, mini_batch_size=64, seed=seed))
    trainer4.train(total_timesteps=3000)
    evaluator4 = PolicyEvaluator(env=env4, policy=trainer4.policy)
    trajs4 = evaluator4.evaluate_policy(num_episodes=num_episodes, seed=seed)
    metrics4 = SuccessMetricsAnalyzer.compute_success_metrics(trajs4)
    env4.close()
    print(f"      Arm 4 Mean Reward: {metrics4['mean_completion_time']:.1f} steps | Collisions: {metrics4['mean_collisions']:.1f}")

    # Compute Comparative Matrix
    comp_matrix = {
        "Random Policy": metrics1,
        "Baseline PPO": metrics2,
        "PPO + PBRS": metrics3,
        "PPO + PBRS + DAM": metrics4,
    }

    # Generate Action Statistics & Plots for Arm 4
    act_stats = ActionStatisticsAnalyzer.compute_action_frequencies(trajs4)
    ActionStatisticsAnalyzer.plot_action_distribution(act_stats, os.path.join(out_dir, "action_distribution_arm4.png"))

    # Generate Trajectory Plot for Arm 4
    if trajs4:
        TrajectoryVisualizer.plot_trajectory_path(trajs4[0], 8, 8, os.path.join(out_dir, "trajectory_path_arm4.png"))

    # Save summary JSON
    results = {
        "comparative_matrix": comp_matrix,
        "arm4_action_statistics": act_stats,
        "arm4_failure_distribution": FailureClassifier.summarize_failure_distribution(trajs4),
    }

    json_path = os.path.join(out_dir, "behavior_analysis_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Report
    EvaluationReportGenerator.generate_markdown_report(results, docs_path)

    print(f"\n[+] Behavior analysis metrics saved to: {json_path}")
    print(f"[+] Policy behavior report saved to: {docs_path}\n")

    return results


if __name__ == "__main__":
    run_behavior_analysis_framework()
