"""PPO Training & Benchmark Runner comparing PPO vs. Random Policy baseline on WarehouseGymEnv."""

import json
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl import EnvConfig, WarehouseGymEnv
from marl.algorithms.ppo import PPOConfig, PPOTrainer


def run_random_policy_baseline(env: WarehouseGymEnv, num_episodes: int = 10, seed: int = 42) -> Dict[str, float]:
    """Evaluates a Random Policy baseline on WarehouseGymEnv."""
    ep_rewards = []
    ep_lengths = []
    ep_successes = []

    for ep_idx in range(num_episodes):
        obs, info = env.reset(seed=seed + ep_idx)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        ep_successes.append(1.0 if info.get("is_success", False) else 0.0)

    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "mean_length": float(np.mean(ep_lengths)),
        "success_rate": float(np.mean(ep_successes)),
    }


def run_ppo_benchmark() -> None:
    """Executes PPO training run and compares performance against Random Policy baseline."""
    print("=" * 65)
    print("        PPO vs. RANDOM POLICY BENCHMARK ENGINE")
    print("=" * 65)

    seed = 42
    env_config = EnvConfig(grid_width=10, grid_height=10, max_episode_steps=100, seed=seed)
    env = WarehouseGymEnv(config=env_config)

    # 1. Run Random Policy Baseline
    print("\n[1/3] Running Random Policy Baseline...")
    random_metrics = run_random_policy_baseline(env, num_episodes=10, seed=seed)
    print(f"      Random Policy Mean Reward : {random_metrics['mean_reward']:.2f}")
    print(f"      Random Policy Success Rate: {random_metrics['success_rate'] * 100:.1f}%")

    # 2. Train PPO Agent
    print("\n[2/3] Training PPO Agent on WarehouseGymEnv...")
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        clip_eps=0.2,
        epochs=4,
        batch_size=500,
        mini_batch_size=64,
        eval_interval=1000,
        eval_episodes=5,
        seed=seed,
    )

    trainer = PPOTrainer(env=env, config=ppo_config)

    # Train PPO for 6,000 timesteps
    trainer.train(total_timesteps=6000)

    # 3. Final Deterministic Evaluation of PPO Policy
    print("\n[3/3] Running Final Deterministic PPO Evaluation...")
    ppo_eval_metrics = trainer.evaluate(num_episodes=10)
    print(f"      PPO Policy Mean Reward : {ppo_eval_metrics['eval_mean_reward']:.2f}")
    print(f"      PPO Policy Success Rate: {ppo_eval_metrics['eval_success_rate'] * 100:.1f}%")

    # Summary table
    print("\n" + "=" * 65)
    print(f"| {'Policy Agent':<25} | {'Mean Reward':<15} | {'Success Rate':<15} |")
    print("| " + "-" * 25 + " | " + "-" * 15 + " | " + "-" * 15 + " |")
    print(f"| {'Random Baseline':<25} | {random_metrics['mean_reward']:<15.2f} | {random_metrics['success_rate']*100:<14.1f}% |")
    print(f"| {'PPO Trained Agent':<25} | {ppo_eval_metrics['eval_mean_reward']:<15.2f} | {ppo_eval_metrics['eval_success_rate']*100:<14.1f}% |")
    print("=" * 65)

    # Generate Matplotlib Learning Curve Plot
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Random Baseline", "PPO Agent"]
    rewards = [random_metrics["mean_reward"], ppo_eval_metrics["eval_mean_reward"]]
    colors = ["#e74c3c", "#2ecc71"]

    bars = ax.bar(categories, rewards, color=colors, width=0.4)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Warehouse Digital Twin: Random Baseline vs. PPO Agent")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plot_path = os.path.join(out_dir, "ppo_vs_random_benchmark.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save benchmark JSON summary
    summary_data = {
        "random_baseline": random_metrics,
        "ppo_eval": ppo_eval_metrics,
    }
    json_path = os.path.join(out_dir, "ppo_benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[+] Benchmark plot saved to: {plot_path}")
    print(f"[+] Benchmark summary saved to: {json_path}\n")

    env.close()


if __name__ == "__main__":
    from typing import Dict
    run_ppo_benchmark()
