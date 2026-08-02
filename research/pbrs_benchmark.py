"""PBRS Benchmark Script comparing Baseline PPO vs. PPO + Potential-Based Reward Shaping under identical seeds."""

import json
import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl import EnvConfig, WarehouseGymEnv
from marl.algorithms.ppo import PPOConfig, PPOTrainer


def run_pbrs_benchmark() -> None:
    """Executes controlled ablation experiment comparing Baseline PPO vs. PPO + PBRS."""
    print("=" * 65)
    print("  POTENTIAL-BASED REWARD SHAPING (PBRS) ABLATION BENCHMARK")
    print("=" * 65)

    seed = 42
    total_timesteps = 8000
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)

    # 1. Baseline PPO (enable_reward_shaping = False)
    print("\n[1/2] Training Baseline PPO (Original Reward)...")
    env_base = WarehouseGymEnv(config=EnvConfig(
        grid_width=8, grid_height=8, max_episode_steps=80, seed=seed, enable_reward_shaping=False
    ))
    ppo_cfg_base = PPOConfig(
        learning_rate=3e-4, epochs=4, batch_size=400, mini_batch_size=64, eval_interval=1000, eval_episodes=5, seed=seed
    )
    trainer_base = PPOTrainer(env=env_base, config=ppo_cfg_base)
    trainer_base.train(total_timesteps=total_timesteps)
    eval_base = trainer_base.evaluate(num_episodes=10)
    env_base.close()
    print(f"      Baseline PPO Mean Reward: {eval_base['eval_mean_reward']:.2f}")

    # 2. PPO + PBRS (enable_reward_shaping = True)
    print("\n[2/2] Training PPO + PBRS (Potential-Based Shaped Reward)...")
    env_pbrs = WarehouseGymEnv(config=EnvConfig(
        grid_width=8, grid_height=8, max_episode_steps=80, seed=seed, enable_reward_shaping=True, shaping_scale=1.0, shaping_gamma=0.99
    ))
    ppo_cfg_pbrs = PPOConfig(
        learning_rate=3e-4, epochs=4, batch_size=400, mini_batch_size=64, eval_interval=1000, eval_episodes=5, seed=seed
    )
    trainer_pbrs = PPOTrainer(env=env_pbrs, config=ppo_cfg_pbrs)
    trainer_pbrs.train(total_timesteps=total_timesteps)
    eval_pbrs = trainer_pbrs.evaluate(num_episodes=10)
    env_pbrs.close()
    print(f"      PPO + PBRS Mean Reward  : {eval_pbrs['eval_mean_reward']:.2f}")

    # Summary table
    print("\n" + "=" * 65)
    print(f"| {'Experimental Arm':<25} | {'Mean Reward':<15} | {'Success Rate':<15} |")
    print("| " + "-" * 25 + " | " + "-" * 15 + " | " + "-" * 15 + " |")
    print(f"| {'Baseline PPO':<25} | {eval_base['eval_mean_reward']:<15.2f} | {eval_base['eval_success_rate']*100:<14.1f}% |")
    print(f"| {'PPO + PBRS':<25} | {eval_pbrs['eval_mean_reward']:<15.2f} | {eval_pbrs['eval_success_rate']*100:<14.1f}% |")
    print("=" * 65)

    # Generate Comparative Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Baseline PPO", "PPO + PBRS"]
    rewards = [eval_base["eval_mean_reward"], eval_pbrs["eval_mean_reward"]]
    colors = ["#e74c3c", "#2ecc71"]

    bars = ax.bar(categories, rewards, color=colors, width=0.4)
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title("Reward Shaping Ablation: Baseline PPO vs. PPO + PBRS")
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

    plot_path = os.path.join(out_dir, "pbrs_vs_baseline_benchmark.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save summary JSON
    summary_data = {
        "baseline_ppo": eval_base,
        "ppo_plus_pbrs": eval_pbrs,
    }
    json_path = os.path.join(out_dir, "pbrs_benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[+] Benchmark plot saved to: {plot_path}")
    print(f"[+] Benchmark summary saved to: {json_path}\n")


if __name__ == "__main__":
    run_pbrs_benchmark()
