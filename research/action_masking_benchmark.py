"""Action Masking Benchmark Script running controlled 3-Arm Ablation Study:
Arm 1: Baseline PPO
Arm 2: PPO + PBRS
Arm 3: PPO + PBRS + Dynamic Action Masking (DAM)
"""

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


def run_action_masking_benchmark() -> None:
    """Executes controlled 3-arm ablation experiment comparing PPO variants."""
    print("=" * 70)
    print("   DYNAMIC ACTION MASKING (DAM) CONTROLLED 3-ARM ABLATION BENCHMARK")
    print("=" * 70)

    seed = 42
    total_timesteps = 8000
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)

    # Arm 1: Baseline PPO
    print("\n[1/3] Training Arm 1: Baseline PPO...")
    env_arm1 = WarehouseGymEnv(config=EnvConfig(
        grid_width=8, grid_height=8, max_episode_steps=80, seed=seed, enable_reward_shaping=False, enable_action_masking=False
    ))
    ppo_cfg_arm1 = PPOConfig(
        learning_rate=3e-4, epochs=4, batch_size=400, mini_batch_size=64, eval_interval=1000, eval_episodes=5, seed=seed
    )
    trainer_arm1 = PPOTrainer(env=env_arm1, config=ppo_cfg_arm1)
    trainer_arm1.train(total_timesteps=total_timesteps)
    eval_arm1 = trainer_arm1.evaluate(num_episodes=10)
    env_arm1.close()
    print(f"      Arm 1 (Baseline PPO) Mean Reward: {eval_arm1['eval_mean_reward']:.2f}")

    # Arm 2: PPO + PBRS
    print("\n[2/3] Training Arm 2: PPO + PBRS...")
    env_arm2 = WarehouseGymEnv(config=EnvConfig(
        grid_width=8, grid_height=8, max_episode_steps=80, seed=seed, enable_reward_shaping=True, enable_action_masking=False
    ))
    ppo_cfg_arm2 = PPOConfig(
        learning_rate=3e-4, epochs=4, batch_size=400, mini_batch_size=64, eval_interval=1000, eval_episodes=5, seed=seed
    )
    trainer_arm2 = PPOTrainer(env=env_arm2, config=ppo_cfg_arm2)
    trainer_arm2.train(total_timesteps=total_timesteps)
    eval_arm2 = trainer_arm2.evaluate(num_episodes=10)
    env_arm2.close()
    print(f"      Arm 2 (PPO + PBRS) Mean Reward   : {eval_arm2['eval_mean_reward']:.2f}")

    # Arm 3: PPO + PBRS + Dynamic Action Masking (DAM)
    print("\n[3/3] Training Arm 3: PPO + PBRS + Dynamic Action Masking...")
    env_arm3 = WarehouseGymEnv(config=EnvConfig(
        grid_width=8, grid_height=8, max_episode_steps=80, seed=seed, enable_reward_shaping=True, enable_action_masking=True
    ))
    ppo_cfg_arm3 = PPOConfig(
        learning_rate=3e-4, epochs=4, batch_size=400, mini_batch_size=64, eval_interval=1000, eval_episodes=5, seed=seed
    )
    trainer_arm3 = PPOTrainer(env=env_arm3, config=ppo_cfg_arm3)
    trainer_arm3.train(total_timesteps=total_timesteps)
    eval_arm3 = trainer_arm3.evaluate(num_episodes=10)
    env_arm3.close()
    print(f"      Arm 3 (PPO + PBRS + DAM) Reward : {eval_arm3['eval_mean_reward']:.2f}")

    # 3-Arm Comparative Summary Table
    print("\n" + "=" * 70)
    print(f"| {'Experimental Arm':<30} | {'Mean Reward':<15} | {'Success Rate':<15} |")
    print("| " + "-" * 30 + " | " + "-" * 15 + " | " + "-" * 15 + " |")
    print(f"| {'Arm 1: Baseline PPO':<30} | {eval_arm1['eval_mean_reward']:<15.2f} | {eval_arm1['eval_success_rate']*100:<14.1f}% |")
    print(f"| {'Arm 2: PPO + PBRS':<30} | {eval_arm2['eval_mean_reward']:<15.2f} | {eval_arm2['eval_success_rate']*100:<14.1f}% |")
    print(f"| {'Arm 3: PPO + PBRS + DAM':<30} | {eval_arm3['eval_mean_reward']:<15.2f} | {eval_arm3['eval_success_rate']*100:<14.1f}% |")
    print("=" * 70)

    # Plot 3-Arm Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    categories = ["Arm 1:\nBaseline PPO", "Arm 2:\nPPO + PBRS", "Arm 3:\nPPO + PBRS + DAM"]
    rewards = [eval_arm1["eval_mean_reward"], eval_arm2["eval_mean_reward"], eval_arm3["eval_mean_reward"]]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    bars = ax.bar(categories, rewards, color=colors, width=0.45)
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title("3-Arm Controlled Ablation: Action Masking & Reward Shaping Impact")
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

    plot_path = os.path.join(out_dir, "dam_vs_pbrs_baseline_benchmark.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    summary_data = {
        "arm1_baseline_ppo": eval_arm1,
        "arm2_ppo_plus_pbrs": eval_arm2,
        "arm3_ppo_pbrs_dam": eval_arm3,
    }
    json_path = os.path.join(out_dir, "action_masking_benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[+] Benchmark plot saved to: {plot_path}")
    print(f"[+] Benchmark summary saved to: {json_path}\n")


if __name__ == "__main__":
    run_action_masking_benchmark()
