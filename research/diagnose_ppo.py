"""PPO Diagnostic Script running hyperparameter sweeps, reward shaping ablations, and plot generation."""

import json
import os
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl import EnvConfig, WarehouseGymEnv
from marl.algorithms.ppo import PPOConfig, PPOTrainer


def run_ppo_diagnostic_sweep() -> Dict[str, Any]:
    """Runs systematic PPO diagnostic experiments across hyperparameter configurations."""
    print("=" * 65)
    print("      PPO RL DIAGNOSTIC & ENVIRONMENT AUDIT ENGINE")
    print("=" * 65)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)

    env_config = EnvConfig(grid_width=8, grid_height=8, max_episode_steps=60, seed=42)
    env = WarehouseGymEnv(config=env_config)

    experiments = [
        {"name": "Baseline PPO (lr=3e-4, ent=0.01)", "lr": 3e-4, "ent": 0.01, "clip": 0.2},
        {"name": "High Entropy (lr=3e-4, ent=0.05)", "lr": 3e-4, "ent": 0.05, "clip": 0.2},
        {"name": "Low LR (lr=1e-4, ent=0.01)", "lr": 1e-4, "ent": 0.01, "clip": 0.2},
        {"name": "High Clip (lr=3e-4, ent=0.01, clip=0.3)", "lr": 3e-4, "ent": 0.01, "clip": 0.3},
    ]

    results = {}
    history = {}

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n[+] Running Diagnostic Experiment: {exp_name}...")

        config = PPOConfig(
            learning_rate=exp["lr"],
            entropy_coef=exp["ent"],
            clip_eps=exp["clip"],
            epochs=3,
            batch_size=300,
            mini_batch_size=64,
            eval_interval=600,
            eval_episodes=3,
            seed=42,
        )

        trainer = PPOTrainer(env=env, config=config)
        trainer.train(total_timesteps=3000)

        eval_m = trainer.evaluate(num_episodes=5)
        results[exp_name] = eval_m
        print(f"    -> Final Eval Mean Reward: {eval_m['eval_mean_reward']:.2f}")

    env.close()

    # Generate Diagnostic Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_names = list(results.keys())
    eval_rewards = [results[k]["eval_mean_reward"] for k in exp_names]

    bars = ax.barh(exp_names, eval_rewards, color=["#3498db", "#9b59b6", "#e67e22", "#2ecc71"])
    ax.set_xlabel("Mean Evaluation Reward")
    ax.set_title("PPO Diagnostic Sweep: Hyperparameter & Policy Convergence")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.1f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plot_path = os.path.join(out_dir, "ppo_diagnostic_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    json_path = os.path.join(out_dir, "ppo_diagnostic_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Diagnostic curves plot saved to: {plot_path}")
    print(f"[+] Diagnostic sweep results saved to: {json_path}\n")

    return results


if __name__ == "__main__":
    from typing import Any
    run_ppo_diagnostic_sweep()
