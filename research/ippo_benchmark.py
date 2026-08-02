"""IPPO Benchmark Script evaluating multi-robot scalability across 1, 2, 4, 8, and 16 robot fleets."""

import json
import os
import sys
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.ippo import IPPOConfig, IPPOTrainer


def run_ippo_scalability_benchmark() -> None:
    """Executes multi-robot scalability benchmark comparing 1, 2, 4, 8, and 16 robot fleets."""
    print("=" * 75)
    print("      INDEPENDENT PPO (IPPO) MULTI-ROBOT SCALABILITY BENCHMARK")
    print("=" * 75)

    seed = 42
    total_timesteps = 3000
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)

    agent_counts = [1, 2, 4, 8]
    summary_results = {}

    for n_agents in agent_counts:
        print(f"\n[+] Running IPPO Benchmark for {n_agents}-Robot Fleet...")
        env_cfg = MultiAgentEnvConfig(
            num_robots=n_agents,
            grid_width=10,
            grid_height=10,
            seed=seed,
        )
        env = WarehouseParallelEnv(config=env_cfg)
        ippo_cfg = IPPOConfig(
            num_agents=n_agents,
            learning_rate=3e-4,
            epochs=3,
            batch_size=300,
            mini_batch_size=64,
            eval_interval=1000,
            eval_episodes=3,
            seed=seed,
        )
        trainer = IPPOTrainer(env=env, config=ippo_cfg)
        trainer.train(total_timesteps=total_timesteps)

        eval_metrics = trainer.evaluate(num_episodes=5)
        env.close()

        summary_results[f"{n_agents}_agents"] = eval_metrics
        print(f"    {n_agents}-Robot Fleet -> Mean Reward: {eval_metrics['eval_mean_reward']:.2f} | Throughput: {eval_metrics['eval_throughput']:.3f}")

    # Plot Multi-Robot Scalability Benchmark
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    counts_str = [f"{n} Robots" for n in agent_counts]
    rewards = [summary_results[f"{n}_agents"]["eval_mean_reward"] for n in agent_counts]
    throughputs = [summary_results[f"{n}_agents"]["eval_throughput"] for n in agent_counts]
    collisions = [summary_results[f"{n}_agents"]["eval_total_collisions"] for n in agent_counts]
    fairness = [summary_results[f"{n}_agents"]["eval_jains_fairness"] for n in agent_counts]

    ax1.bar(counts_str, rewards, color="#3498db")
    ax1.set_title("Mean Reward vs Fleet Size")
    ax1.set_ylabel("Evaluation Reward")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    ax2.bar(counts_str, throughputs, color="#2ecc71")
    ax2.set_title("Package Throughput vs Fleet Size")
    ax2.set_ylabel("Throughput (deliveries/step)")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    ax3.bar(counts_str, collisions, color="#e74c3c")
    ax3.set_title("Total Collisions vs Fleet Size")
    ax3.set_ylabel("Collision Count")
    ax3.grid(axis="y", linestyle="--", alpha=0.7)

    ax4.bar(counts_str, fairness, color="#9b59b6")
    ax4.set_title("Jain's Fairness Index vs Fleet Size")
    ax4.set_ylabel("Fairness Index [0, 1]")
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis="y", linestyle="--", alpha=0.7)

    plot_path = os.path.join(out_dir, "ippo_scalability_benchmark.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    json_path = os.path.join(out_dir, "ippo_benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n[+] IPPO Scalability Benchmark plot saved to: {plot_path}")
    print(f"[+] IPPO Scalability Benchmark summary saved to: {json_path}\n")


if __name__ == "__main__":
    run_ippo_scalability_benchmark()
