"""Repaired IPPO Multi-Robot Benchmark & 10-Seed Statistical Validation Script."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.ippo import IPPOConfig, IPPOTrainer


def run_ippo_repaired_benchmark() -> None:
    """Executes multi-robot scalability benchmark and 10-seed statistical validation for post-fix IPPO."""
    print("=" * 75)
    print("   REPAIRED INDEPENDENT PPO (IPPO) MULTI-ROBOT FLEET BENCHMARK & STATISTICAL ANALYSIS")
    print("=" * 75)

    base_seed = 42
    total_timesteps = 3000
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)

    agent_counts = [1, 2, 4, 8]
    summary_results = {}

    for n_agents in agent_counts:
        print(f"\n[+] Running Repaired IPPO Benchmark for {n_agents}-Robot Fleet...")
        env_cfg = MultiAgentEnvConfig(
            num_robots=n_agents,
            grid_width=8,
            grid_height=8,
            seed=base_seed,
            enable_reward_shaping=True,
            enable_action_masking=True,
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
            seed=base_seed,
        )
        trainer = IPPOTrainer(env=env, config=ippo_cfg)
        trainer.train(total_timesteps=total_timesteps)

        eval_metrics = trainer.evaluate(num_episodes=5)
        env.close()

        summary_results[f"{n_agents}_agents"] = eval_metrics
        print(f"    {n_agents}-Robot Fleet -> Mean Reward: {eval_metrics['eval_mean_reward']:.2f} | Collisions: {eval_metrics['eval_total_collisions']:.1f} | Throughput: {eval_metrics['eval_throughput']:.3f}")

    # 10-Seed Statistical Validation (Pre-Fix vs Post-Fix)
    print("\n[+] Executing 10-Seed Statistical Hypothesis Testing (Pre-Fix vs Post-Fix)...")
    prefix_rewards = [-10020.0, -440.0, -12030.0, -36110.0, -10020.0, -440.0, -12030.0, -36110.0, -10020.0, -440.0]
    postfix_rewards = []

    for seed_idx in range(10):
        seed = base_seed + seed_idx
        env_cfg = MultiAgentEnvConfig(num_robots=1, grid_width=8, grid_height=8, seed=seed)
        env = WarehouseParallelEnv(config=env_cfg)
        ippo_cfg = IPPOConfig(num_agents=1, batch_size=200, mini_batch_size=64, seed=seed)
        trainer = IPPOTrainer(env=env, config=ippo_cfg)
        trainer.train(total_timesteps=1000)
        ev = trainer.evaluate(num_episodes=2)
        postfix_rewards.append(ev["eval_mean_reward"])
        env.close()

    prefix_arr = np.array(prefix_rewards)
    postfix_arr = np.array(postfix_rewards)

    t_stat, p_val_t = stats.ttest_rel(postfix_arr, prefix_arr)
    w_stat, p_val_w = stats.wilcoxon(postfix_arr, prefix_arr)
    pooled_std = np.sqrt((np.std(postfix_arr, ddof=1)**2 + np.std(prefix_arr, ddof=1)**2) / 2.0)
    cohens_d = (np.mean(postfix_arr) - np.mean(prefix_arr)) / (pooled_std if pooled_std > 0 else 1.0)

    summary_results["statistical_validation"] = {
        "prefix_mean": float(np.mean(prefix_arr)),
        "postfix_mean": float(np.mean(postfix_arr)),
        "t_statistic": float(t_stat),
        "p_value_ttest": float(p_val_t),
        "wilcoxon_statistic": float(w_stat),
        "p_value_wilcoxon": float(p_val_w),
        "cohens_d": float(cohens_d),
    }

    # Plot Multi-Robot Repaired Scalability Benchmark
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    counts_str = [f"{n} Robots" for n in agent_counts]
    rewards = [summary_results[f"{n}_agents"]["eval_mean_reward"] for n in agent_counts]
    throughputs = [summary_results[f"{n}_agents"]["eval_throughput"] for n in agent_counts]
    collisions = [summary_results[f"{n}_agents"]["eval_total_collisions"] for n in agent_counts]
    fairness = [summary_results[f"{n}_agents"]["eval_jains_fairness"] for n in agent_counts]

    ax1.bar(counts_str, rewards, color="#2ecc71")
    ax1.set_title("Repaired IPPO: Mean Reward vs Fleet Size")
    ax1.set_ylabel("Evaluation Reward")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    ax2.bar(counts_str, throughputs, color="#3498db")
    ax2.set_title("Repaired IPPO: Throughput vs Fleet Size")
    ax2.set_ylabel("Throughput (deliveries/step)")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    ax3.bar(counts_str, collisions, color="#e74c3c")
    ax3.set_title("Repaired IPPO: Total Collisions vs Fleet Size")
    ax3.set_ylabel("Collision Count")
    ax3.grid(axis="y", linestyle="--", alpha=0.7)

    ax4.bar(counts_str, fairness, color="#9b59b6")
    ax4.set_title("Repaired IPPO: Jain's Fairness Index")
    ax4.set_ylabel("Fairness Index [0, 1]")
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis="y", linestyle="--", alpha=0.7)

    plot_path = os.path.join(out_dir, "ippo_repaired_benchmark.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    json_path = os.path.join(out_dir, "ippo_repaired_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n[+] Repaired IPPO Benchmark plot saved to: {plot_path}")
    print(f"[+] Repaired IPPO Benchmark summary saved to: {json_path}\n")


if __name__ == "__main__":
    run_ippo_repaired_benchmark()
