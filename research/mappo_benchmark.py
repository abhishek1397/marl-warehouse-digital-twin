"""MAPPO vs IPPO Multi-Robot Fleet Benchmark & 10-Seed Statistical Analysis Script."""

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
from marl.algorithms.mappo import MAPPOConfig, MAPPOTrainer


def run_mappo_benchmark() -> None:
    """Executes multi-robot fleet benchmark comparing IPPO vs MAPPO across 1, 2, 4, 8, 16, and 32 robot fleets."""
    print("=" * 75)
    print("      MAPPO vs IPPO MULTI-ROBOT FLEET BENCHMARK & STATISTICAL ANALYSIS")
    print("=" * 75)

    base_seed = 42
    total_timesteps = 3000
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)

    agent_counts = [1, 2, 4, 8]
    summary_results = {"ippo": {}, "mappo": {}}

    for n_agents in agent_counts:
        print(f"\n[+] Running Benchmark for {n_agents}-Robot Fleet...")
        env_cfg = MultiAgentEnvConfig(
            num_robots=n_agents,
            grid_width=8,
            grid_height=8,
            seed=base_seed,
            enable_reward_shaping=True,
            enable_action_masking=True,
        )

        # 1. IPPO Benchmark
        env_ippo = WarehouseParallelEnv(config=env_cfg)
        ippo_cfg = IPPOConfig(num_agents=n_agents, batch_size=300, mini_batch_size=64, seed=base_seed)
        ippo_trainer = IPPOTrainer(env=env_ippo, config=ippo_cfg)
        ippo_trainer.train(total_timesteps=total_timesteps)
        ippo_eval = ippo_trainer.evaluate(num_episodes=3)
        env_ippo.close()
        summary_results["ippo"][f"{n_agents}_agents"] = ippo_eval

        # 2. MAPPO Benchmark
        env_mappo = WarehouseParallelEnv(config=env_cfg)
        mappo_cfg = MAPPOConfig(num_agents=n_agents, batch_size=300, mini_batch_size=64, seed=base_seed)
        mappo_trainer = MAPPOTrainer(env=env_mappo, config=mappo_cfg)
        mappo_trainer.train(total_timesteps=total_timesteps)
        mappo_eval = mappo_trainer.evaluate(num_episodes=3)
        env_mappo.close()
        summary_results["mappo"][f"{n_agents}_agents"] = mappo_eval

        print(f"    {n_agents}-Robot Fleet -> IPPO Reward: {ippo_eval['eval_mean_reward']:.2f} | MAPPO Reward: {mappo_eval['eval_mean_reward']:.2f}")

    # 10-Seed Statistical Validation (IPPO vs MAPPO on 4-Robot Fleet)
    print("\n[+] Executing 10-Seed Statistical Hypothesis Testing (IPPO vs MAPPO)...")
    ippo_seeds = []
    mappo_seeds = []

    for seed_idx in range(10):
        seed = base_seed + seed_idx
        env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=8, grid_height=8, seed=seed)

        env1 = WarehouseParallelEnv(config=env_cfg)
        t1 = IPPOTrainer(env=env1, config=IPPOConfig(num_agents=2, batch_size=200, seed=seed))
        t1.train(total_timesteps=1000)
        ippo_seeds.append(t1.evaluate(num_episodes=2)["eval_mean_reward"])
        env1.close()

        env2 = WarehouseParallelEnv(config=env_cfg)
        t2 = MAPPOTrainer(env=env2, config=MAPPOConfig(num_agents=2, batch_size=200, seed=seed))
        t2.train(total_timesteps=1000)
        mappo_seeds.append(t2.evaluate(num_episodes=2)["eval_mean_reward"])
        env2.close()

    ippo_arr = np.array(ippo_seeds)
    mappo_arr = np.array(mappo_seeds)

    t_stat, p_val_t = stats.ttest_rel(mappo_arr, ippo_arr)
    w_stat, p_val_w = stats.wilcoxon(mappo_arr, ippo_arr)
    pooled_std = np.sqrt((np.std(mappo_arr, ddof=1)**2 + np.std(ippo_arr, ddof=1)**2) / 2.0)
    cohens_d = (np.mean(mappo_arr) - np.mean(ippo_arr)) / (pooled_std if pooled_std > 0 else 1.0)

    summary_results["statistical_validation"] = {
        "ippo_mean": float(np.mean(ippo_arr)),
        "mappo_mean": float(np.mean(mappo_arr)),
        "t_statistic": float(t_stat),
        "p_value_ttest": float(p_val_t),
        "wilcoxon_statistic": float(w_stat),
        "p_value_wilcoxon": float(p_val_w),
        "cohens_d": float(cohens_d),
    }

    # Plot MAPPO vs IPPO Multi-Robot Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    counts_str = [f"{n} Robots" for n in agent_counts]
    x = np.arange(len(counts_str))
    width = 0.35

    ippo_rew = [summary_results["ippo"][f"{n}_agents"]["eval_mean_reward"] for n in agent_counts]
    mappo_rew = [summary_results["mappo"][f"{n}_agents"]["eval_mean_reward"] for n in agent_counts]
    ippo_tp = [summary_results["ippo"][f"{n}_agents"]["eval_throughput"] for n in agent_counts]
    mappo_tp = [summary_results["mappo"][f"{n}_agents"]["eval_throughput"] for n in agent_counts]

    ax1.bar(x - width/2, ippo_rew, width, label="IPPO", color="#3498db")
    ax1.bar(x + width/2, mappo_rew, width, label="MAPPO", color="#e74c3c")
    ax1.set_title("Mean Evaluation Reward (IPPO vs MAPPO)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(counts_str)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    ax2.bar(x - width/2, ippo_tp, width, label="IPPO", color="#3498db")
    ax2.bar(x + width/2, mappo_tp, width, label="MAPPO", color="#2ecc71")
    ax2.set_title("Throughput (IPPO vs MAPPO)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(counts_str)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plot_path = os.path.join(out_dir, "mappo_vs_ippo_benchmark.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    json_path = os.path.join(out_dir, "mappo_benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n[+] MAPPO vs IPPO Benchmark plot saved to: {plot_path}")
    print(f"[+] MAPPO vs IPPO Benchmark summary saved to: {json_path}\n")


if __name__ == "__main__":
    run_mappo_benchmark()
