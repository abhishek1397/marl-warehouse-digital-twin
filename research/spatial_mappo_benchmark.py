"""Spatial MAPPO (S-MAPPO) 3-Way Benchmark & 10-Seed Statistical Validation Script."""

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
from marl.algorithms.spatial_mappo import SpatialMAPPOConfig, SpatialMAPPOTrainer


def run_spatial_mappo_benchmark() -> None:
    """Executes 3-way multi-robot fleet benchmark comparing IPPO vs MAPPO (MLP Critic) vs Spatial MAPPO (CNN Critic)."""
    print("=" * 75)
    print("   3-WAY MULTI-ROBOT BENCHMARK: IPPO vs MAPPO (MLP) vs SPATIAL MAPPO (CNN)")
    print("=" * 75)

    base_seed = 42
    total_timesteps = 3000
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/benchmarks"))
    os.makedirs(out_dir, exist_ok=True)

    agent_counts = [1, 2, 4, 8]
    summary_results = {"ippo": {}, "mappo_mlp": {}, "spatial_mappo_cnn": {}}

    for n_agents in agent_counts:
        print(f"\n[+] Running 3-Way Benchmark for {n_agents}-Robot Fleet...")
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

        # 2. MAPPO MLP Critic Benchmark
        env_mappo = WarehouseParallelEnv(config=env_cfg)
        mappo_cfg = MAPPOConfig(num_agents=n_agents, batch_size=300, mini_batch_size=64, seed=base_seed)
        mappo_trainer = MAPPOTrainer(env=env_mappo, config=mappo_cfg)
        mappo_trainer.train(total_timesteps=total_timesteps)
        mappo_eval = mappo_trainer.evaluate(num_episodes=3)
        env_mappo.close()
        summary_results["mappo_mlp"][f"{n_agents}_agents"] = mappo_eval

        # 3. Spatial MAPPO CNN Critic Benchmark
        env_smappo = WarehouseParallelEnv(config=env_cfg)
        smappo_cfg = SpatialMAPPOConfig(num_agents=n_agents, batch_size=300, mini_batch_size=64, seed=base_seed)
        smappo_trainer = SpatialMAPPOTrainer(env=env_smappo, config=smappo_cfg)
        smappo_trainer.train(total_timesteps=total_timesteps)
        smappo_eval = smappo_trainer.evaluate(num_episodes=3)
        env_smappo.close()
        summary_results["spatial_mappo_cnn"][f"{n_agents}_agents"] = smappo_eval

        print(f"    {n_agents}-Robot Fleet -> IPPO: {ippo_eval['eval_mean_reward']:.2f} | MAPPO (MLP): {mappo_eval['eval_mean_reward']:.2f} | Spatial MAPPO (CNN): {smappo_eval['eval_mean_reward']:.2f}")

    # 10-Seed Statistical Validation (MAPPO MLP vs Spatial MAPPO CNN on 4-Robot Fleet)
    print("\n[+] Executing 10-Seed Statistical Hypothesis Testing (MAPPO MLP vs Spatial MAPPO CNN)...")
    mlp_seeds = []
    cnn_seeds = []

    for seed_idx in range(10):
        seed = base_seed + seed_idx
        env_cfg = MultiAgentEnvConfig(num_robots=4, grid_width=8, grid_height=8, seed=seed)

        env1 = WarehouseParallelEnv(config=env_cfg)
        t1 = MAPPOTrainer(env=env1, config=MAPPOConfig(num_agents=4, batch_size=200, seed=seed))
        t1.train(total_timesteps=1000)
        mlp_seeds.append(t1.evaluate(num_episodes=2)["eval_mean_reward"])
        env1.close()

        env2 = WarehouseParallelEnv(config=env_cfg)
        t2 = SpatialMAPPOTrainer(env=env2, config=SpatialMAPPOConfig(num_agents=4, batch_size=200, seed=seed))
        t2.train(total_timesteps=1000)
        cnn_seeds.append(t2.evaluate(num_episodes=2)["eval_mean_reward"])
        env2.close()

    mlp_arr = np.array(mlp_seeds)
    cnn_arr = np.array(cnn_seeds)

    t_stat, p_val_t = stats.ttest_rel(cnn_arr, mlp_arr)
    w_stat, p_val_w = stats.wilcoxon(cnn_arr, mlp_arr)
    pooled_std = np.sqrt((np.std(cnn_arr, ddof=1)**2 + np.std(mlp_arr, ddof=1)**2) / 2.0)
    cohens_d = (np.mean(cnn_arr) - np.mean(mlp_arr)) / (pooled_std if pooled_std > 0 else 1.0)

    summary_results["statistical_validation"] = {
        "mappo_mlp_mean": float(np.mean(mlp_arr)),
        "spatial_mappo_cnn_mean": float(np.mean(cnn_arr)),
        "t_statistic": float(t_stat),
        "p_value_ttest": float(p_val_t),
        "wilcoxon_statistic": float(w_stat),
        "p_value_wilcoxon": float(p_val_w),
        "cohens_d": float(cohens_d),
    }

    # Plot 3-Way Multi-Robot Fleet Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    counts_str = [f"{n} Robots" for n in agent_counts]
    x = np.arange(len(counts_str))
    width = 0.25

    ippo_rew = [summary_results["ippo"][f"{n}_agents"]["eval_mean_reward"] for n in agent_counts]
    mlp_rew = [summary_results["mappo_mlp"][f"{n}_agents"]["eval_mean_reward"] for n in agent_counts]
    cnn_rew = [summary_results["spatial_mappo_cnn"][f"{n}_agents"]["eval_mean_reward"] for n in agent_counts]

    ax1.bar(x - width, ippo_rew, width, label="IPPO", color="#3498db")
    ax1.bar(x, mlp_rew, width, label="MAPPO (MLP Critic)", color="#e74c3c")
    ax1.bar(x + width, cnn_rew, width, label="Spatial MAPPO (CNN Critic)", color="#2ecc71")
    ax1.set_title("Mean Evaluation Reward (3-Way Comparison)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(counts_str)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    plot_path = os.path.join(out_dir, "spatial_mappo_benchmark.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    json_path = os.path.join(out_dir, "spatial_mappo_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n[+] 3-Way Benchmark plot saved to: {plot_path}")
    print(f"[+] 3-Way Benchmark summary saved to: {json_path}\n")


if __name__ == "__main__":
    run_spatial_mappo_benchmark()
