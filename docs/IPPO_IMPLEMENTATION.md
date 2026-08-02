# Independent Proximal Policy Optimization (IPPO) (`marl/algorithms/ippo/`)

This document details the architectural design, implementation, parameter sharing modes, and benchmark results for Independent Proximal Policy Optimization (IPPO).

---

## 1. Algorithm Overview & Mathematical Formulation

Independent PPO treats each robot $i \in \{0, \dots, N-1\}$ in a multi-agent environment as an independent single-agent PPO optimizer within a shared environment.

$$\mathcal{L}^{\text{IPPO}}_i(\theta_i) = \hat{\mathbb{E}}_t \left[ \min \left( r_{t, i}(\theta_i) \hat{A}_{t, i}, \text{clip}(r_{t, i}(\theta_i), 1-\epsilon, 1+\epsilon) \hat{A}_{t, i} \right) \right] - c_1 \mathcal{L}^{\text{VF}}_i(\theta_i) + c_2 S[\pi_{\theta_i}](s_{t, i})$$

### Core Decentralized Properties
- **Decentralized Learning & Execution (DLDE)**: Each agent independently observes $o_i$, samples action $a_i$, receives individual reward $r_i$, computes GAE advantage $\hat{A}_i$, and optimizes policy weights $\theta_i$.
- **No Centralized Information**: No joint value functions $V(s_1, \dots, s_N)$, communication channels, or attention mechanisms are used.

---

## 2. Parameter Sharing Modes

| Parameter Mode | Configuration Flag | Description |
| :--- | :--- | :--- |
| **Mode 1: Independent Parameters** | `shared_policy = False` | Each robot owns a distinct `PolicyNetwork` and `PPOOptimizer`. Recommended for heterogeneous robot fleets. |
| **Mode 2: Shared Parameters** | `shared_policy = True` | All robots share a single `PolicyNetwork` weight matrix, but collect independent trajectories and accumulate gradients. Recommended for homogeneous robot fleets. |

---

## 3. Package Architecture (`marl/algorithms/ippo/`)

- [config.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/ippo/config.py): `IPPOConfig` dataclass (`num_agents`, `shared_policy`, hyperparameter controls).
- [agent.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/ippo/agent.py): `IPPOAgent` encapsulating agent policy, optimizer, scheduler, rollout buffer, and loss engine.
- [policy_manager.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/ippo/policy_manager.py): `PolicyManager` instantiating and looking up agent policies under Independent or Shared modes.
- [rollout_manager.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/ippo/rollout_manager.py): `IPPORolloutManager` managing multi-agent trajectory collection on PettingZoo `WarehouseParallelEnv`.
- [metrics.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/ippo/metrics.py): `IPPOMetricsTracker` computing per-agent rewards, losses, throughput, and Jain's Fairness Index.
- [evaluator.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/ippo/evaluator.py): `IPPOEvaluator` evaluating multi-robot fleets (2, 4, 8, 16 agents).
- [checkpoint.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/ippo/checkpoint.py): `IPPOCheckpointHandler` serializing multi-agent policy states.
- [trainer.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/ippo/trainer.py): `IPPOTrainer` orchestrating IPPO training loops.

---

## 4. Usage Code Example

```python
from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.ippo import IPPOConfig, IPPOTrainer

# 1. Initialize PettingZoo Multi-Robot Parallel Environment
env_cfg = MultiAgentEnvConfig(
    num_robots=4,
    grid_width=12,
    grid_height=12,
    enable_reward_shaping=True,
    enable_action_masking=True,
)
env = WarehouseParallelEnv(config=env_cfg)

# 2. Instantiate IPPO Trainer (Shared or Independent parameter mode)
ippo_config = IPPOConfig(
    num_agents=4,
    shared_policy=False,  # Independent parameter mode
    learning_rate=3e-4,
    epochs=4,
    batch_size=400,
)
trainer = IPPOTrainer(env=env, config=ippo_config)

# 3. Train IPPO Fleet
trainer.train(total_timesteps=20000)

# 4. Evaluate Multi-Robot Fleet
eval_metrics = trainer.evaluate(num_episodes=10)
print(f"IPPO 4-Robot Fleet Throughput: {eval_metrics['eval_throughput']:.3f} | Fairness: {eval_metrics['eval_jains_fairness']:.2f}")

env.close()
```

---

## 5. Multi-Robot Scalability Benchmark Results

- **Scalability Plot**: Saved to `runs/benchmarks/ippo_scalability_benchmark.png`
- **Scalability Metrics JSON**: Saved to `runs/benchmarks/ippo_benchmark_summary.json`
