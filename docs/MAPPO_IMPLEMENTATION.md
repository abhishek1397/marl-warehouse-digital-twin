# Multi-Agent Proximal Policy Optimization (MAPPO) (`marl/algorithms/mappo/`)

This document details the architectural design, Centralized Training with Decentralized Execution (CTDE) formulation, package structure, and benchmark comparison against IPPO for Multi-Agent Proximal Policy Optimization (MAPPO).

---

## 1. CTDE Paradigm & Mathematical Formulation

MAPPO optimizes multi-robot warehouse fleet logistics using the **Centralized Training Decentralized Execution (CTDE)** paradigm.

### Centralized Critic $V_{\phi}(S)$ (Training Time Only)
During training, the centralized critic $V_{\phi}(S)$ processes privileged global warehouse state $S \in \mathbb{R}^{H \times W}$ (extracted via `env.state()`) or joint agent state vectors:

$$\mathcal{L}^{\text{Critic}}(\phi) = \frac{1}{B} \sum_{i=1}^B \left( V_{\phi}(S_i) - \hat{R}_i \right)^2$$

### Decentralized Actor $\pi_{\theta}(a_i | o_i)$ (Training & Execution Time)
During both rollout collection and execution inference, each robot actor receives **only** its local observation $o_i$ without access to global state $S$:

$$\mathcal{L}^{\text{Actor}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t^{\text{CTDE}}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t^{\text{CTDE}} \right) \right] + c_2 S[\pi_{\theta}](o_t)$$

---

## 2. PPO vs. IPPO vs. MAPPO Comparison Matrix

| Architectural Feature | Single-Agent PPO | Independent PPO (IPPO) | Multi-Agent PPO (MAPPO) |
| :--- | :--- | :--- | :--- |
| **Critic Input** | Local Obs $o$ | Local Obs $o_i$ per agent | **Global State $S$ (Centralized)** |
| **Actor Policy** | Single Network $\pi_{\theta}$ | Independent $\pi_{\theta_i}$ or Shared $\pi_{\theta}$ | **Shared Decentralized Actor $\pi_{\theta}$** |
| **Advantage Source** | Local $V(o)$ | Local $V_i(o_i)$ | **Centralized $V(S)$** |
| **Training Paradigm** | Single-Agent | DLDE | **CTDE (Centralized Training, Decentralized Execution)** |

---

## 3. Package Architecture (`marl/algorithms/mappo/`)

- [config.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/config.py): `MAPPOConfig` dataclass (`num_agents`, `shared_policy`, `centralized_critic`, hyperparameter settings).
- [centralized_critic.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/centralized_critic.py): `CentralizedValueNetwork` mapping global state $S \in \mathbb{R}^{H \times W}$ to scalar value estimates $V(S)$.
- [shared_policy.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/shared_policy.py): `SharedPolicyManager` managing shared actor network weights across all active robots during decentralized execution.
- [agent.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/agent.py): `MAPPOAgent` encapsulating agent policy, optimizer, scheduler, and rollout buffer.
- [rollout_manager.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/rollout_manager.py): `MAPPORolloutManager` collecting local observations for actors AND global state for centralized critic on PettingZoo `WarehouseParallelEnv`.
- [batch_builder.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/batch_builder.py): `MAPPOBatchBuilder` constructing mini-batches with local actor obs and global state tensors.
- [metrics.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/metrics.py): `MAPPOMetricsTracker` computing actor losses, critic loss, entropy, throughput, and Jain's Fairness Index.
- [evaluator.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/evaluator.py): `MAPPOEvaluator` evaluating multi-robot fleets (1, 2, 4, 8, 16, 32 agents).
- [checkpoint.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/checkpoint.py): `MAPPOCheckpointHandler` serializing shared actor policy weights, centralized critic weights, optimizers, and metadata.
- [trainer.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/mappo/trainer.py): `MAPPOTrainer` orchestrating MAPPO CTDE training loops.

---

## 4. Usage Code Example

```python
from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.mappo import MAPPOConfig, MAPPOTrainer

# 1. Initialize PettingZoo Multi-Robot Parallel Environment
env_cfg = MultiAgentEnvConfig(
    num_robots=4,
    grid_width=12,
    grid_height=12,
    enable_reward_shaping=True,
    enable_action_masking=True,
)
env = WarehouseParallelEnv(config=env_cfg)

# 2. Instantiate MAPPO Trainer (CTDE Paradigm with Centralized Critic V(S))
mappo_config = MAPPOConfig(
    num_agents=4,
    shared_policy=True,
    centralized_critic=True,
    actor_lr=3e-4,
    critic_lr=5e-4,
    batch_size=400,
)
trainer = MAPPOTrainer(env=env, config=mappo_config)

# 3. Train MAPPO Fleet
trainer.train(total_timesteps=20000)

# 4. Evaluate Decentralized Actors
eval_metrics = trainer.evaluate(num_episodes=10)
print(f"MAPPO 4-Robot Fleet Throughput: {eval_metrics['eval_throughput']:.3f} | Fairness: {eval_metrics['eval_jains_fairness']:.2f}")

env.close()
```

---

## 5. Benchmark Comparison (MAPPO vs. IPPO)

- **Benchmark Plot**: Saved to `runs/benchmarks/mappo_vs_ippo_benchmark.png`
- **Summary Metrics JSON**: Saved to `runs/benchmarks/mappo_benchmark_summary.json`
