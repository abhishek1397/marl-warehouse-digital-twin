# Spatial MAPPO (S-MAPPO): CNN Centralized Critic Architecture (`docs/SPATIAL_MAPPO.md`)

This technical research report introduces **Spatial MAPPO (S-MAPPO)**, which resolves the state dimension explosion bottleneck identified in the MAPPO diagnostic report by replacing the flat MLP centralized critic with a 2D Convolutional Neural Network (CNN) Centralized Critic $V_{\phi}(S_{\text{spatial}})$.

---

## 1. Motivation & Mathematical Formulation

In standard MAPPO, the Centralized Value Network receives a 1D flattened global state vector $S$. As warehouse grid dimensions $(H, W)$ and robot count $N$ grow, flat MLP parameter complexity scales as $O(H \times W)$, destroying spatial translation equivariance and entity permutation invariance.

**Spatial MAPPO (S-MAPPO)** represents global warehouse state as a 5-channel 2D spatial grid tensor $S_{\text{spatial}} \in \mathbb{R}^{5 \times H \times W}$:

- **Channel 0**: Robot positions & normalized battery levels
- **Channel 1**: Shelf locations
- **Channel 2**: Static obstacles & warehouse perimeter walls
- **Channel 3**: Charging station locations
- **Channel 4**: Active package pickup and drop targets

### CNN Centralized Value Network $V_{\phi}(S_{\text{spatial}})$
$$\mathcal{L}^{\text{S-MAPPO}}(\phi) = \frac{1}{B} \sum_{i=1}^B \left( V_{\phi}\left(\text{Conv2D}\left(S_{\text{spatial}, i}\right)\right) - \hat{R}_i \right)^2$$

Using 2D Convolutional blocks followed by `AdaptiveAvgPool2d((4, 4))`, the parameter count of $V_{\phi}(S_{\text{spatial}})$ remains **$O(1)$ constant** relative to warehouse grid dimensions and robot fleet size $N$.

---

## 2. CTDE Paradigm Guarantees

- **Centralized Training (CNN Critic)**: The spatial critic $V_{\phi}(S_{\text{spatial}})$ processes full 5-channel 2D grid tensors during optimization.
- **Decentralized Execution (Unchanged Actor)**: Decentralized robot actors $\pi_{\theta}(a_i | o_i)$ receive **only** their local observations $o_i$ (neighborhood grid, local battery, assigned task), ensuring **zero privileged information leakage**.

---

## 3. Package Architecture (`marl/algorithms/spatial_mappo/`)

- [config.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/config.py): `SpatialMAPPOConfig` dataclass (`num_agents`, `cnn_channels=5`, `conv_filters=(32, 64, 128)`, `hidden_dim=128`).
- [spatial_encoder.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/spatial_encoder.py): `WarehouseSpatialEncoder` constructing 5-channel 2D spatial grid state tensors.
- [cnn_critic.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/cnn_critic.py): `CNNCentralizedCritic` with Conv2D blocks + `AdaptiveAvgPool2d((4, 4))` supporting variable warehouse sizes.
- [feature_visualizer.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/feature_visualizer.py): `SpatialFeatureVisualizer` exporting activation heatmaps to `runs/visualizations/`.
- [agent.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/agent.py): `SpatialMAPPOAgent` encapsulating actor policy and buffer.
- [rollout_manager.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/rollout_manager.py): `SpatialMAPPORolloutManager` collecting local obs for actors AND 5-channel 2D spatial tensors for `CNNCentralizedCritic`.
- [batch_builder.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/batch_builder.py): `SpatialMAPPOBatchBuilder` constructing mini-batches.
- [metrics.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/metrics.py): `SpatialMAPPOMetricsTracker` aggregating metrics.
- [evaluator.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/evaluator.py): `SpatialMAPPOEvaluator` evaluating multi-robot fleets (1 to 32 agents).
- [checkpoint.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/checkpoint.py): `SpatialMAPPOCheckpointHandler` serializing actor and CNN critic weights.
- [trainer.py](file:///d:/PG/summer%20training/MARL/marl/algorithms/spatial_mappo/trainer.py): `SpatialMAPPOTrainer` orchestrating S-MAPPO CTDE training loops.

---

## 4. Usage Code Example

```python
from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.spatial_mappo import SpatialMAPPOConfig, SpatialMAPPOTrainer

# 1. Initialize PettingZoo Parallel Environment
env_cfg = MultiAgentEnvConfig(
    num_robots=4,
    grid_width=12,
    grid_height=12,
    enable_reward_shaping=True,
    enable_action_masking=True,
)
env = WarehouseParallelEnv(config=env_cfg)

# 2. Instantiate Spatial MAPPO Trainer (CNN Centralized Critic V(S_spatial))
smappo_config = SpatialMAPPOConfig(
    num_agents=4,
    cnn_channels=5,
    conv_filters=(32, 64, 128),
    actor_lr=3e-4,
    critic_lr=5e-4,
    batch_size=400,
)
trainer = SpatialMAPPOTrainer(env=env, config=smappo_config)

# 3. Train Spatial MAPPO Fleet
trainer.train(total_timesteps=20000)

# 4. Evaluate Decentralized Actors
eval_metrics = trainer.evaluate(num_episodes=10)
print(f"Spatial MAPPO 4-Robot Fleet Reward: {eval_metrics['eval_mean_reward']:.2f} | Fairness: {eval_metrics['eval_jains_fairness']:.2f}")

env.close()
```

---

## 5. Benchmark Comparison (IPPO vs. MAPPO MLP vs. Spatial MAPPO CNN)

- **Benchmark Plot**: Saved to `runs/benchmarks/spatial_mappo_benchmark.png`
- **Summary Metrics JSON**: Saved to `runs/benchmarks/spatial_mappo_summary.json`
