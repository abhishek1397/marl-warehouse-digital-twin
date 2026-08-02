# PyTorch Neural Network Library Documentation (`marl/networks/`)

This document details `marl/networks/`, the modular, production-quality PyTorch neural network library supporting MARL algorithms (PPO, MAPPO, QMIX, IPPO, DQN, SAC).

---

## 1. Class Hierarchy & Architecture Overview

All network modules inherit from `BaseNetwork` (which inherits from PyTorch `nn.Module`).

```
                              +-----------------------+
                              |      BaseNetwork      |
                              |      (nn.Module)      |
                              +-----------+-----------+
                                          |
        +------------------+--------------+-------------+------------------+
        |                  |                            |                  |
   +----+---+     +--------+--------+          +--------+--------+   +-----+-----+
   |  MLP   |     |  CNNFeature     |          | FeatureExtractor|   | Distribution|
   +--------+     |   Extractor     |          +--------+--------+   +-----------+
                  +-----------------+                   |
                                          +-------------+-------------+
                                          |                           |
                                   +------+------+             +------+------+
                                   | ActorNetwork|             |CriticNetwork|
                                   +------+------+             +------+------+
                                          |                           |
                                          +-------------+-------------+
                                                        |
                                            +-----------+-----------+
                                            | SharedActorCritic     |
                                            +-----------------------+
```

---

## 2. Module Specifications

| Module Name | Inherits From | Key Purpose / Output |
| :--- | :--- | :--- |
| `BaseNetwork` | `nn.Module` | Abstract base class with `save()`, `load()`, `to_device()`, `count_parameters()`, `weight_statistics()`, `get_summary()`. |
| `MLP` | `BaseNetwork` | Configurable multi-layer perceptron with layer/batch norm, dropout, and residual connections. |
| `CNNFeatureExtractor` | `BaseNetwork` | 2D convolutional feature extractor for spatial grid/image inputs. |
| `FeatureExtractor` | `BaseNetwork` | Unified feature extractor processing vector states, 2D image grids, or Gymnasium `Dict` observations. |
| `ActorNetwork` | `BaseNetwork` | Policy actor head producing action logits and `CategoricalDistribution` objects. |
| `CriticNetwork` | `BaseNetwork` | Value critic head producing scalar state values $V(s)$ or state-action values $Q(s, a)$. |
| `SharedActorCritic` | `BaseNetwork` | Shared feature backbone with dedicated Actor and Critic heads. |
| `ValueNetwork` | `BaseNetwork` | Standalone state-value $V(s)$ network. |
| `PolicyNetwork` | `BaseNetwork` | High-level policy wrapper supporting `act()`, `evaluate_actions()`, and `predict()`. |
| `NetworkFactory` | Standard Class | Factory pattern instantiating networks by string name (`"mlp"`, `"actor"`, `"critic"`, `"shared_actor_critic"`). |

---

## 3. Usage Code Examples

### Direct Instantiation

```python
import torch
from marl.networks import ActorNetwork, CriticNetwork, SharedActorCritic

# 1. Discrete Actor Network
actor = ActorNetwork(observation_space=64, action_dim=8)
dist = actor(torch.randn(32, 64))
actions = dist.sample()

# 2. Value Critic Network
critic = CriticNetwork(observation_space=64, action_dim=None)
values = critic(torch.randn(32, 64))  # Shape (32, 1)

# 3. Shared Actor-Critic Network
shared_ac = SharedActorCritic(observation_space=64, action_dim=8)
dist, values = shared_ac(torch.randn(32, 64))
```

### Factory Pattern Instantiation

```python
from marl.networks import NetworkFactory

# Create network from factory string name
actor = NetworkFactory.create("actor", observation_space=64, action_dim=8)
critic = NetworkFactory.create("critic", observation_space=64)
shared_ac = NetworkFactory.create("shared_actor_critic", observation_space=64, action_dim=8)
```

---

## 4. Extension Guide for Future RL Algorithms

- **PPO / IPPO**: Reuse `ActorNetwork` and `CriticNetwork` or `PolicyNetwork`.
- **MAPPO**: Reuse `SharedActorCritic` for centralized critics with decentralized actors.
- **QMIX**: Reuse `CriticNetwork(action_dim=8)` for joint action-value $Q(s, a)$ estimation.
- **SAC / Continuous Control**: Use `NormalDistribution` and `IndependentDistribution` wrappers with continuous policy heads.
