# Rollout Buffer, Trajectories & GAE Subsystem Documentation (`marl/storage/`)

This document details `marl/storage/`, the trajectory collection, Generalized Advantage Estimation (GAE), return computation, rollout buffer, and mini-batch sampling subsystem.

---

## 1. Storage Architecture & Data Flow

```
+------------------+     Appends     +------------------+
|    Transition    | --------------> |    Trajectory    |
| (obs, act, rew,  |                 | (list of steps,  |
|  v, log_prob...) |                 |  return, length) |
+--------+---------+                 +--------+---------+
         |                                    |
         +-----------------+------------------+
                           | Inserts into
                           v
                 +-------------------+
                 |   RolloutBuffer   |
                 | (compute_gae,     |
                 |  advantages, ret) |
                 +---------+---------+
                           | Slices via MiniBatchSampler
                           v
                 +-------------------+
                 |    Batch Tensor   | ---> (Feeds into PPO / MAPPO Optimizers)
                 | (obs, act, adv,   |
                 |  ret, log_probs)  |
                 +-------------------+
```

---

## 2. Generalized Advantage Estimation (GAE) Derivation

Following Schulman et al. (2015/2017), the temporal difference error $\delta_t$ and GAE advantage $A_t^{\text{GAE}(\gamma, \lambda)}$ are defined as:

$$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_{t+1}) - V(s_t)$$

$$A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} = \delta_t + \gamma \lambda (1 - d_{t+1}) A_{t+1}^{\text{GAE}(\gamma, \lambda)}$$

$$\hat{R}_t = A_t^{\text{GAE}(\gamma, \lambda)} + V(s_t)$$

where $d_t$ is true for terminal state transitions.

---

## 3. Class Specifications

| Class / Module | Description |
| :--- | :--- |
| `Transition` | Dataclass storing a single step experience (`observation`, `action`, `reward`, `next_observation`, `terminated`, `truncated`, `value_estimate`, `log_prob`, `agent_id`, `step`). |
| `Trajectory` | Container holding an ordered sequence of transitions for an episode. |
| `Batch` | Tensor batch container storing mini-batch slices for PyTorch optimizers (`observations`, `actions`, `advantages`, `returns`, `values`, `old_log_probs`, `masks`). |
| `compute_gae` | Vectorized $O(T)$ GAE advantage and target return calculation. |
| `RolloutBuffer` | Rollout buffer storing transitions, computing GAE advantages, and generating PyTorch mini-batches. |
| `MiniBatchSampler` | Mini-batch sampler shuffling and slicing buffer dataset over $N$ training epochs. |
| `BufferStatistics` | Telemetry analytics calculating mean reward, advantage stats, and buffer utilization. |

---

## 4. Usage Code Examples

### Collecting Trajectories & Computing Advantages

```python
from marl.storage import RolloutBuffer, Transition

buffer = RolloutBuffer(capacity=10000, device="cpu")

# 1. Insert transitions during environment rollout
for step in range(200):
    trans = Transition(
        observation=obs,
        action=action,
        reward=reward,
        value_estimate=value_float,
        log_prob=log_prob_float,
        agent_id="robot_0",
        step=step,
    )
    buffer.insert(trans)

# 2. Compute GAE advantages and target returns
buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95, normalize_adv=True)

# 3. Iterate over mini-batches for PPO / MAPPO optimization epochs
for batch in buffer.get_generator(mini_batch_size=64, num_epochs=4):
    print("Batch obs shape:", batch.observations.shape)
    print("Batch advantages shape:", batch.advantages.shape)
    # Feed directly into PPO / MAPPO loss functions...
```
