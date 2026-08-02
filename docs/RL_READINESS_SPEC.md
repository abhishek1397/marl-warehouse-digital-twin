# Multi-Agent Reinforcement Learning (MARL) Readiness Specification

This document provides the formal architectural specification for wrapping the Warehouse Digital Twin simulator into a Gymnasium / PettingZoo MARL environment interface (`PettingZoo.ParallelEnv`).

---

## 1. Environment Interface Overview

The simulator engine (`SimulationEngine`) is designed for wrapping into a multi-agent environment where each Autonomous Mobile Robot (AMR) acts as an individual learning agent.

```
+-------------------------------------------------------------------------+
|                  PettingZoo Multi-Agent ParallelEnv                      |
+-------------------------------------------------------------------------+
       |                                                    ^
       | step({agent_id: action})                           | obs, reward,
       v                                                    | done, info
+-------------------------------------------------------------------------+
|                   Warehouse Digital Twin Engine                         |
|      (Grid, Robots, Tasks, ReservationTable, TrafficController)         |
+-------------------------------------------------------------------------+
```

---

## 2. Action Space

- **Type**: `gymnasium.spaces.Discrete(5)` per agent.
- **Actions**:
  0. `0`: Move **NORTH** `(0, -1)`
  1. `1`: Move **SOUTH** `(0, 1)`
  2. `2`: Move **EAST** `(1, 0)`
  3. `3`: Move **WEST** `(-1, 0)`
  4. `4`: **STAY / WAIT** `(0, 0)`

---

## 3. Observation Space

Each robot agent receives a structured local grid observation combined with vector features:

- **Type**: `gymnasium.spaces.Dict` per agent containing:
  1. `local_grid`: `Box(low=0, high=5, shape=(7, 7), dtype=int32)`
     - 7x7 spatial egocentric grid window centered on agent position.
     - Cell Encodings: `0` Empty, `1` Obstacle/Shelf, `2` Other Robot, `3` Target Location, `4` Charging Dock, `5` Self.
  2. `agent_state`: `Box(shape=(6,), dtype=float32)`
     - `[normalized_x, normalized_y, battery_pct, is_carrying_package, target_dx, target_dy]`

---

## 4. Reward Signal Design

To facilitate cooperative MARL algorithms (e.g. MAPPO, QMIX, IPPO), rewards are split into sparse goal rewards and dense shaping rewards:

| Event / State Transition | Individual Reward | Team Shared Reward | Description |
| :--- | :---: | :---: | :--- |
| **Package Pickup** | `+2.0` | `+1.0` | Agent reaches assigned package pickup location. |
| **Package Delivery** | `+10.0` | `+5.0` | Agent completes delivery at drop destination. |
| **Step Penalty** | `-0.1` | `0.0` | Small cost per step to encourage minimum-time routes. |
| **Wait Penalty** | `-0.2` | `0.0` | Penalty for idle waiting when path is open. |
| **Collision (Vertex/Swap)** | `-5.0` | `-2.0` | Penalty for attempted movement into an occupied cell or swap. |
| **Battery Depletion Failure** | `-10.0` | `-5.0` | Agent runs out of battery outside a charging station. |
| **Recharge Completed** | `+1.0` | `0.0` | Agent successfully replenishes battery to 100%. |

---

## 5. Episode Termination & Truncation

An episode terminates (`dones[agent_id] = True` or `truncations[agent_id] = True`) under the following conditions:

1. **Terminal Completion**: All warehouse tasks in `TaskManager` have reached status `COMPLETED`.
2. **Step Limit Truncation**: Simulation steps reach `max_episode_steps` (e.g., 500 steps).
3. **Catastrophic Failure**: Any robot battery drops to 0.0% or unrecoverable deadlock occurs.

---

## 6. Reset & Deterministic Seeding

- **Reset Signature**: `env.reset(seed: Optional[int] = None, options: Optional[dict] = None)`
- **Deterministic Seeding**:
  - `seed(seed_value)` initializes Python `random`, NumPy RNG, and warehouse entity placement generator.
  - Calling `reset(seed=42)` guarantees identical layout topology, initial robot spawn positions, and package delivery queues across runs.

---

## 7. Gap Analysis & Missing Functionality

Prior to building the PettingZoo environment wrapper in Phase 4, the following minor extensions will be added:
1. **Gymnasium/PettingZoo Dependency**: Install `gymnasium` and `pettingzoo` packages into `marl_env`.
2. **Egocentric Grid Extractor**: A helper method in `Grid` (`get_local_window(pos, window_size=7)`) to generate egocentric observation matrices.
3. **Reward Calculator Component**: A dedicated `RewardCalculator` class evaluating state transitions during `step()`.
