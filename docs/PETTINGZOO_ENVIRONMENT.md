# PettingZoo Multi-Agent Reinforcement Learning Environment Documentation (`WarehouseParallelEnv`)

This document provides complete documentation for `WarehouseParallelEnv`, the PettingZoo Parallel Environment wrapper (`pettingzoo.parallel.ParallelEnv`) wrapping the Warehouse Digital Twin simulator.

---

## 1. Environment Architecture

`WarehouseParallelEnv` converts the multi-robot warehouse simulation into a multi-agent environment where every Autonomous Mobile Robot (AMR) acts as an independent learning agent.

```
+-----------------------------------------------------------------------------+
|                  PettingZoo WarehouseParallelEnv                            |
+-----------------------------------------------------------------------------+
   |                                                                     ^
   | step({"robot_0": a_0, "robot_1": a_1, "robot_2": a_2})              | obs_dict, reward_dict,
   v                                                                     | term_dict, trunc_dict, info
+-----------------------------------------------------------------------------+
|                     Warehouse Digital Twin Simulator                        |
|   (Grid, Fleet, Tasks, Inter-Agent Collision Detector, Communication)       |
+-----------------------------------------------------------------------------+
```

---

## 2. PettingZoo Parallel API Usage Example

`WarehouseParallelEnv` is compatible out-of-the-box with MARL libraries such as RLlib, CleanRL, TorchRL, MAPPO, QMIX, and Independent PPO.

```python
from marl import MultiAgentEnvConfig, WarehouseParallelEnv

# Instantiate environment with custom configuration
config = MultiAgentEnvConfig(
    num_robots=3,
    num_tasks=10,
    observation_mode="hybrid",  # "local", "global", or "hybrid"
    reward_mode="hybrid",        # "individual", "team", or "hybrid"
    comm_mode="radius",         # "none", "broadcast", or "radius"
    seed=42,
)

env = WarehouseParallelEnv(config=config)
observations, infos = env.reset(seed=42)

while env.agents:
    # Select joint actions for all active agents
    actions = {
        agent: env.action_space(agent).sample() for agent in env.agents
    }
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Global state vector for MAPPO/QMIX centralized critics
    global_state = env.state()
    
    print(f"Step Active Agents: {env.agents} | Mean Reward: {sum(rewards.values())/len(rewards):.2f}")

env.close()
```

---

## 3. Agents & Action Space

- **Agent Identifiers**: String IDs `robot_0`, `robot_1`, ..., `robot_{N-1}`.
- **Action Space**: `gymnasium.spaces.Discrete(8)` per agent.

| Index | Action Name | Description |
| :---: | :--- | :--- |
| `0` | **Move Up** | Moves robot North `(dx=0, dy=-1)`. |
| `1` | **Move Down** | Moves robot South `(dx=0, dy=1)`. |
| `2` | **Move Left** | Moves robot West `(dx=-1, dy=0)`. |
| `3` | **Move Right** | Moves robot East `(dx=1, dy=0)`. |
| `4` | **Wait** | Waits in current grid cell. |
| `5` | **Pick Package** | Picks up assigned package if at pickup location. |
| `6` | **Drop Package** | Drops carried package if at drop destination. |
| `7` | **Go Charge** | Docks at charging station if present at station position. |

---

## 4. Observation Modes

Selectable via `config.observation_mode`:

1. **`local` Mode**:
   - `robot_position`: `Box(2,)`
   - `goal_position`: `Box(2,)`
   - `battery_level`: `Box(1,)`
   - `package_status`: `Box(1,)`
   - `local_occupancy`: `Box(7, 7)` ($7 \times 7$ egocentric local grid window)
   - `charging_station_distance`: `Box(1,)`
   - `comm_message`: `Box(comm_msg_dim,)`

2. **`global` Mode**:
   - Contains all local features plus full `global_fleet_summary` array containing positions and battery levels of all robots in the fleet.

3. **`hybrid` Mode**:
   - Combines local egocentric views with global fleet summaries for centralized training with decentralized execution (CTDE).

---

## 5. Reward Modes

Selectable via `config.reward_mode`:

1. **`individual`**: Each agent receives a reward strictly based on its own action outcome and state transition.
2. **`team`**: All active agents receive the mean reward across the entire fleet to promote team cooperation.
3. **`hybrid`**: $R_{\text{agent}} = (1 - w) \cdot R_{\text{individual}} + w \cdot R_{\text{team}}$, balanced by `config.team_reward_weight`.

---

## 6. Communication Channels

Selectable via `config.comm_mode`:

1. **`none`**: No inter-agent communication.
2. **`broadcast`**: Every agent receives incoming message vectors from all other agents.
3. **`radius`**: Agents receive incoming message vectors from other agents within `config.comm_radius` Manhattan distance.

---

## 7. Official PettingZoo Test Compliance

`WarehouseParallelEnv` passes the official PettingZoo parallel API test suite:

```python
from pettingzoo.test import parallel_api_test
from marl import WarehouseParallelEnv

env = WarehouseParallelEnv()
parallel_api_test(env, num_cycles=100)
```
