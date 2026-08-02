# Gymnasium Reinforcement Learning Environment Documentation (`WarehouseGymEnv`)

This document provides complete documentation for `WarehouseGymEnv`, the Gymnasium-compatible Reinforcement Learning environment wrapping the Warehouse Digital Twin simulator.

---

## 1. Overview & Standard API Usage

`WarehouseGymEnv` complies fully with standard Gymnasium environment specifications (`gymnasium.Env`).

### Basic Usage Example

```python
import gymnasium as gym
from marl import EnvConfig, WarehouseGymEnv

# Option A: Direct instantiation
config = EnvConfig(grid_width=20, grid_height=20, max_episode_steps=200, seed=42)
env = WarehouseGymEnv(config=config)

# Option B: Gymnasium Registry
# env = gym.make("Warehouse-v0")

obs, info = env.reset(seed=42)

for step in range(100):
    action = env.action_space.sample()  # Select action (0-7)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step} | Reward: {reward:.2f} | Robot Battery: {obs['battery_level'][0]:.1f}%")
    
    if terminated or truncated:
        print("Episode finished. Resetting...")
        obs, info = env.reset()

env.close()
```

---

## 2. Action Space

- **Type**: `gymnasium.spaces.Discrete(8)`
- **Discrete Action Index Mapping**:

| Index | Action Name | Description |
| :---: | :--- | :--- |
| `0` | **Move Up** | Moves robot North `(dx=0, dy=-1)`. |
| `1` | **Move Down** | Moves robot South `(dx=0, dy=1)`. |
| `2` | **Move Left** | Moves robot West `(dx=-1, dy=0)`. |
| `3` | **Move Right** | Moves robot East `(dx=1, dy=0)`. |
| `4` | **Wait** | Remains at current position and increments idle steps. |
| `5` | **Pick Package** | Picks up assigned package if at pickup location. |
| `6` | **Drop Package** | Drops carried package if at drop destination. |
| `7` | **Go Charge** | Docks at charging station if present at station position. |

---

## 3. Observation Space

- **Type**: `gymnasium.spaces.Dict`

| Observation Key | Space Type | Shape | Values / Description |
| :--- | :--- | :---: | :--- |
| `robot_position` | `Box(low=0, high=max_dim)` | `(2,)` | `[x, y]` integer grid coordinates of robot. |
| `goal_position` | `Box(low=0, high=max_dim)` | `(2,)` | `[x, y]` target coordinates (pickup or drop position). |
| `battery_level` | `Box(low=0.0, high=100.0)` | `(1,)` | Current battery percentage (`0.0` to `100.0`). |
| `package_status` | `Box(low=0, high=3)` | `(1,)` | `0`: None, `1`: Pickup, `2`: Carrying/In-Transit, `3`: Delivered. |
| `local_occupancy` | `Box(low=0, high=5)` | `(7, 7)` | Egocentric $7 \times 7$ local grid window. Encodings: `0` Empty, `1` Obstacle, `2` Shelf, `3` Charger, `4` Other Robot, `5` Out of Bounds/Self. |
| `charging_station_distance` | `Box(low=0.0, high=2*max_dim)`| `(1,)` | Manhattan distance to nearest charging station. |
| `task_status` | `Box(low=0, high=4)` | `(1,)` | `0`: None, `1`: Created, `2`: Assigned, `3`: In-Progress, `4`: Completed. |

---

## 4. Configurable Reward Structure

Step rewards are dynamically computed by `RewardEngine` using values defined in `EnvConfig`:

| Event / Action Outcome | Default Weight | Config Field |
| :--- | :---: | :--- |
| **Step Time Penalty** | `-0.1` | `step_time_penalty` |
| **Successful Package Delivery** | `+100.0` | `successful_delivery_reward` |
| **Package Pickup** | `+20.0` | `package_pickup_reward` |
| **Collision (Wall / Obstacle)** | `-50.0` | `collision_penalty` |
| **Invalid Action** | `-10.0` | `invalid_action_penalty` |
| **Idle Waiting Penalty** | `-1.0` | `waiting_penalty` |
| **Battery Depletion Failure** | `-100.0` | `battery_empty_penalty` |
| **Successful Charging** | `+5.0` | `successful_charging_reward` |

---

## 5. Episode Lifecycle & Termination

- **Episode Reset**: `obs, info = env.reset(seed=seed)` initializes grid topology, robot placement, packages, and task queue deterministically.
- **Terminated (`terminated = True`)**:
  - All assigned warehouse tasks reached `COMPLETED` status (Success).
  - Robot battery dropped to `0.0%` (Failure).
- **Truncated (`truncated = True`)**:
  - Cumulative step count reached `max_episode_steps` (Timeout).

---

## 6. Rendering Modes

- **Human Mode (`render_mode="human"`)**: Returns formatted ASCII text grid matrix.
- **RGB Array Mode (`render_mode="rgb_array"`)**: Returns a uint8 NumPy image array `(H * cell_size, W * cell_size, 3)` suitable for visual displays and web dashboards.
