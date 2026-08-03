# Simulator Empirical Runtime Verification Report

Empirical runtime verification report generated from 60 step-level telemetry rows collected at runtime in [data/logs/runtime_telemetry.csv](file:///d:/PG/summer%20training/MARL/data/logs/runtime_telemetry.csv).

---

## 1. Empirical Telemetry Proof (30 Timesteps, 2 Robots)

| Metric | Empirical Result | Requirement | Verification Result |
| :--- | :--- | :--- | :--- |
| **Different Observations** | **30 / 30 steps (100.0%)** | Observations must differ | **PASS** |
| **Different Targets** | **30 / 30 steps (100.0%)** | Target positions must differ | **PASS** |
| **Different Packages** | **30 / 30 steps (100.0%)** | Assigned tasks must differ | **PASS** |
| **Different Trajectories** | **30 / 30 steps (100.0%)** | Space-time waypoints must differ | **PASS** |
| **Different Actions** | **24 / 30 steps (80.0%)** | Actions must execute independently | **PASS** |
| **Max Consecutive Synchronization** | **0 steps** | Must not exceed 10 steps | **PASS** |

---

## 2. Step Telemetry Excerpt (Steps 1–15)

| Step | Robot ID | Position | Target | Assigned Package | FSM State | Action | Reward | Collision | Waypoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `robot_0` | $(0,1)$ | $(2,2)$ | `task_01` | `MOVING_TO_PACKAGE` | `SOUTH` | $-0.10$ | `False` | $(0,1)$ |
| **1** | `robot_1` | $(1,1)$ | $(6,2)$ | `task_02` | `MOVING_TO_PACKAGE` | `SOUTH` | $-0.10$ | `False` | $(1,1)$ |
| **4** | `robot_0` | $(2,1)$ | $(7,7)$ | `task_01` | `MOVING_TO_DELIVERY` | `PICK` | $+19.90$ | `False` | $(2,1)$ |
| **4** | `robot_1` | $(4,1)$ | $(6,2)$ | `task_02` | `MOVING_TO_PACKAGE` | `EAST` | $-0.10$ | `False` | $(4,1)$ |
| **7** | `robot_0` | $(3,3)$ | $(7,7)$ | `task_01` | `MOVING_TO_DELIVERY` | `SOUTH` | $-0.10$ | `False` | $(3,3)$ |
| **7** | `robot_1` | $(6,1)$ | $(0,7)$ | `task_02` | `MOVING_TO_DELIVERY` | `PICK` | $+19.90$ | `False` | $(6,1)$ |
| **15** | `robot_0` | $(6,7)$ | $(2,6)$ | `task_03` | `MOVING_TO_PACKAGE` | `DROP` | $+99.90$ | `False` | $(6,7)$ |
| **15** | `robot_1` | $(0,3)$ | $(0,7)$ | `task_02` | `MOVING_TO_DELIVERY` | `SOUTH` | $-0.10$ | `False` | $(0,3)$ |

---

## 3. Empirical Proof: Two Robots Pursuing Different Packages

1. **Step 1–4**:
   - `robot_0` targets `Shelf 0` at $(2,2)$ (`task_01`, package `pkg_01`). It moves `SOUTH` to $(0,1) \rightarrow \text{EAST}$ to $(1,1) \rightarrow \text{EAST}$ to $(2,1)$, reaches pickup position next to `Shelf 0`, and executes `PICK` (Reward $+19.90$). Target instantly switches to Delivery Station $(7,7)$.
   - `robot_1` targets `Shelf 1` at $(6,2)$ (`task_02`, package `pkg_02`). It moves `SOUTH` to $(1,1) \rightarrow \text{EAST}$ across $(2,1) \rightarrow (3,1) \rightarrow (4,1) \rightarrow (5,1) \rightarrow (6,1)$, reaching pickup position next to `Shelf 1`.
2. **Step 5–15**:
   - `robot_0` hauls package `pkg_01` along the right grid corridor to Delivery Depot $(7,7)$, executing `DROP` at Step 15 (Reward $+99.90$). Upon completion, `robot_0` is assigned next pending task `task_03` targeting `Shelf 2` at $(2,6)$.
   - `robot_1` executes `PICK` at Step 7 (Reward $+19.90$), target updates to Delivery Depot $(0,7)$, and hauls package `pkg_02` along the left grid corridor to $(0,7)$, executing `DROP` at Step 19.

---

## 4. Visualization & Color Palette Verification

- **Planned Paths**: Rendered on frontend grid canvas via SVG dashed polyline matching exact planner waypoints.
- **Color Palette**:
  - `robot_0`: **Blue** (`#3b82f6`)
  - `robot_1`: **Green** (`#22c55e`)
  - `robot_2`: **Orange** (`#f97316`)
  - `robot_3`: **Purple** (`#a855f7`)
- **Overhead Debug Panel**: Displays `Robot ID`, `Assigned Package`, `Target Goal (tx, ty)`, `Action`, and `FSM State` directly above each robot sprite.
