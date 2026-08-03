"""SimulationService managing WarehouseParallelEnv simulator instances."""

from typing import Any, Dict, List, Optional
import numpy as np

from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from backend.app.core.exceptions import SimulationNotFoundError, InvalidSimulationStateError
from backend.app.schemas.simulation import (
    GridEntitySchema,
    LiveMetricsSchema,
    RobotStateSchema,
    SimulationStateResponse,
)


class SimulationService:
    """Service wrapping WarehouseParallelEnv digital twin simulator."""

    _instance: Optional["SimulationService"] = None

    def __init__(self) -> None:
        self.env: Optional[WarehouseParallelEnv] = None
        self.is_running: bool = False
        self.is_paused: bool = False
        self.step_count: int = 0
        self.total_collisions: int = 0
        self.total_deliveries: int = 0
        self.obs_dict: Dict[str, Any] = {}
        self.info_dict: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "SimulationService":
        if cls._instance is None:
            cls._instance = SimulationService()
        return cls._instance

    def create_simulation(
        self,
        grid_width: int = 8,
        grid_height: int = 8,
        num_robots: int = 2,
        enable_pbrs: bool = True,
        enable_dam: bool = True,
    ) -> SimulationStateResponse:
        """Initializes a new WarehouseParallelEnv instance."""
        if self.env is not None:
            self.env.close()

        env_cfg = MultiAgentEnvConfig(
            num_robots=num_robots,
            grid_width=grid_width,
            grid_height=grid_height,
            enable_reward_shaping=enable_pbrs,
            enable_action_masking=enable_dam,
        )
        self.env = WarehouseParallelEnv(config=env_cfg)
        self.obs_dict, self.info_dict = self.env.reset(seed=42)

        self.is_running = False
        self.is_paused = False
        self.step_count = 0
        self.total_collisions = 0
        self.total_deliveries = 0

        return self.get_state()

    def start(self) -> SimulationStateResponse:
        """Starts simulation execution."""
        if self.env is None:
            raise SimulationNotFoundError("Simulation is not initialized. Create a simulation first.")
        self.is_running = True
        self.is_paused = False
        return self.get_state()

    def pause(self) -> SimulationStateResponse:
        """Pauses simulation execution."""
        if self.env is None:
            raise SimulationNotFoundError("Simulation is not initialized.")
        self.is_paused = True
        return self.get_state()

    def reset(self) -> SimulationStateResponse:
        """Resets simulation environment."""
        if self.env is None:
            raise SimulationNotFoundError("Simulation is not initialized.")
        self.obs_dict, self.info_dict = self.env.reset(seed=42)
        self.is_running = False
        self.is_paused = False
        self.step_count = 0
        self.total_collisions = 0
        self.total_deliveries = 0
        return self.get_state()

    def step(self, steps: int = 1) -> SimulationStateResponse:
        """Steps simulation environment forward by N timesteps."""
        if self.env is None:
            raise SimulationNotFoundError("Simulation is not initialized.")

        for _ in range(steps):
            if not self.env.agents:
                self.obs_dict, self.info_dict = self.env.reset()
                break

            from backend.app.services.algorithm_service import AlgorithmService
            actions_dict = AlgorithmService.predict_actions(self.env, self.obs_dict, self.info_dict)

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions_dict)
            self.step_count += 1
            self._log_telemetry(actions_dict, rewards, infos)

            for agent_id, info in infos.items():
                if not info.get("action_valid", True):
                    self.total_collisions += 1
                if "Delivered package" in str(info.get("action_message", "")):
                    self.total_deliveries += 1

            self.obs_dict, self.info_dict = next_obs, infos

        return self.get_state()

    def _log_telemetry(self, actions_dict: Dict[str, int], rewards: Dict[str, float], infos: Dict[str, Any]) -> None:
        """Logs per-robot per-timestep telemetry to CSV file."""
        import os, csv
        log_dir = os.path.join(os.getcwd(), "data", "logs")
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "runtime_telemetry.csv")
        file_exists = os.path.exists(csv_path)

        from marl.action import ActionMapper
        action_mapper = ActionMapper()

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "step",
                    "robot_id",
                    "position",
                    "observation_summary",
                    "target_position",
                    "assigned_package",
                    "fsm_state",
                    "selected_action",
                    "reward",
                    "collision_flag",
                    "planner_waypoint",
                ])

            from backend.app.services.algorithm_service import AlgorithmService

            for k, robot in self.env._fleet.items():
                pos_str = f"({robot.position.x},{robot.position.y})"
                obs = self.obs_dict.get(k)
                obs_summary = f"pos:{pos_str}|battery:{robot.battery_level:.1f}" if obs is not None else "N/A"

                t = robot.assigned_task
                target_str = "None"
                pkg_str = t.task_id if t else "None"
                if t:
                    if getattr(robot, "has_package", False) or getattr(robot, "carrying_package", None) is not None:
                        target_str = f"({t.drop_position.x},{t.drop_position.y})"
                    else:
                        target_str = f"({t.pickup_position.x},{t.pickup_position.y})"

                state_str = str(robot.state.name if hasattr(robot.state, "name") else robot.state)
                if "CHARGE" in state_str:
                    fsm = "CHARGING"
                elif t:
                    if getattr(robot, "carrying_package", None) is not None or getattr(robot, "has_package", False):
                        fsm = "DELIVERING" if robot.position == t.drop_position else "MOVING_TO_DELIVERY"
                    else:
                        dist = abs(robot.position.x - t.pickup_position.x) + abs(robot.position.y - t.pickup_position.y)
                        fsm = "PICKING" if dist <= 1 else "MOVING_TO_PACKAGE"
                elif "IDLE" in state_str:
                    fsm = "IDLE"
                else:
                    fsm = "WAITING"

                act_idx = actions_dict.get(k, 4)
                act_name = action_mapper.ACTION_MAP.get(act_idx, "STAY").name if act_idx in action_mapper.ACTION_MAP else ("PICK" if act_idx == 5 else ("DROP" if act_idx == 6 else "STAY"))

                rw = float(rewards.get(k, 0.0))
                inf = infos.get(k, {})
                coll = not inf.get("action_valid", True)

                path = AlgorithmService._last_planned_paths.get(k, [])
                waypoint_str = f"({path[1][0]},{path[1][1]})" if path and len(path) > 1 else pos_str

                writer.writerow([
                    self.step_count,
                    k,
                    pos_str,
                    obs_summary,
                    target_str,
                    pkg_str,
                    fsm,
                    act_name,
                    f"{rw:.2f}",
                    coll,
                    waypoint_str,
                ])

    def get_state(self) -> SimulationStateResponse:
        """Constructs current simulation state response."""
        if self.env is None:
            return SimulationStateResponse(
                is_initialized=False,
                is_running=False,
                is_paused=False,
                step_count=0,
                grid_size=[0, 0],
                robots=[],
                entities=[],
                metrics=LiveMetricsSchema(
                    episode=0,
                    step=0,
                    reward=0.0,
                    throughput=0.0,
                    collisions=0,
                    idle_robots=0,
                    battery_avg=100.0,
                    packages_delivered=0,
                ),
            )

        robots_list: List[RobotStateSchema] = []
        battery_levels: List[float] = []
        idle_count = 0

        for k, robot in self.env._fleet.items():
            pos = [robot.position.x, robot.position.y]
            b_level = float(robot.battery_level)
            battery_levels.append(b_level)
            state_str = str(robot.state.name if hasattr(robot.state, "name") else robot.state)

            # Standard FSM States: IDLE, MOVING_TO_PACKAGE, PICKING, MOVING_TO_DELIVERY, DELIVERING, CHARGING, WAITING
            if "CHARGE" in state_str:
                fsm_state = "CHARGING"
            elif robot.assigned_task:
                task = robot.assigned_task
                if getattr(robot, "carrying_package", None) is not None or getattr(robot, "has_package", False):
                    if robot.position == task.drop_position:
                        fsm_state = "DELIVERING"
                    else:
                        fsm_state = "MOVING_TO_DELIVERY"
                else:
                    dist_to_pickup = abs(robot.position.x - task.pickup_position.x) + abs(robot.position.y - task.pickup_position.y)
                    if dist_to_pickup <= 1:
                        fsm_state = "PICKING"
                    else:
                        fsm_state = "MOVING_TO_PACKAGE"
            elif "IDLE" in state_str:
                fsm_state = "IDLE"
            else:
                fsm_state = "WAITING"

            if fsm_state == "IDLE":
                idle_count += 1

            pkg_id = robot.assigned_task.task_id if robot.assigned_task else None
            r_id = getattr(robot, "robot_id", getattr(robot, "id", k))

            target_pos_arr = None
            if robot.assigned_task:
                t = robot.assigned_task
                if getattr(robot, "has_package", False) or getattr(robot, "carrying_package", None) is not None:
                    target_pos_arr = [t.drop_position.x, t.drop_position.y]
                else:
                    target_pos_arr = [t.pickup_position.x, t.pickup_position.y]

            last_info = self.info_dict.get(k, {})
            last_act_name = last_info.get("action_name", "IDLE")
            is_coll = not last_info.get("action_valid", True)

            from backend.app.services.algorithm_service import AlgorithmService
            path_arr = AlgorithmService._last_planned_paths.get(k, None)

            robots_list.append(
                RobotStateSchema(
                    id=r_id,
                    position=pos,
                    battery_level=b_level,
                    state=fsm_state,
                    assigned_task=pkg_id,
                    target_position=target_pos_arr,
                    current_action=last_act_name,
                    is_collision=is_coll,
                    planned_path=path_arr,
                )
            )

        entities_list: List[GridEntitySchema] = []
        if self.env._shelves:
            for k, s in self.env._shelves.items():
                s_id = getattr(s, "shelf_id", getattr(s, "id", k))
                entities_list.append(GridEntitySchema(id=s_id, position=[s.position.x, s.position.y], type="shelf"))
        if self.env._charging_stations:
            for k, c in self.env._charging_stations.items():
                c_id = getattr(c, "station_id", getattr(c, "id", k))
                entities_list.append(GridEntitySchema(id=c_id, position=[c.position.x, c.position.y], type="charging_station"))

        avg_battery = float(np.mean(battery_levels)) if battery_levels else 100.0
        throughput = float(self.total_deliveries / max(1, self.step_count))

        return SimulationStateResponse(
            is_initialized=True,
            is_running=self.is_running,
            is_paused=self.is_paused,
            step_count=self.step_count,
            grid_size=[self.env.config.grid_width, self.env.config.grid_height],
            robots=robots_list,
            entities=entities_list,
            metrics=LiveMetricsSchema(
                episode=1,
                step=self.step_count,
                reward=float(-self.step_count * 0.1),
                throughput=throughput,
                collisions=self.total_collisions,
                idle_robots=idle_count,
                battery_avg=avg_battery,
                packages_delivered=self.total_deliveries,
            ),
        )
