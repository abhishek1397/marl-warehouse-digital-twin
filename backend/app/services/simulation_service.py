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

            for agent_id, info in infos.items():
                if not info.get("action_valid", True):
                    self.total_collisions += 1
                if "Delivered package" in str(info.get("action_message", "")):
                    self.total_deliveries += 1

            self.obs_dict, self.info_dict = next_obs, infos

        return self.get_state()

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

            if "IDLE" in state_str:
                idle_count += 1

            pkg_id = robot.assigned_task.task_id if robot.assigned_task else None
            r_id = getattr(robot, "robot_id", getattr(robot, "id", k))

            robots_list.append(
                RobotStateSchema(
                    id=r_id,
                    position=pos,
                    battery_level=b_level,
                    state=state_str,
                    assigned_task=pkg_id,
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
