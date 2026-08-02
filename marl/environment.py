"""WarehouseGymEnv Gymnasium environment wrapping the Warehouse Digital Twin simulator."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np

from marl.action import ActionMapper, ActionResult
from marl.config import EnvConfig
from marl.episode import EpisodeManager
from marl.observation import ObservationEncoder
from marl.rendering import EnvironmentRenderer
from marl.reward import RewardEngine
from marl.spaces import get_action_space, get_observation_space
from marl.utils import set_seed
from simulator.battery_manager import BatteryManager
from simulator.cell import CellType
from simulator.charging_station import ChargingStation
from simulator.metrics import MetricsCollector
from simulator.obstacle import Obstacle
from simulator.package import Package
from simulator.position import Position
from simulator.robot import Robot
from simulator.shelf import Shelf
from simulator.task import Task, TaskPriority, TaskType
from simulator.task_manager import TaskManager
from simulator.warehouse import Warehouse


from marl.action_masking import ActionMaskConfig, ActionMaskGenerator
from marl.reward_shaping import RewardShapingConfig, ShapedRewardEngine

class WarehouseGymEnv(gym.Env):
    """Gymnasium single-agent / multi-agent compatible environment for Warehouse Digital Twin."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        super().__init__()
        self.config: EnvConfig = config or EnvConfig()
        self.render_mode: Optional[str] = self.config.render_mode

        # Define Gymnasium spaces
        self.action_space: Discrete = get_action_space()
        self.observation_space: gym.spaces.Dict = get_observation_space(self.config)

        # Core sub-modules
        self.observation_encoder: ObservationEncoder = ObservationEncoder(self.config)
        self.action_mapper: ActionMapper = ActionMapper()
        self.reward_engine: RewardEngine = RewardEngine(self.config)
        self.shaped_reward_engine: ShapedRewardEngine = ShapedRewardEngine(
            base_reward_engine=self.reward_engine,
            config=RewardShapingConfig(
                enable_reward_shaping=self.config.enable_reward_shaping,
                shaping_scale=self.config.shaping_scale,
                gamma=self.config.shaping_gamma,
            ),
        )
        self.mask_generator: ActionMaskGenerator = ActionMaskGenerator(
            config=ActionMaskConfig(enable_action_masking=self.config.enable_action_masking)
        )
        self.episode_manager: EpisodeManager = EpisodeManager(self.config)
        self.renderer: EnvironmentRenderer = EnvironmentRenderer()

        # Simulator references
        self._warehouse: Optional[Warehouse] = None
        self._task_manager: Optional[TaskManager] = None
        self._battery_manager: Optional[BatteryManager] = None
        self._metrics_collector: Optional[MetricsCollector] = None
        self._robot: Optional[Robot] = None
        self._fleet: Dict[str, Robot] = {}
        self._charging_stations: Dict[str, ChargingStation] = {}
        self._shelves: Dict[str, Shelf] = {}

        # Set initial seed
        self.seed(self.config.seed)

    @property
    def warehouse(self) -> Optional[Warehouse]:
        """Returns internal warehouse instance."""
        return self._warehouse

    @property
    def robot(self) -> Optional[Robot]:
        """Returns primary agent robot instance."""
        return self._robot

    def seed(self, seed: Optional[int] = None) -> None:
        """Sets random seed for deterministic reproduction."""
        self.config.seed = seed
        self.py_rng, self.np_rng = set_seed(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets environment state for a new episode.

        Returns:
            Tuple of (observation_dict, info_dict).
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # Re-initialize simulator components
        w, h = self.config.grid_width, self.config.grid_height
        self._warehouse = Warehouse(width=w, height=h, name="GymWarehouse")
        self._task_manager = TaskManager()
        self._battery_manager = BatteryManager()
        self._metrics_collector = MetricsCollector()
        self._fleet.clear()
        self._charging_stations.clear()
        self._shelves.clear()

        # 1. Place Charging Station at corner
        station_pos = Position(0, h - 1)
        station = ChargingStation("charger_0", station_pos, charge_rate=20.0, capacity=2)
        self._warehouse.place_object(station_pos, station, cell_type=CellType.CHARGING_STATION)
        self._charging_stations["charger_0"] = station

        # 2. Place Shelves & Obstacles
        shelf_pos = Position(w // 2, 2)
        shelf = Shelf("shelf_0", shelf_pos, capacity=10)
        self._warehouse.place_object(shelf_pos, shelf, cell_type=CellType.SHELF)
        self._shelves["shelf_0"] = shelf

        obs_pos = Position(w // 2, 3)
        obs = Obstacle("obs_0", obs_pos)
        self._warehouse.place_object(obs_pos, obs, cell_type=CellType.OBSTACLE)

        # 3. Spawn Primary Robot Agent
        robot_pos = Position(0, 0)
        self._robot = Robot("agent_0", initial_position=robot_pos, max_battery=100.0)
        self._fleet["agent_0"] = self._robot
        self._warehouse.place_object(robot_pos, self._robot)

        # 4. Spawn Tasks & Packages
        for t_idx in range(1, self.config.task_count + 1):
            src_pos = Position(w // 2, 2)
            dst_pos = Position(w - 1, h - 1)
            pkg = Package(f"pkg_{t_idx:02d}", source_position=src_pos, destination_position=dst_pos)

            self._task_manager.create_task(
                task_id=f"task_{t_idx:02d}",
                task_type=TaskType.PICKUP_AND_DELIVER,
                pickup_position=src_pos,
                drop_position=dst_pos,
                package=pkg,
                priority=TaskPriority.MEDIUM,
            )

        # Assign initial task to robot
        initial_task = self._task_manager.assign_next_task(self._robot)

        # Reset episode manager
        self.episode_manager.start_episode()

        # Encode initial observation
        obs = self.observation_encoder.encode(
            robot=self._robot,
            warehouse=self._warehouse,
            fleet=self._fleet,
            task=initial_task,
            charging_stations=self._charging_stations,
        )

        mask_obj = self.mask_generator.generate_mask(
            robot=self._robot,
            warehouse=self._warehouse,
            task=initial_task,
            charging_stations=self._charging_stations,
        )

        info = {
            "step": 0,
            "cumulative_reward": 0.0,
            "assigned_task_id": initial_task.task_id if initial_task else None,
            "action_mask": mask_obj.mask_array,
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Executes a single step transition in the environment.

        Args:
            action: Discrete action index (0 to 7).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        if self._robot is None or self._warehouse is None or self._task_manager is None:
            raise RuntimeError("Environment must be reset before calling step().")

        current_task = self._robot.assigned_task
        robot_prev_pos = Position(self._robot.position.x, self._robot.position.y)
        robot_prev_pkg = self._robot.carrying_package
        robot_prev_battery = self._robot.battery_level

        # Create lightweight prev robot state copy for potential computation
        robot_prev = Robot(robot_id=self._robot.robot_id, initial_position=robot_prev_pos)
        if robot_prev_pkg is not None:
            robot_prev.pick_up_package(robot_prev_pkg)
        robot_prev.battery_level = robot_prev_battery

        # 1. Execute action via ActionMapper
        action_result: ActionResult = self.action_mapper.execute_action(
            action=action,
            robot=self._robot,
            warehouse=self._warehouse,
            task=current_task,
            charging_stations=self._charging_stations,
        )

        # 2. Update battery consumption
        self._battery_manager.update_robot_battery(self._robot)

        # 3. Calculate shaped reward
        shaped_out = self.shaped_reward_engine.calculate_reward(
            action_result=action_result,
            robot_prev=robot_prev,
            robot_next=self._robot,
            task=current_task,
            charging_stations=self._charging_stations,
        )
        reward = shaped_out.total_reward

        # 4. Check if robot needs next task assignment after task completion
        if current_task and current_task.is_terminal():
            self._robot.clear_task()

        if self._robot.is_idle():
            self._task_manager.assign_next_task(self._robot)

        # 5. Check episode status
        terminated, truncated, info = self.episode_manager.check_step(
            reward, self._robot, self._task_manager
        )

        # 6. Encode new observation
        obs = self.observation_encoder.encode(
            robot=self._robot,
            warehouse=self._warehouse,
            fleet=self._fleet,
            task=self._robot.assigned_task,
            charging_stations=self._charging_stations,
        )

        mask_obj = self.mask_generator.generate_mask(
            robot=self._robot,
            warehouse=self._warehouse,
            task=self._robot.assigned_task,
            charging_stations=self._charging_stations,
        )

        info["action_message"] = action_result.message
        info["action_valid"] = action_result.is_valid
        info["action_mask"] = mask_obj.mask_array

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray | str]:
        """Renders environment state based on self.render_mode."""
        if self.render_mode is None or self._warehouse is None:
            return None
        return self.renderer.render(self.render_mode, self._warehouse, self._fleet)

    def close(self) -> None:
        """Cleans up environment resources."""
        self._fleet.clear()
        self._warehouse = None
