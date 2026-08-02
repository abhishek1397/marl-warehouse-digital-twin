"""PettingZoo Parallel Environment wrapper for Multi-Agent Reinforcement Learning."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from functools import lru_cache
import gymnasium as gym
from gymnasium.spaces import Discrete, Space
from pettingzoo import ParallelEnv

from marl.action_masking import ActionMaskGenerator
from marl.agent_action import AgentActionMapper, ActionResult
from marl.agent_manager import AgentManager
from marl.agent_observation import AgentObservationEncoder
from marl.agent_reward import AgentRewardEngine
from marl.communication import CommunicationManager
from marl.multi_agent_config import MultiAgentEnvConfig
from marl.rendering import EnvironmentRenderer
from marl.spaces import get_action_space
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


class WarehouseParallelEnv(ParallelEnv):
    """PettingZoo Parallel API environment wrapper for multi-agent warehouse robot coordination."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "warehouse_parallel_v0"}

    def __init__(self, config: Optional[MultiAgentEnvConfig] = None) -> None:
        super().__init__()
        self.config: MultiAgentEnvConfig = config or MultiAgentEnvConfig()
        self.render_mode: Optional[str] = self.config.render_mode

        # Core PettingZoo Agent Management
        self.agent_manager: AgentManager = AgentManager(self.config.num_robots)
        self.possible_agents: List[str] = list(self.agent_manager.possible_agents)
        self.agents: List[str] = list(self.possible_agents)

        # Multi-Agent Sub-modules
        self.obs_encoder: AgentObservationEncoder = AgentObservationEncoder(self.config)
        self.action_mapper: AgentActionMapper = AgentActionMapper(self.config)
        self.reward_engine: AgentRewardEngine = AgentRewardEngine(self.config)
        self.comm_manager: CommunicationManager = CommunicationManager(self.config)
        self.mask_generator: ActionMaskGenerator = ActionMaskGenerator()
        self.renderer: EnvironmentRenderer = EnvironmentRenderer()

        # Build Dict Spaces per Agent
        self._action_spaces: Dict[str, Discrete] = {
            agent: get_action_space() for agent in self.possible_agents
        }
        self._observation_spaces: Dict[str, Space] = {
            agent: self.obs_encoder.get_observation_space(agent) for agent in self.possible_agents
        }

        # Simulator instance references
        self._warehouse: Optional[Warehouse] = None
        self._task_manager: Optional[TaskManager] = None
        self._battery_manager: Optional[BatteryManager] = None
        self._metrics_collector: Optional[MetricsCollector] = None
        self._fleet: Dict[str, Robot] = {}
        self._charging_stations: Dict[str, ChargingStation] = {}
        self._shelves: Dict[str, Shelf] = {}

        self._current_step: int = 0

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Space:
        """Returns observation space for a specific agent."""
        return self._observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Space:
        """Returns action space for a specific agent."""
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Any]]]:
        """Resets the environment for all agents.

        Returns:
            Tuple of (observations_dict, infos_dict) keyed by active agent IDs.
        """
        if seed is not None:
            self.config.seed = seed
        set_seed(self.config.seed)

        self._current_step = 0
        self.agent_manager.reset()
        self.agents = list(self.possible_agents)
        self.comm_manager.clear()

        # 1. Re-build simulator environment
        w, h = self.config.grid_width, self.config.grid_height
        self._warehouse = Warehouse(width=w, height=h, name="ParallelGymWarehouse")
        self._task_manager = TaskManager()
        self._battery_manager = BatteryManager()
        self._metrics_collector = MetricsCollector()
        self._fleet.clear()
        self._charging_stations.clear()
        self._shelves.clear()

        # 2. Spawn Charging Stations
        station_pos = Position(0, h - 1)
        station = ChargingStation("charger_0", station_pos, charge_rate=20.0, capacity=self.config.num_robots)
        self._warehouse.place_object(station_pos, station, cell_type=CellType.CHARGING_STATION)
        self._charging_stations["charger_0"] = station

        # 3. Spawn Shelves & Obstacles
        shelf_pos = Position(w // 2, 2)
        shelf = Shelf("shelf_0", shelf_pos, capacity=20)
        self._warehouse.place_object(shelf_pos, shelf, cell_type=CellType.SHELF)
        self._shelves["shelf_0"] = shelf

        obs_pos = Position(w // 2, 3)
        obs = Obstacle("obs_0", obs_pos)
        self._warehouse.place_object(obs_pos, obs, cell_type=CellType.OBSTACLE)

        # 4. Spawn Fleet Robots (robot_0, robot_1, ..., robot_n)
        for idx, agent_id in enumerate(self.possible_agents):
            r_pos = Position(idx % w, idx // w)
            robot = Robot(agent_id, initial_position=r_pos, max_battery=100.0)
            self._fleet[agent_id] = robot
            self._warehouse.place_object(r_pos, robot)

        self.agent_manager.initialize_agents(self._fleet)

        # 5. Spawn Tasks & Packages
        for t_idx in range(1, self.config.num_tasks + 1):
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

        # Assign initial tasks to idle robots
        for robot in self._fleet.values():
            if robot.is_idle():
                self._task_manager.assign_next_task(robot)

        # Encode initial observations and infos
        observations: Dict[str, Dict[str, np.ndarray]] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        for agent_id in self.agents:
            robot = self.agent_manager.get_robot(agent_id)
            if robot:
                observations[agent_id] = self.obs_encoder.encode(
                    robot=robot,
                    warehouse=self._warehouse,
                    fleet=self._fleet,
                    comm_manager=self.comm_manager,
                    task=robot.assigned_task,
                    charging_stations=self._charging_stations,
                )
                info_dict = {"step": 0, "status": "active"}
                if self.config.enable_action_masking:
                    mask_obj = self.mask_generator.generate_mask(
                        robot=robot,
                        warehouse=self._warehouse,
                        task=robot.assigned_task,
                        charging_stations=self._charging_stations,
                    )
                    info_dict["action_mask"] = mask_obj.mask_array
                infos[agent_id] = info_dict

        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Executes a joint multi-agent step transition.

        Args:
            actions: Dictionary mapping active agent_id -> discrete action index (0 to 7).

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos) for all active agents.
        """
        if not actions:
            self.agents.clear()
            return {}, {}, {}, {}, {}

        self._current_step += 1

        # 1. Map & execute joint actions with inter-agent collision detection
        tasks_map = {
            a_id: (self._fleet[a_id].assigned_task if a_id in self._fleet else None)
            for a_id in self.agents
        }
        action_results = self.action_mapper.execute_joint_actions(
            actions=actions,
            fleet=self._fleet,
            warehouse=self._warehouse,
            tasks=tasks_map,
            charging_stations=self._charging_stations,
            shelves=self._shelves,
        )

        # 2. Update battery status for all active robots
        self._battery_manager.update_fleet(self._fleet)

        # 3. Assign next pending task to robots that finished current task
        for agent_id in self.agents:
            robot = self._fleet.get(agent_id)
            if robot:
                if robot.assigned_task and robot.assigned_task.is_terminal():
                    robot.clear_task()
                if robot.is_idle():
                    self._task_manager.assign_next_task(robot)

        # 4. Compute joint rewards (individual, team, or hybrid)
        rewards = self.reward_engine.calculate_joint_rewards(action_results, self._fleet)

        # 5. Evaluate termination and truncation flags
        global_success = self._task_manager.all_tasks_completed() and self._task_manager.total_tasks > 0
        is_timeout = self._current_step >= self.config.max_episode_steps

        observations: Dict[str, Dict[str, np.ndarray]] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        agents_at_start = list(self.agents)

        for agent_id in agents_at_start:
            robot = self._fleet.get(agent_id)
            if robot is None:
                continue

            is_battery_empty = robot.battery_level <= 0.0
            term = global_success or is_battery_empty
            trunc = is_timeout and not term

            terminations[agent_id] = term
            truncations[agent_id] = trunc

            obs = self.obs_encoder.encode(
                robot=robot,
                warehouse=self._warehouse,
                fleet=self._fleet,
                comm_manager=self.comm_manager,
                task=robot.assigned_task,
                charging_stations=self._charging_stations,
            )
            observations[agent_id] = obs

            res = action_results.get(agent_id, ActionResult(action=4))
            info_dict = {
                "step": self._current_step,
                "action_message": res.message,
                "action_valid": res.is_valid,
                "is_collision": res.is_collision,
            }
            if self.config.enable_action_masking:
                mask_obj = self.mask_generator.generate_mask(
                    robot=robot,
                    warehouse=self._warehouse,
                    task=robot.assigned_task,
                    charging_stations=self._charging_stations,
                )
                info_dict["action_mask"] = mask_obj.mask_array
            infos[agent_id] = info_dict

            if term or trunc:
                self.agent_manager.remove_agent(agent_id)

        self.agents = list(self.agent_manager.active_agents())

        return observations, rewards, terminations, truncations, infos

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """Returns current observation dictionary for a specified agent."""
        robot = self.agent_manager.get_robot(agent)
        if robot is None or self._warehouse is None:
            return {}
        return self.obs_encoder.encode(
            robot=robot,
            warehouse=self._warehouse,
            fleet=self._fleet,
            comm_manager=self.comm_manager,
            task=robot.assigned_task,
            charging_stations=self._charging_stations,
        )

    def state(self) -> np.ndarray:
        """Returns global environment state array for centralized critics in MAPPO / QMIX."""
        if self._warehouse is None:
            return np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.float32)
        return self.obs_encoder.extract_global_state(self._warehouse, self._fleet)

    def render(self) -> Optional[np.ndarray | str]:
        """Renders environment state based on self.render_mode."""
        if self.render_mode is None or self._warehouse is None:
            return None
        return self.renderer.render(self.render_mode, self._warehouse, self._fleet)

    def close(self) -> None:
        """Cleans up environment resources."""
        self._fleet.clear()
        self.agents.clear()
        self._warehouse = None
