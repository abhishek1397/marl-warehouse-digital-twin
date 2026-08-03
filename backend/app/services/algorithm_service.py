"""AlgorithmService handling MARL policy inference, metadata registry, and MultiRobotPlanner integration."""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.core.exceptions import AlgorithmNotFoundError
from backend.app.schemas.algorithm import AlgorithmMetadataSchema


class AlgorithmService:
    """Service providing algorithm metadata, policy selection, and multi-robot action generation."""

    _active_algorithm: str = "Spatial MAPPO"
    _loaded_models: Dict[str, Any] = {}
    _planner_instance: Optional[Any] = None
    _last_planned_paths: Dict[str, List[List[int]]] = {}

    _algorithms: Dict[str, AlgorithmMetadataSchema] = {
        "A*": AlgorithmMetadataSchema(
            name="A*",
            category="Classical Planner",
            paradigm="Centralized Search",
            description="Prioritized Multi-Robot Space-Time A* Search with Time-Expanded Reservation Table.",
            action_masking=True,
            reward_shaping=False,
            actor_architecture="Space-Time Search Tree",
            critic_architecture="N/A",
        ),
        "PPO": AlgorithmMetadataSchema(
            name="PPO",
            category="Single-Agent RL",
            paradigm="Gym Baseline",
            description="Proximal Policy Optimization baseline without reward shaping or action masking.",
            action_masking=False,
            reward_shaping=False,
            actor_architecture="MLP (64x64)",
            critic_architecture="MLP (64x64)",
        ),
        "PPO + PBRS": AlgorithmMetadataSchema(
            name="PPO + PBRS",
            category="Single-Agent RL",
            paradigm="Gym + Reward Shaping",
            description="PPO with Potential-Based Reward Shaping (Ng et al., 1999) preserving policy invariance.",
            action_masking=False,
            reward_shaping=True,
            actor_architecture="MLP (64x64)",
            critic_architecture="MLP (64x64)",
        ),
        "PPO + DAM": AlgorithmMetadataSchema(
            name="PPO + DAM",
            category="Single-Agent RL",
            paradigm="Gym + Masking",
            description="PPO with Dynamic Action Masking eliminating invalid obstacle collisions.",
            action_masking=True,
            reward_shaping=True,
            actor_architecture="MLP (64x64)",
            critic_architecture="MLP (64x64)",
        ),
        "IPPO": AlgorithmMetadataSchema(
            name="IPPO",
            category="Multi-Agent RL",
            paradigm="Decentralized Actors",
            description="Independent PPO where each agent learns decentralized policy independently.",
            action_masking=True,
            reward_shaping=True,
            actor_architecture="Shared MLP (64x64)",
            critic_architecture="Decentralized MLP (64x64)",
        ),
        "MAPPO": AlgorithmMetadataSchema(
            name="MAPPO",
            category="Multi-Agent RL",
            paradigm="CTDE Flat MLP",
            description="Multi-Agent PPO with Centralized Critic evaluating concatenated 1D global state.",
            action_masking=True,
            reward_shaping=True,
            actor_architecture="Shared MLP (64x64)",
            critic_architecture="Flat Global State MLP",
        ),
        "Spatial MAPPO": AlgorithmMetadataSchema(
            name="Spatial MAPPO",
            category="Spatial MARL",
            paradigm="CTDE 2D Spatial CNN",
            description="Spatial MAPPO featuring 5-Channel 2D Spatial CNN Centralized Critic with O(1) parameter complexity.",
            action_masking=True,
            reward_shaping=True,
            actor_architecture="Shared MLP (64x64)",
            critic_architecture="5-Channel 2D Spatial CNN",
        ),
    }

    @classmethod
    def get_all_algorithms(cls) -> List[AlgorithmMetadataSchema]:
        """Returns list of all supported MARL algorithms."""
        return list(cls._algorithms.values())

    @classmethod
    def get_algorithm(cls, name: str) -> AlgorithmMetadataSchema:
        """Returns metadata for specific algorithm or raises 404 error."""
        if name not in cls._algorithms:
            raise AlgorithmNotFoundError(name)
        return cls._algorithms[name]

    @classmethod
    def set_active_algorithm(cls, name: str) -> AlgorithmMetadataSchema:
        """Sets active algorithm variant."""
        meta = cls.get_algorithm(name)
        cls._active_algorithm = name
        return meta

    @classmethod
    def get_active_algorithm(cls) -> str:
        """Returns name of active algorithm."""
        return cls._active_algorithm

    @classmethod
    def predict_actions(cls, env: Any, obs_dict: Dict[str, Any], info_dict: Dict[str, Any]) -> Dict[str, int]:
        """Generates independent, policy-driven actions for all active agents in environment.

        Ensures:
        1. Multi-Agent Independence: Independent observations, targets, and reservation paths.
        2. Dynamic Action Masking: 100% filtering of non-traversable shelf and obstacle cells.
        3. Correct Semantics: Shelves remain static; robots navigate to packages and haul to drop depots.
        """
        actions_dict: Dict[str, int] = {}
        policy_model = cls._loaded_models.get(cls._active_algorithm, None)

        # 1. If trained neural policy model exists, perform independent inference per agent
        if policy_model is not None and hasattr(policy_model, "predict"):
            for agent_id in env.agents:
                obs = obs_dict.get(agent_id)
                mask = info_dict.get(agent_id, {}).get("action_mask", None)
                if obs is not None:
                    act = policy_model.predict(obs, mask=mask)
                    actions_dict[agent_id] = int(act)
                else:
                    actions_dict[agent_id] = 4  # STAY
            return actions_dict

        # 2. Multi-Robot Space-Time A* Planner & Priority Conflict Resolver
        if cls._planner_instance is None:
            from simulator.planner import MultiRobotPlanner
            cls._planner_instance = MultiRobotPlanner()

        planner: Any = cls._planner_instance
        grid = env._warehouse.grid if hasattr(env, "_warehouse") else None

        requests = []
        for idx, agent_id in enumerate(env.agents):
            robot = env._fleet.get(agent_id) if hasattr(env, "_fleet") else None
            if robot is None:
                continue

            mask = info_dict.get(agent_id, {}).get("action_mask", None)

            # Auto-execute Pick (Action 5) or Drop (Action 6) if adjacent to target
            if mask is not None and len(mask) >= 7:
                if robot.carrying_package is not None and mask[6]:
                    actions_dict[agent_id] = 6  # Drop package
                    continue
                elif robot.carrying_package is None and mask[5]:
                    actions_dict[agent_id] = 5  # Pick package
                    continue

            # Determine independent goal position
            goal_pos = None
            if robot.assigned_task:
                task = robot.assigned_task
                if robot.carrying_package is not None:
                    goal_pos = task.drop_position
                else:
                    goal_pos = task.pickup_position
                    if grid and (not grid.is_in_bounds(goal_pos) or not grid.get_cell(goal_pos).cell_type.is_traversable):
                        neighbors = grid.get_neighbors(goal_pos, include_diagonals=False)
                        traversable = [n for n in neighbors if n.cell_type.is_traversable]
                        if traversable:
                            goal_pos = traversable[0].position

            if goal_pos is None and hasattr(env, "_shelves") and env._shelves:
                shelves = list(env._shelves.values())
                assigned_shelf = shelves[idx % len(shelves)]
                goal_pos = assigned_shelf.position
                if grid and (not grid.is_in_bounds(goal_pos) or not grid.get_cell(goal_pos).cell_type.is_traversable):
                    neighbors = grid.get_neighbors(goal_pos, include_diagonals=False)
                    traversable = [n for n in neighbors if n.cell_type.is_traversable]
                    if traversable:
                        goal_pos = traversable[0].position

            if goal_pos is not None:
                from simulator.planner import PlanningRequest
                requests.append(PlanningRequest(robot=robot, goal_position=goal_pos, priority=10 - idx))

        if grid and requests:
            joint_paths = planner.plan_joint_paths(grid=grid, requests=requests, clear_reservations=True)
            for req in requests:
                robot_id = req.robot.robot_id
                mask = info_dict.get(robot_id, {}).get("action_mask", None)
                res = joint_paths.get(robot_id)

                if res and res.success and len(res.path) > 1:
                    cls._last_planned_paths[robot_id] = [[p.x, p.y] for p in res.path]
                    next_pos = res.path[1]
                    curr_pos = req.robot.position
                    dx = next_pos.x - curr_pos.x
                    dy = next_pos.y - curr_pos.y

                    act = 4
                    if (dx, dy) == (0, -1):
                        act = 0  # NORTH (UP)
                    elif (dx, dy) == (0, 1):
                        act = 1  # SOUTH (DOWN)
                    elif (dx, dy) == (-1, 0):
                        act = 2  # WEST (LEFT)
                    elif (dx, dy) == (1, 0):
                        act = 3  # EAST (RIGHT)

                    if mask is None or (act < len(mask) and mask[act]):
                        actions_dict[robot_id] = act
                    else:
                        actions_dict[robot_id] = 4
                else:
                    actions_dict[robot_id] = 4

        # Final fallback: ensure every active agent gets a valid action
        for agent_id in env.agents:
            if agent_id not in actions_dict:
                mask = info_dict.get(agent_id, {}).get("action_mask", None)
                if mask is not None and len(mask) == env.action_space(agent_id).n and any(mask):
                    valid_actions = [i for i, valid in enumerate(mask) if valid]
                    actions_dict[agent_id] = int(valid_actions[0])
                else:
                    actions_dict[agent_id] = 4  # STAY

        return actions_dict
