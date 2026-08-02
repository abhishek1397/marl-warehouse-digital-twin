"""AgentActionMapper executing joint actions dictionary and detecting multi-agent collisions."""

from typing import Dict, List, Optional, Set, Tuple

from marl.action import ActionMapper, ActionResult
from marl.multi_agent_config import MultiAgentEnvConfig
from simulator.charging_station import ChargingStation
from simulator.constants import Direction
from simulator.position import Position
from simulator.robot import Robot
from simulator.shelf import Shelf
from simulator.task import Task
from simulator.warehouse import Warehouse


class AgentActionMapper:
    """Executes joint action dictionaries and detects multi-agent vertex and swap collisions."""

    def __init__(self, config: MultiAgentEnvConfig) -> None:
        self.config: MultiAgentEnvConfig = config
        self.single_mapper: ActionMapper = ActionMapper()

    def execute_joint_actions(
        self,
        actions: Dict[str, int],
        fleet: Dict[str, Robot],
        warehouse: Warehouse,
        tasks: Dict[str, Optional[Task]],
        charging_stations: Dict[str, ChargingStation],
        shelves: Dict[str, Shelf],
    ) -> Dict[str, ActionResult]:
        """Applies joint actions to the fleet simultaneously while checking for inter-agent collisions.

        Args:
            actions: Map of agent_id -> action index (0 to 7).
            fleet: Active robot fleet.
            warehouse: Warehouse environment.
            tasks: Map of agent_id -> assigned Task instance, if any.
            charging_stations: Map of ChargingStation instances.
            shelves: Map of Shelf instances.

        Returns:
            Map of agent_id -> ActionResult.
        """
        results: Dict[str, ActionResult] = {}

        # Record initial positions
        initial_positions: Dict[str, Position] = {
            a_id: robot.position for a_id, robot in fleet.items()
        }
        intended_positions: Dict[str, Position] = {}

        # 1. Evaluate single-agent actions & compute intended target positions
        for agent_id, action in actions.items():
            robot = fleet.get(agent_id)
            if robot is None:
                continue

            current_task = tasks.get(agent_id)

            # Pre-compute intended target position for movement actions (0 - 3)
            if action in self.single_mapper.ACTION_MAP and action != 4:
                direction = self.single_mapper.ACTION_MAP[action]
                intended_pos = robot.position.get_neighbor(direction)
            else:
                intended_pos = robot.position

            intended_positions[agent_id] = intended_pos

            res = self.single_mapper.execute_action(
                action=action,
                robot=robot,
                warehouse=warehouse,
                task=current_task,
                charging_stations=charging_stations,
            )
            results[agent_id] = res

        # 2. Detect multi-agent vertex collisions (multiple agents in same cell)
        vertex_occupancy: Dict[Position, List[str]] = {}
        for agent_id, robot in fleet.items():
            vertex_occupancy.setdefault(robot.position, []).append(agent_id)

        collided_agents: Set[str] = set()
        for pos, occupying_agents in vertex_occupancy.items():
            if len(occupying_agents) > 1:
                for a_id in occupying_agents:
                    collided_agents.add(a_id)

        # 3. Detect multi-agent edge-swap collisions (posA <-> posB)
        agent_list = list(fleet.keys())
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a1, a2 = agent_list[i], agent_list[j]
                init1, curr1 = initial_positions[a1], fleet[a1].position
                init2, curr2 = initial_positions[a2], fleet[a2].position

                if init1 == curr2 and curr1 == init2 and init1 != curr1:
                    collided_agents.add(a1)
                    collided_agents.add(a2)

        # 4. Rollback positions for agents involved in multi-agent collisions
        for a_id in collided_agents:
            robot = fleet[a_id]
            robot.position = initial_positions[a_id]  # Restore initial position
            results[a_id] = ActionResult(
                action=actions.get(a_id, 4),
                is_valid=False,
                is_collision=True,
                message=f"Inter-agent collision detected for robot '{a_id}'. Position restored.",
            )

        return results
