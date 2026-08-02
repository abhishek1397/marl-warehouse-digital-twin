"""AgentManager managing multi-agent identities and simulator Robot mappings."""

from typing import Dict, List, Optional

from simulator.robot import Robot


class AgentManager:
    """Manages PettingZoo agent string IDs and maps them to simulator Robot objects."""

    def __init__(self, num_robots: int = 3) -> None:
        self.num_robots: int = num_robots
        self.possible_agents: List[str] = [f"robot_{i}" for i in range(num_robots)]
        self.agents: List[str] = list(self.possible_agents)

        # Maps agent_id -> Robot instance
        self._robot_map: Dict[str, Robot] = {}

    def initialize_agents(self, fleet: Dict[str, Robot]) -> None:
        """Initializes agent mapping from active simulator fleet."""
        self._robot_map.clear()
        self.agents = list(self.possible_agents)

        for agent_id, robot in fleet.items():
            if agent_id in self.possible_agents:
                self._robot_map[agent_id] = robot

    def get_robot(self, agent_id: str) -> Optional[Robot]:
        """Retrieves Robot instance for a given agent_id."""
        return self._robot_map.get(agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Removes an agent from active agents list upon termination."""
        if agent_id in self.agents:
            self.agents.remove(agent_id)

    def active_agents(self) -> List[str]:
        """Returns list of currently active agent string IDs."""
        return list(self.agents)

    def reset(self) -> None:
        """Resets active agent list to all possible agents."""
        self.agents = list(self.possible_agents)
        self._robot_map.clear()
