"""CommunicationManager handling non-learning environment communication buffers."""

from typing import Dict, List, Optional

import numpy as np

from marl.multi_agent_config import MultiAgentEnvConfig
from simulator.position import Position
from simulator.robot import Robot


class CommunicationManager:
    """Manages environment-level agent communication buffers (none, broadcast, radius)."""

    def __init__(self, config: MultiAgentEnvConfig) -> None:
        self.config: MultiAgentEnvConfig = config

        # Message buffer: agent_id -> np.ndarray of shape (comm_msg_dim,)
        self._message_buffer: Dict[str, np.ndarray] = {}

    def set_message(self, agent_id: str, message: np.ndarray) -> None:
        """Stores a communication message from an agent."""
        if self.config.comm_mode == "none":
            return
        self._message_buffer[agent_id] = np.asarray(message, dtype=np.float32)

    def get_received_messages(
        self, agent_id: str, agent_pos: Position, fleet: Dict[str, Robot]
    ) -> np.ndarray:
        """Retrieves aggregated or stacked incoming messages for an agent based on comm_mode.

        Returns:
            NumPy array of shape (comm_msg_dim,).
        """
        msg_dim = self.config.comm_msg_dim
        zero_msg = np.zeros(msg_dim, dtype=np.float32)

        if self.config.comm_mode == "none" or not self._message_buffer:
            return zero_msg

        received_msgs: List[np.ndarray] = []

        for other_id, msg in self._message_buffer.items():
            if other_id == agent_id:
                continue

            if self.config.comm_mode == "broadcast":
                received_msgs.append(msg)
            elif self.config.comm_mode == "radius":
                other_robot = fleet.get(other_id)
                if other_robot and agent_pos.manhattan_distance(other_robot.position) <= self.config.comm_radius:
                    received_msgs.append(msg)

        if not received_msgs:
            return zero_msg

        # Return elementwise mean of received messages
        return np.mean(received_msgs, axis=0, dtype=np.float32)

    def clear(self) -> None:
        """Clears all message buffers."""
        self._message_buffer.clear()
