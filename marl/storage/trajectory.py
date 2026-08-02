"""Trajectory dataclass representing an ordered sequence of transitions for an episode."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from marl.storage.transition import Transition


@dataclass
class Trajectory:
    """Represents a complete or partial episode trajectory containing an ordered transition list."""

    transitions: List[Transition] = field(default_factory=list)
    agent_id: str = "agent_0"
    episode_id: int = 0

    def append(self, transition: Transition) -> None:
        """Appends a transition to the trajectory."""
        self.transitions.append(transition)

    def compute_episode_length(self) -> int:
        """Returns total step length of trajectory."""
        return len(self.transitions)

    def compute_return(self, gamma: float = 1.0) -> float:
        """Computes undiscounted or discounted total episode return."""
        total_return = 0.0
        discount = 1.0
        for t in self.transitions:
            total_return += discount * t.reward
            discount *= gamma
        return float(total_return)

    def statistics(self) -> Dict[str, float]:
        """Returns statistical summary of trajectory metrics."""
        if not self.transitions:
            return {"length": 0.0, "total_reward": 0.0, "mean_reward": 0.0}

        rewards = [t.reward for t in self.transitions]
        return {
            "length": float(len(self.transitions)),
            "total_reward": float(sum(rewards)),
            "mean_reward": float(sum(rewards) / len(rewards)),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serializes trajectory data into standard dictionary format."""
        return {
            "agent_id": self.agent_id,
            "episode_id": self.episode_id,
            "length": self.compute_episode_length(),
            "total_return": self.compute_return(),
            "num_transitions": len(self.transitions),
        }
