"""TrajectoryRecorder module recording step-by-step state, action, and reward telemetry."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from simulator.position import Position


@dataclass
class TrajectoryStep:
    """Dataclass storing step-level execution telemetry."""

    timestep: int
    position: Position
    action: int
    reward: float
    env_reward: float
    potential_reward: float
    battery_level: float
    carrying_package: bool
    goal_position: Optional[Position]
    task_status: str
    is_collision: bool
    is_pickup: bool
    is_delivery: bool


@dataclass
class EpisodeTrajectory:
    """Dataclass storing full episode trajectory telemetry."""

    episode_id: int
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    total_env_reward: float = 0.0
    total_potential_reward: float = 0.0
    episode_length: int = 0
    is_success: bool = False
    total_collisions: int = 0
    total_pickups: int = 0
    total_deliveries: int = 0


class TrajectoryRecorder:
    """Records step-by-step evaluation trajectories for policy analysis."""

    def __init__(self) -> None:
        self.current_trajectory: Optional[EpisodeTrajectory] = None

    def start_episode(self, episode_id: int) -> None:
        """Initializes a new episode trajectory recording session."""
        self.current_trajectory = EpisodeTrajectory(episode_id=episode_id)

    def record_step(
        self,
        timestep: int,
        position: Position,
        action: int,
        reward: float,
        env_reward: float = 0.0,
        potential_reward: float = 0.0,
        battery_level: float = 100.0,
        carrying_package: bool = False,
        goal_position: Optional[Position] = None,
        task_status: str = "IDLE",
        is_collision: bool = False,
        is_pickup: bool = False,
        is_delivery: bool = False,
    ) -> None:
        """Appends a step record to the active trajectory."""
        if self.current_trajectory is None:
            raise RuntimeError("TrajectoryRecorder: start_episode() must be called before record_step().")

        step = TrajectoryStep(
            timestep=timestep,
            position=Position(position.x, position.y),
            action=action,
            reward=reward,
            env_reward=env_reward,
            potential_reward=potential_reward,
            battery_level=battery_level,
            carrying_package=carrying_package,
            goal_position=Position(goal_position.x, goal_position.y) if goal_position else None,
            task_status=task_status,
            is_collision=is_collision,
            is_pickup=is_pickup,
            is_delivery=is_delivery,
        )

        self.current_trajectory.steps.append(step)
        self.current_trajectory.total_reward += reward
        self.current_trajectory.total_env_reward += env_reward
        self.current_trajectory.total_potential_reward += potential_reward
        self.current_trajectory.episode_length += 1

        if is_collision:
            self.current_trajectory.total_collisions += 1
        if is_pickup:
            self.current_trajectory.total_pickups += 1
        if is_delivery:
            self.current_trajectory.total_deliveries += 1
            self.current_trajectory.is_success = True

    def finish_episode(self) -> EpisodeTrajectory:
        """Finalizes and returns completed episode trajectory."""
        if self.current_trajectory is None:
            raise RuntimeError("TrajectoryRecorder: No active trajectory to finish.")
        traj = self.current_trajectory
        self.current_trajectory = None
        return traj
