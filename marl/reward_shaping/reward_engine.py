"""ShapedRewardEngine wrapping base reward calculation with Ng et al. Potential-Based Reward Shaping."""

from dataclasses import dataclass
from typing import Dict, Optional

from marl.action import ActionResult
from marl.reward import RewardEngine
from marl.reward_shaping.config import RewardShapingConfig
from marl.reward_shaping.potential import ManhattanPotential, PotentialFunction
from marl.reward_shaping.utils import calculate_goal_progress, calculate_shaping_reward
from simulator.charging_station import ChargingStation
from simulator.robot import Robot
from simulator.task import Task


@dataclass
class ShapedRewardOutput:
    """Dataclass storing individual reward components and progress metrics."""

    env_reward: float
    potential_reward: float
    total_reward: float
    distance_improvement: float
    goal_progress: float


class ShapedRewardEngine:
    """Refactored Reward Engine combining raw environment rewards with potential-based shaping rewards."""

    def __init__(
        self,
        base_reward_engine: RewardEngine,
        config: Optional[RewardShapingConfig] = None,
        potential_fn: Optional[PotentialFunction] = None,
    ) -> None:
        self.base_engine: RewardEngine = base_reward_engine
        self.config: RewardShapingConfig = config or RewardShapingConfig()
        self.potential_fn: PotentialFunction = potential_fn or ManhattanPotential()

    def calculate_reward(
        self,
        action_result: ActionResult,
        robot_prev: Robot,
        robot_next: Robot,
        task: Optional[Task] = None,
        charging_stations: Optional[Dict[str, ChargingStation]] = None,
    ) -> ShapedRewardOutput:
        """Calculates raw environment reward, potential reward, and combined total reward.

        Formulas:
            Phi(s) = -ManhattanDistance(robot_prev, active_goal)
            Phi(s') = -ManhattanDistance(robot_next, active_goal)
            F(s, a, s') = scale * (gamma * Phi(s') - Phi(s))
            R_total = R_env + F(s, a, s')
        """
        # 1. Base environment reward calculation
        env_reward = float(self.base_engine.calculate_reward(action_result, robot_next))

        if not self.config.enable_reward_shaping:
            return ShapedRewardOutput(
                env_reward=env_reward,
                potential_reward=0.0,
                total_reward=env_reward,
                distance_improvement=0.0,
                goal_progress=0.0,
            )

        # 2. Compute state potentials Phi(s) and Phi(s')
        phi_prev = self.potential_fn.compute_potential(
            robot=robot_prev, task=task, charging_stations=charging_stations
        )
        phi_next = self.potential_fn.compute_potential(
            robot=robot_next, task=task, charging_stations=charging_stations
        )

        # 3. Compute potential-based shaping reward F(s, a, s') = scale * (gamma * Phi(s') - Phi(s))
        potential_reward = calculate_shaping_reward(
            phi_current=phi_prev,
            phi_next=phi_next,
            gamma=self.config.gamma,
            scale=self.config.shaping_scale,
        )

        # 4. Total combined reward
        total_reward = float(env_reward + potential_reward)

        # Distance progress metrics (dist = -phi)
        dist_prev = -phi_prev
        dist_next = -phi_next
        progress = calculate_goal_progress(dist_prev, dist_next)

        return ShapedRewardOutput(
            env_reward=env_reward,
            potential_reward=potential_reward,
            total_reward=total_reward,
            distance_improvement=progress,
            goal_progress=progress,
        )
