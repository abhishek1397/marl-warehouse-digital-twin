"""PolicyEvaluator executing environment evaluation episodes and capturing trajectory telemetry."""

from typing import List, Optional

import torch

from marl import WarehouseGymEnv
from marl.networks.policy_network import PolicyNetwork
from research.trajectory_recorder import EpisodeTrajectory, TrajectoryRecorder
from simulator.position import Position


class PolicyEvaluator:
    """Evaluates policy network performance on WarehouseGymEnv and records trajectory telemetry."""

    def __init__(self, env: WarehouseGymEnv, policy: Optional[PolicyNetwork] = None) -> None:
        self.env: WarehouseGymEnv = env
        self.policy: Optional[PolicyNetwork] = policy
        self.recorder: TrajectoryRecorder = TrajectoryRecorder()

    def evaluate_policy(
        self,
        num_episodes: int = 10,
        seed: Optional[int] = 42,
        deterministic: bool = True,
    ) -> List[EpisodeTrajectory]:
        """Runs evaluation episodes and records full step telemetry for each episode."""
        if self.policy is not None:
            self.policy.eval()

        trajectories: List[EpisodeTrajectory] = []

        for ep_idx in range(num_episodes):
            ep_seed = seed + ep_idx if seed is not None else None
            obs, info = self.env.reset(seed=ep_seed)

            self.recorder.start_episode(episode_id=ep_idx)
            done = False
            step_count = 0

            while not done:
                mask = info.get("action_mask", None)

                if self.policy is not None:
                    with torch.no_grad():
                        action = self.policy.predict(obs, mask=mask, deterministic=deterministic)
                    action_int = int(action.item() if isinstance(action, torch.Tensor) and action.numel() == 1 else action)
                else:
                    # Random Policy Baseline
                    action_int = int(self.env.action_space.sample())

                next_obs, reward, terminated, truncated, info = self.env.step(action_int)
                step_count += 1

                # Extract telemetry fields
                robot = self.env.robot
                env_rew = float(info.get("env_reward", reward))
                pot_rew = float(info.get("potential_reward", 0.0))
                is_valid = bool(info.get("action_valid", True))
                is_collision = not is_valid

                # Detect pickup and delivery events
                is_pickup = False
                is_delivery = False
                action_msg = str(info.get("action_message", ""))
                if "Picked up package" in action_msg:
                    is_pickup = True
                if "Delivered package" in action_msg:
                    is_delivery = True

                self.recorder.record_step(
                    timestep=step_count,
                    position=Position(robot.position.x, robot.position.y) if robot else Position(0, 0),
                    action=action_int,
                    reward=reward,
                    env_reward=env_rew,
                    potential_reward=pot_rew,
                    battery_level=float(robot.battery_level) if robot else 100.0,
                    carrying_package=robot.carrying_package is not None if robot else False,
                    goal_position=robot.assigned_task.drop_position if robot and robot.carrying_package and robot.assigned_task else (robot.assigned_task.pickup_position if robot and robot.assigned_task else None),
                    task_status=str(robot.assigned_task.status) if robot and robot.assigned_task else "IDLE",
                    is_collision=is_collision,
                    is_pickup=is_pickup,
                    is_delivery=is_delivery,
                )

                obs = next_obs
                done = terminated or truncated

            trajectories.append(self.recorder.finish_episode())

        return trajectories
