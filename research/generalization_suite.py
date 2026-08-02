"""GeneralizationEvaluator conducting zero-shot layout transfer and robustness sweeps across unseen warehouse configurations."""

from typing import Any, Dict, List, Optional, Tuple

from marl import EnvConfig, WarehouseGymEnv
from marl.networks.policy_network import PolicyNetwork
from research.policy_evaluator import PolicyEvaluator
from research.success_analyzer import SuccessMetricsAnalyzer


class GeneralizationEvaluator:
    """Evaluates trained policy zero-shot across unseen grid dimensions, obstacle densities, and battery capacities."""

    GRID_SIZES: List[Tuple[int, int]] = [(8, 8), (12, 12), (16, 16), (20, 20), (24, 24)]

    @staticmethod
    def evaluate_zero_shot_generalization(
        policy: PolicyNetwork,
        grid_sizes: Optional[List[Tuple[int, int]]] = None,
        num_episodes: int = 5,
        seed: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates policy zero-shot across varied grid dimensions without retraining."""
        sizes = grid_sizes or GeneralizationEvaluator.GRID_SIZES
        results: Dict[str, Dict[str, float]] = {}

        for width, height in sizes:
            tag = f"{width}x{height}"
            env_cfg = EnvConfig(
                grid_width=width,
                grid_height=height,
                max_episode_steps=max(80, width * 4),
                seed=seed,
                enable_reward_shaping=True,
                enable_action_masking=True,
            )
            env = WarehouseGymEnv(config=env_cfg)
            evaluator = PolicyEvaluator(env=env, policy=policy)
            trajs = evaluator.evaluate_policy(num_episodes=num_episodes, seed=seed)
            metrics = SuccessMetricsAnalyzer.compute_success_metrics(trajs)
            results[tag] = metrics
            env.close()

        return results

    @staticmethod
    def evaluate_robustness_sweep(
        policy: PolicyNetwork,
        task_counts: Optional[List[int]] = None,
        max_steps_list: Optional[List[int]] = None,
        num_episodes: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Evaluates policy robustness across varied task counts and episode step horizons."""
        t_counts = task_counts or [1, 5, 10]
        step_horizons = max_steps_list or [80, 150, 200]

        results: Dict[str, Any] = {"task_count_sweep": {}, "step_horizon_sweep": {}}

        # 1. Task Count Sweep
        for tc in t_counts:
            tag = f"tasks_{tc}"
            env_cfg = EnvConfig(
                grid_width=8,
                grid_height=8,
                task_count=tc,
                seed=seed,
                enable_reward_shaping=True,
                enable_action_masking=True,
            )
            env = WarehouseGymEnv(config=env_cfg)
            evaluator = PolicyEvaluator(env=env, policy=policy)
            trajs = evaluator.evaluate_policy(num_episodes=num_episodes, seed=seed)
            results["task_count_sweep"][tag] = SuccessMetricsAnalyzer.compute_success_metrics(trajs)
            env.close()

        # 2. Step Horizon Sweep
        for horizon in step_horizons:
            tag = f"horizon_{horizon}"
            env_cfg = EnvConfig(
                grid_width=8,
                grid_height=8,
                max_episode_steps=horizon,
                seed=seed,
                enable_reward_shaping=True,
                enable_action_masking=True,
            )
            env = WarehouseGymEnv(config=env_cfg)
            evaluator = PolicyEvaluator(env=env, policy=policy)
            trajs = evaluator.evaluate_policy(num_episodes=num_episodes, seed=seed)
            results["step_horizon_sweep"][tag] = SuccessMetricsAnalyzer.compute_success_metrics(trajs)
            env.close()

        return results
