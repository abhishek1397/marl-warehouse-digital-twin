"""Comprehensive test suite for research multi-seed statistical evaluation suite."""

import os
import pytest
import numpy as np

from marl import EnvConfig, WarehouseGymEnv
from marl.algorithms.ppo import PPOConfig, PPOTrainer
from research.confidence_intervals import ConfidenceIntervalCalculator
from research.evaluation_tables import EvaluationTableGenerator
from research.generalization_suite import GeneralizationEvaluator
from research.latex_tables import LaTeXTableGenerator
from research.multi_seed_runner import MultiSeedExperimentRunner
from research.plot_generator import PublicationPlotGenerator
from research.report_generator import ReportGenerator
from research.statistical_analysis import StatisticalAnalyzer


def test_confidence_intervals() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, ci_l, ci_u = ConfidenceIntervalCalculator.compute_t_ci(data)

    assert mean == 3.0
    assert ci_l < mean < ci_u

    mean_b, b_l, b_u = ConfidenceIntervalCalculator.compute_bootstrap_ci(data, num_bootstraps=100)
    assert mean_b == 3.0
    assert b_l <= mean_b <= b_u


def test_statistical_analyzer() -> None:
    s1 = [10.0, 12.0, 14.0, 16.0, 18.0]
    s2 = [1.0, 2.0, 3.0, 4.0, 5.0]

    stats = StatisticalAnalyzer.compute_descriptive_stats(s1)
    assert stats["mean"] == 14.0
    assert stats["std"] > 0.0

    sig = StatisticalAnalyzer.perform_significance_tests(s1, s2)
    assert "t_statistic" in sig
    assert "p_value_ttest" in sig
    assert "cohens_d" in sig
    assert sig["cohens_d"] > 0.0


def test_latex_table_generator() -> None:
    arm_metrics = {
        "Baseline PPO": {"mean_reward": -3658.0, "success_rate": 0.0, "mean_collisions": 200.0, "mean_completion_time": 0.0},
        "PPO + PBRS + DAM": {"mean_reward": -3.0, "success_rate": 1.0, "mean_collisions": 0.0, "mean_completion_time": 24.0},
    }

    perf_tex = LaTeXTableGenerator.generate_performance_latex_table(arm_metrics)
    assert r"\begin{table}" in perf_tex
    assert r"Baseline PPO" in perf_tex

    stats_tex = LaTeXTableGenerator.generate_statistical_summary_latex_table({"mean": -3.0})
    assert r"\begin{table}" in stats_tex

    gen_tex = LaTeXTableGenerator.generate_generalization_latex_table({"8x8": {"success_rate": 1.0}})
    assert r"\begin{table}" in gen_tex


def test_evaluation_tables_generator() -> None:
    stats_tbl = EvaluationTableGenerator.generate_statistical_summary_table({"mean": 10.0})
    assert "Statistic Metric" in stats_tbl

    abl_tbl = EvaluationTableGenerator.generate_ablation_summary_table({"Arm1": {"mean_reward": 5.0}})
    assert "Experimental Arm" in abl_tbl


def test_plot_generator(tmp_path) -> None:
    p_lc = os.path.join(tmp_path, "lc.png")
    PublicationPlotGenerator.plot_learning_curves_with_ci([1, 2], [[1.0, 2.0], [1.5, 2.5]], p_lc)
    assert os.path.exists(p_lc)

    p_bv = os.path.join(tmp_path, "bv.png")
    PublicationPlotGenerator.plot_reward_box_violin({"Arm1": [1.0, 2.0], "Arm2": [3.0, 4.0]}, p_bv)
    assert os.path.exists(p_bv)

    p_cdf = os.path.join(tmp_path, "cdf.png")
    PublicationPlotGenerator.plot_cdf_distribution({"Arm1": [1.0, 2.0]}, p_cdf)
    assert os.path.exists(p_cdf)

    p_hm = os.path.join(tmp_path, "hm.png")
    PublicationPlotGenerator.plot_generalization_heatmap({"8x8": {"success_rate": 1.0}}, p_hm)
    assert os.path.exists(p_hm)


def test_report_generator(tmp_path) -> None:
    rep_path = os.path.join(tmp_path, "report.md")
    ReportGenerator.generate_multi_seed_report(
        stats={"mean": -3.0},
        sig_tests={"t_statistic": 10.0, "p_value_ttest": 0.001, "wilcoxon_statistic": 0.0, "p_value_wilcoxon": 0.001, "cohens_d": 5.0},
        gen_results={"8x8": {"success_rate": 1.0}},
        latex_tables={"performance_table": "% tex"},
        output_path=rep_path,
    )
    assert os.path.exists(rep_path)


def test_generalization_suite() -> None:
    env_cfg = EnvConfig(grid_width=6, grid_height=6, enable_action_masking=True)
    env = WarehouseGymEnv(config=env_cfg)
    trainer = PPOTrainer(env=env, config=PPOConfig(learning_rate=3e-4, epochs=1, batch_size=200, mini_batch_size=64))

    gen_res = GeneralizationEvaluator.evaluate_zero_shot_generalization(
        policy=trainer.policy, grid_sizes=[(6, 6)], num_episodes=1, seed=42
    )
    assert "6x6" in gen_res

    rob_res = GeneralizationEvaluator.evaluate_robustness_sweep(
        policy=trainer.policy, task_counts=[1], max_steps_list=[80], num_episodes=1, seed=42
    )
    assert "task_count_sweep" in rob_res
    env.close()


def test_multi_seed_runner(tmp_path) -> None:
    base_dir = os.path.join(tmp_path, "test_exp")
    res = MultiSeedExperimentRunner.run_multi_seed_experiments(
        seeds=[42], total_timesteps=200, base_dir=str(base_dir)
    )
    assert 42 in res
    assert os.path.exists(os.path.join(base_dir, "seed_42", "config.yaml"))
    assert os.path.exists(os.path.join(base_dir, "seed_42", "evaluation.json"))
