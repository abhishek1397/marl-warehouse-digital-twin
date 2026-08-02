"""StatisticalAnalyzer computing descriptive statistics, confidence intervals, and hypothesis significance tests."""

from typing import Dict, List, Union

import numpy as np
from scipy import stats

from research.confidence_intervals import ConfidenceIntervalCalculator


class StatisticalAnalyzer:
    """Computes descriptive statistics, parametric/bootstrap CIs, paired t-tests, Wilcoxon tests, and Cohen's d effect sizes."""

    @staticmethod
    def compute_descriptive_stats(data: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """Calculates comprehensive descriptive metrics across data samples."""
        arr = np.asarray(data, dtype=float)
        if len(arr) == 0:
            return {
                "mean": 0.0, "median": 0.0, "var": 0.0, "std": 0.0,
                "ci_lower": 0.0, "ci_upper": 0.0, "min": 0.0, "max": 0.0,
                "iqr": 0.0, "cv": 0.0,
            }

        mean_val, ci_lower, ci_upper = ConfidenceIntervalCalculator.compute_t_ci(arr)
        median_val = float(np.median(arr))
        var_val = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        q75, q25 = np.percentile(arr, [75, 25])
        iqr_val = float(q75 - q25)
        cv_val = float(std_val / abs(mean_val)) if abs(mean_val) > 1e-8 else 0.0

        return {
            "mean": mean_val,
            "median": median_val,
            "var": var_val,
            "std": std_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "min": min_val,
            "max": max_val,
            "iqr": iqr_val,
            "cv": cv_val,
        }

    @staticmethod
    def perform_significance_tests(
        sample1: Union[List[float], np.ndarray],
        sample2: Union[List[float], np.ndarray],
    ) -> Dict[str, float]:
        """Performs paired t-test, Wilcoxon signed-rank test, and computes Cohen's d effect size."""
        s1 = np.asarray(sample1, dtype=float)
        s2 = np.asarray(sample2, dtype=float)

        if len(s1) < 2 or len(s2) < 2 or len(s1) != len(s2):
            return {
                "t_statistic": 0.0,
                "p_value_ttest": 1.0,
                "wilcoxon_statistic": 0.0,
                "p_value_wilcoxon": 1.0,
                "cohens_d": 0.0,
            }

        # 1. Paired Student's t-test
        t_stat, p_ttest = stats.ttest_rel(s1, s2)

        # 2. Wilcoxon Signed-Rank Test
        diff = s1 - s2
        if np.all(diff == 0):
            w_stat, p_wilcoxon = 0.0, 1.0
        else:
            w_stat, p_wilcoxon = stats.wilcoxon(s1, s2)

        # 3. Cohen's d Effect Size (pooled standard deviation)
        m1, m2 = np.mean(s1), np.mean(s2)
        v1, v2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
        s_pooled = np.sqrt((v1 + v2) / 2.0)
        cohens_d = float((m1 - m2) / s_pooled) if s_pooled > 1e-8 else 0.0

        return {
            "t_statistic": float(t_stat),
            "p_value_ttest": float(p_ttest),
            "wilcoxon_statistic": float(w_stat),
            "p_value_wilcoxon": float(p_wilcoxon),
            "cohens_d": cohens_d,
        }
