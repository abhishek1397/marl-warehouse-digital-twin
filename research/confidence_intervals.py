"""ConfidenceIntervalCalculator computing parametric Student's t and non-parametric bootstrap CIs."""

from typing import List, Tuple, Union

import numpy as np
from scipy import stats


class ConfidenceIntervalCalculator:
    """Computes parametric and non-parametric 95% confidence intervals for evaluation metrics."""

    @staticmethod
    def compute_t_ci(
        data: Union[List[float], np.ndarray], confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """Calculates mean and parametric Student's t-distribution confidence interval bounds.

        Returns:
            Tuple of (mean, ci_lower, ci_upper).
        """
        arr = np.asarray(data, dtype=float)
        n = len(arr)
        if n < 2:
            val = float(arr[0]) if n == 1 else 0.0
            return val, val, val

        mean = float(np.mean(arr))
        sem = float(stats.sem(arr))
        if sem == 0.0:
            return mean, mean, mean

        h = sem * stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)
        return mean, float(mean - h), float(mean + h)

    @staticmethod
    def compute_bootstrap_ci(
        data: Union[List[float], np.ndarray],
        num_bootstraps: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """Calculates non-parametric 95% bootstrap percentile confidence interval bounds.

        Returns:
            Tuple of (mean, ci_lower, ci_upper).
        """
        arr = np.asarray(data, dtype=float)
        n = len(arr)
        if n < 2:
            val = float(arr[0]) if n == 1 else 0.0
            return val, val, val

        rng = np.random.default_rng(seed)
        boot_means = np.zeros(num_bootstraps)

        for i in range(num_bootstraps):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        alpha = 1.0 - confidence
        lower = float(np.percentile(boot_means, (alpha / 2.0) * 100.0))
        upper = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100.0))
        mean = float(np.mean(arr))

        return mean, lower, upper
