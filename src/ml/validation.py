from __future__ import annotations

import math
from typing import List

from scipy.stats import norm


def deflated_sharpe_ratio(
    sr: float,
    sr_benchmark: float = 0.0,
    t: int = 252,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_trials: int = 1,
) -> float:
    """
    Deflated Sharpe Ratio (López de Prado 2018).
    DSR = PSR(SR* | SR, T, skew, kurt) where SR* is expected max from n_trials.
    Values near 1.0 = statistically significant; < 0.95 = potentially overfit.
    """
    if t <= 1:
        return 0.0

    # Expected maximum SR from n_trials independent tests (Eq. 8 in the paper)
    euler_mascheroni = 0.5772156649
    sr_star = sr_benchmark
    if n_trials > 1:
        z = (1.0 - euler_mascheroni) * norm.ppf(1.0 - 1.0 / n_trials) + \
            euler_mascheroni * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        sr_star = sr_benchmark + math.sqrt(
            (1.0 - skew * sr_benchmark + (kurt - 1.0) / 4.0 * sr_benchmark ** 2) / (t - 1)
        ) * z

    # PSR: probability that SR > sr_star
    numerator = (sr - sr_star) * math.sqrt(t - 1)
    denominator = math.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    if denominator <= 0:
        return 0.0
    z_score = numerator / denominator
    return float(norm.cdf(z_score))


def min_track_record_length(
    sr: float,
    sr_benchmark: float = 0.0,
    alpha: float = 0.05,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Minimum number of observations for statistically significant Sharpe."""
    z = norm.ppf(1.0 - alpha)
    denom = sr - sr_benchmark
    if abs(denom) < 1e-10:
        return float("inf")
    t_min = 1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) * (z / denom) ** 2
    return float(t_min)


def probability_of_backtest_overfitting(oos_sharpes: List[float]) -> float:
    """
    Simplified PBO estimate from OOS Sharpes across walk-forward folds.
    PBO = fraction of folds with negative OOS Sharpe (proxy, not exact CPCV).
    Value < 0.3 → acceptable.
    """
    if not oos_sharpes:
        return 1.0
    n_negative = sum(1 for s in oos_sharpes if s < 0)
    return n_negative / len(oos_sharpes)
