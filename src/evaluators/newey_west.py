"""Newey-West significance test for the mean daily Rank IC."""

from __future__ import annotations

from math import erfc, sqrt
from typing import Any

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


def autocovariance(values: np.ndarray, lag: int) -> float:
    centered = values - values.mean()
    if lag == 0:
        return float(np.dot(centered, centered) / len(centered))
    return float(np.dot(centered[lag:], centered[:-lag]) / len(centered))


def select_newey_west_lag(
    autocorrelations: list[float] | np.ndarray,
    *,
    significance_bound: float,
    consecutive: int = 3,
    min_lag: int = 0,
) -> int:
    """Choose the last lag before consecutive insignificant autocorrelations.

    ``autocorrelations`` must start at lag zero. Lag zero is excluded from the
    search because its autocorrelation is one by definition. A lag is treated
    as insignificant when its absolute autocorrelation is within
    ``significance_bound``.
    """

    correlations = np.asarray(autocorrelations, dtype=float)
    if correlations.ndim != 1 or correlations.size < 2:
        raise ValueError("autocorrelations must contain lag zero and at least lag one")
    if (
        not np.isfinite(significance_bound)
        or significance_bound < 0
        or consecutive < 1
        or min_lag < 0
    ):
        raise ValueError(
            "significance_bound and min_lag must be nonnegative, and "
            "consecutive must be positive"
        )
    if min_lag >= correlations.size:
        raise ValueError("min_lag must be smaller than the autocorrelation count")

    run_length = 0
    for lag, correlation in enumerate(correlations[1:], start=1):
        within_band = (
            np.isfinite(correlation)
            and abs(float(correlation)) <= significance_bound
        )
        run_length = run_length + 1 if within_band else 0
        if run_length == consecutive:
            first_insignificant_lag = lag - consecutive + 1
            data_driven_lag = first_insignificant_lag - 1
            return max(min_lag, data_driven_lag)
    return max(min_lag, correlations.size - 1)


def newey_west_mean_test(
    series: pd.Series,
    *,
    consecutive: int = 3,
    horizon: int = 1,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Test whether a serially correlated series mean differs from zero."""

    values = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(
        dtype=float
    )
    if len(values) < max(3, consecutive):
        raise ValueError("Newey-West mean test requires at least three values")
    normalized_horizon = int(horizon)
    if normalized_horizon < 1 or normalized_horizon != horizon:
        raise ValueError("horizon must be a positive integer")
    min_lag = normalized_horizon - 1
    if min_lag >= len(values):
        raise ValueError(
            "Newey-West mean test requires more IC observations than horizon - 1"
        )

    gammas = np.asarray(
        [autocovariance(values, lag) for lag in range(len(values))],
        dtype=float,
    )
    if not np.isfinite(gammas[0]) or gammas[0] <= 0:
        raise ValueError("Newey-West mean test requires a nonconstant IC series")
    autocorrelations = gammas / gammas[0]
    significance_bound = 2.0 / sqrt(len(values))
    selected_lag = select_newey_west_lag(
        autocorrelations,
        significance_bound=significance_bound,
        consecutive=consecutive,
        min_lag=min_lag,
    )

    long_run_variance = gammas[0]
    if selected_lag > 0:
        for lag in range(1, selected_lag + 1):
            weight = 1.0 - lag / (selected_lag + 1.0)
            long_run_variance += 2.0 * weight * gammas[lag]
    standard_error = (
        sqrt(max(long_run_variance, 0.0) / len(values))
        if np.isfinite(long_run_variance)
        else np.nan
    )
    mean = float(values.mean())
    if standard_error > 0:
        t_stat = mean / standard_error
        p_value = erfc(abs(t_stat) / sqrt(2.0))
    elif mean == 0:
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = np.copysign(np.inf, mean), 0.0

    within_significance_band = (
        np.abs(autocorrelations) <= significance_bound
    )
    within_significance_band[0] = False
    detail = pd.DataFrame(
        {
            "lag": np.arange(len(gammas), dtype=int),
            "autocovariance": gammas,
            "abs_autocovariance": np.abs(gammas),
            "autocorrelation": autocorrelations,
            "abs_autocorrelation": np.abs(autocorrelations),
            "autocorrelation_bound": significance_bound,
            "within_significance_band": within_significance_band,
        }
    ).set_index("lag")
    detail["selected_lag"] = detail.index == selected_lag
    metrics = {
        "nw_ic_mean": mean,
        "nw_ic_lag": selected_lag,
        "nw_ic_long_run_variance": long_run_variance,
        "nw_ic_standard_error": standard_error,
        "nw_ic_t_stat": t_stat,
        "nw_ic_p_value": p_value,
        "nw_ic_significant_5pct": p_value < 0.05,
        "nw_ic_autocorr_bound": significance_bound,
        "nw_ic_lag_min": min_lag,
    }
    return metrics, detail


@evaluation_method("newey_west_ic_significance", requires=("rank_ic",))
def evaluate_newey_west_ic_significance(state: EvaluationState) -> None:
    detail = state.require_detail("ic")
    if not isinstance(detail, pd.Series):
        raise TypeError("Rank IC detail must be a pandas Series")
    metrics, autocovariances = newey_west_mean_test(
        detail,
        horizon=state.context.horizon,
    )
    state.add_detail("newey_west_ic_autocovariances", autocovariances)
    state.add_metrics(metrics)
