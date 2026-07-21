"""Compare Kalman, second-order, and Fourier filters on daily Rank IC."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


# A local-level model with Q / R = 0.01 has a steady-state Kalman gain of
# about 9.5%.  It visibly suppresses daily IC noise without making a regime
# change disappear for an entire quarter.  This is deliberately a stable,
# inspectable first setting rather than an in-sample parameter fit.
DEFAULT_PROCESS_TO_OBSERVATION_RATIO = 0.01
DEFAULT_SECOND_ORDER_PERIOD = 31
DEFAULT_FOURIER_MIN_PERIOD = 31
DEFAULT_FILTERED_IC_MEAN_WINDOW = 10
DEFAULT_FILTERED_IC_MEAN_STEP = 5

FILTERED_IC_COLUMNS = (
    "filtered_ic",
    "second_order_low_pass_ic",
    "fourier_low_pass_ic",
)
FILTERED_IC_MEAN_COLUMNS = (
    "kalman_filtered_ic_mean",
    "second_order_low_pass_ic_mean",
    "fourier_low_pass_ic_mean",
)


def _positive_finite(value: float, name: str) -> float:
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return normalized


def _positive_int(value: int, name: str, *, minimum: int = 1) -> int:
    normalized = int(value)
    if normalized != value or normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return normalized


def second_order_low_pass_ic(
    values: np.ndarray,
    *,
    period: int = DEFAULT_SECOND_ORDER_PERIOD,
) -> np.ndarray:
    """Apply the requested causal second-order low-pass recursion.

    The recurrence starts at position three because it needs ``M(t-3)``.
    The first three filtered states are initialized to their corresponding
    observations; this preserves a constant input and avoids an arbitrary
    zero-state transient at the start of the sample.
    """

    normalized_period = _positive_int(period, "period", minimum=3)
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1:
        raise ValueError("Second-order IC filter requires a one-dimensional array")
    if len(observations) < 4:
        raise ValueError("Second-order IC filter requires at least four observations")
    if not np.isfinite(observations).all():
        raise ValueError("Second-order IC filter requires finite observations")

    alpha = 2.0 / normalized_period
    filtered = np.empty(len(observations), dtype=float)
    filtered[:3] = observations[:3]
    for position in range(3, len(observations)):
        filtered[position] = (
            2.0 * (1.0 - alpha) * filtered[position - 1]
            - (1.0 - alpha) ** 2 * filtered[position - 2]
            + (alpha - 0.25 * alpha**2) * observations[position - 1]
            + 0.5 * alpha**2 * observations[position - 2]
            - (alpha - 0.75 * alpha**2) * observations[position - 3]
        )
    return filtered


def fourier_low_pass_ic(
    values: np.ndarray,
    *,
    min_period: int = DEFAULT_FOURIER_MIN_PERIOD,
) -> tuple[np.ndarray, int]:
    """Reconstruct IC using only Fourier components with long periods.

    Frequencies above ``1 / min_period`` cycles per valid IC observation are
    removed.  This is a full-sample, non-causal diagnostic: interior values
    can change when later IC observations are appended.
    """

    normalized_period = _positive_int(min_period, "min_period", minimum=2)
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1:
        raise ValueError("Fourier IC filter requires a one-dimensional array")
    if len(observations) < 2:
        raise ValueError("Fourier IC filter requires at least two observations")
    if not np.isfinite(observations).all():
        raise ValueError("Fourier IC filter requires finite observations")

    spectrum = np.fft.rfft(observations)
    frequencies = np.fft.rfftfreq(len(observations), d=1.0)
    retained = frequencies <= (1.0 / normalized_period)
    spectrum[~retained] = 0.0
    reconstructed = np.fft.irfft(spectrum, n=len(observations))
    return reconstructed.astype(float, copy=False), int(retained.sum())


def _kalman_local_level(
    values: np.ndarray,
    *,
    process_to_observation_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal local-level Kalman states and gains."""

    observation_variance = 1.0
    process_variance = process_to_observation_ratio
    posterior_variance = (
        sqrt(process_variance**2 + 4.0 * process_variance * observation_variance)
        - process_variance
    ) / 2.0

    filtered = np.empty(len(values), dtype=float)
    gains = np.empty(len(values), dtype=float)
    filtered[0] = values[0]
    gains[0] = np.nan

    for position in range(1, len(values)):
        prior_variance = posterior_variance + process_variance
        gain = prior_variance / (prior_variance + observation_variance)
        filtered[position] = filtered[position - 1] + gain * (
            values[position] - filtered[position - 1]
        )
        posterior_variance = (1.0 - gain) * prior_variance
        gains[position] = gain
    return filtered, gains


def compare_ic_trend_filters(
    ic: pd.Series,
    *,
    process_to_observation_ratio: float = DEFAULT_PROCESS_TO_OBSERVATION_RATIO,
    second_order_period: int = DEFAULT_SECOND_ORDER_PERIOD,
    fourier_min_period: int = DEFAULT_FOURIER_MIN_PERIOD,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare three trend estimates on the latest valid daily IC series."""

    if not isinstance(ic, pd.Series):
        raise TypeError("IC trend filter requires a pandas Series")
    ratio = _positive_finite(
        process_to_observation_ratio,
        "process_to_observation_ratio",
    )
    normalized_second_order_period = _positive_int(
        second_order_period,
        "second_order_period",
        minimum=3,
    )
    normalized_fourier_min_period = _positive_int(
        fourier_min_period,
        "fourier_min_period",
        minimum=2,
    )
    valid = (
        ic.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
        .sort_index()
    )
    if len(valid) < 4:
        raise ValueError("IC trend filter comparison requires at least four finite daily IC values")

    values = valid.to_numpy(dtype=float, copy=False)
    kalman_filtered, gains = _kalman_local_level(
        values,
        process_to_observation_ratio=ratio,
    )
    second_order_filtered = second_order_low_pass_ic(
        values,
        period=normalized_second_order_period,
    )
    fourier_filtered, retained_frequency_count = fourier_low_pass_ic(
        values,
        min_period=normalized_fourier_min_period,
    )

    detail = pd.DataFrame(
        {
            "daily_ic": values,
            "filtered_ic": kalman_filtered,
            "second_order_low_pass_ic": second_order_filtered,
            "fourier_low_pass_ic": fourier_filtered,
        },
        index=pd.Index(valid.index, name=valid.index.name or "day"),
    )
    metrics: dict[str, Any] = {
        "ic_trend_filter_valid_ic_days": int(len(valid)),
        "ic_trend_filter_process_to_observation_ratio": ratio,
        "ic_trend_filter_kalman_gain": float(np.nanmean(gains)),
        # Keep the original keys for persisted-result compatibility; these
        # three values continue to refer specifically to the Kalman curve.
        "ic_trend_filter_start": float(kalman_filtered[0]),
        "ic_trend_filter_final": float(kalman_filtered[-1]),
        "ic_trend_filter_change": float(kalman_filtered[-1] - kalman_filtered[0]),
        "ic_trend_filter_second_order_period": normalized_second_order_period,
        "ic_trend_filter_second_order_start": float(second_order_filtered[0]),
        "ic_trend_filter_second_order_final": float(second_order_filtered[-1]),
        "ic_trend_filter_second_order_change": float(
            second_order_filtered[-1] - second_order_filtered[0]
        ),
        "ic_trend_filter_fourier_min_period": normalized_fourier_min_period,
        "ic_trend_filter_fourier_retained_frequency_count": retained_frequency_count,
        "ic_trend_filter_fourier_start": float(fourier_filtered[0]),
        "ic_trend_filter_fourier_final": float(fourier_filtered[-1]),
        "ic_trend_filter_fourier_change": float(
            fourier_filtered[-1] - fourier_filtered[0]
        ),
    }
    return metrics, detail


def overlapping_filtered_ic_mean(
    filtered_detail: pd.DataFrame,
    *,
    window: int = DEFAULT_FILTERED_IC_MEAN_WINDOW,
    step: int = DEFAULT_FILTERED_IC_MEAN_STEP,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Average each filtered IC curve in overlapping observation windows.

    With the defaults, the first window is observations 1--10 and the next
    is observations 6--15.  ``window`` and ``step`` count valid filtered IC
    observations, not calendar days.  Incomplete trailing windows are
    excluded and each row is indexed by its window-end date.
    """

    if not isinstance(filtered_detail, pd.DataFrame):
        raise TypeError("Filtered IC mean requires a pandas DataFrame")
    normalized_window = _positive_int(window, "window", minimum=2)
    normalized_step = _positive_int(step, "step", minimum=1)
    if normalized_step > normalized_window:
        raise ValueError("step must be no greater than window")

    missing_columns = [
        column for column in FILTERED_IC_COLUMNS if column not in filtered_detail
    ]
    if missing_columns:
        raise ValueError(
            "Filtered IC mean is missing required columns: "
            + ", ".join(missing_columns)
        )

    valid = (
        filtered_detail.loc[:, list(FILTERED_IC_COLUMNS)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .astype(float)
        .sort_index()
    )
    if len(valid) < normalized_window:
        raise ValueError(
            "Filtered IC mean requires at least one complete window of "
            f"{normalized_window} valid observations"
        )

    means: list[np.ndarray] = []
    end_days: list[object] = []
    for start in range(0, len(valid) - normalized_window + 1, normalized_step):
        block = valid.iloc[start : start + normalized_window]
        means.append(block.mean(axis=0).to_numpy(dtype=float))
        end_days.append(block.index[-1])

    detail = pd.DataFrame(
        means,
        columns=FILTERED_IC_MEAN_COLUMNS,
        index=pd.Index(end_days, name=valid.index.name or "day"),
    )
    metrics: dict[str, Any] = {
        "ic_trend_filter_mean_window_days": normalized_window,
        "ic_trend_filter_mean_step_days": normalized_step,
        "ic_trend_filter_mean_overlap_days": normalized_window - normalized_step,
        "ic_trend_filter_mean_full_window_count": int(len(detail)),
    }
    return metrics, detail


def kalman_ic_trend(
    ic: pd.Series,
    *,
    process_to_observation_ratio: float = DEFAULT_PROCESS_TO_OBSERVATION_RATIO,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Backward-compatible entry point for the IC filter comparison.

    Existing callers used this name when the module only produced a Kalman
    curve.  It now returns the original Kalman fields plus the second-order and
    Fourier comparison curves.
    """
    return compare_ic_trend_filters(
        ic,
        process_to_observation_ratio=process_to_observation_ratio,
    )


@evaluation_method("ic_trend_filter", requires=("rank_ic",))
def evaluate_ic_trend_filter(state: EvaluationState) -> None:
    """Persist three IC trends and their overlapping 10-observation means."""

    ic = state.require_detail("ic")
    if not isinstance(ic, pd.Series):
        raise TypeError("Rank IC detail must be a pandas Series")
    metrics, detail = compare_ic_trend_filters(ic)
    mean_metrics, mean_detail = overlapping_filtered_ic_mean(detail)
    state.add_detail("ic_trend_filter", detail)
    state.add_detail("ic_trend_filter_mean_10", mean_detail)
    state.add_metrics(metrics)
    state.add_metrics(mean_metrics)
