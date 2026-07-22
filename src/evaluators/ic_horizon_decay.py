"""Rank IC decay across forward-return holding horizons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


IC_HORIZON_DECAY_MIN_HORIZON = 20
IC_HORIZON_DECAY_MAX_HORIZON = 252
_RANK_CHUNK_ROWS = 64


def _positive_integer(value: int, label: str) -> int:
    normalized = int(value)
    if normalized < 1 or normalized != value:
        raise ValueError(f"{label} must be a positive integer; received {value!r}")
    return normalized


def _rowwise_spearman(
    factor_values: np.ndarray,
    return_values: np.ndarray,
) -> np.ndarray:
    """Vectorized row-wise Spearman correlation with pairwise finite samples."""

    valid = np.isfinite(factor_values) & np.isfinite(return_values)
    counts = valid.sum(axis=1)
    factor_rank = pd.DataFrame(np.where(valid, factor_values, np.nan)).rank(
        axis=1,
        method="average",
    ).to_numpy(dtype=float)
    return_rank = pd.DataFrame(np.where(valid, return_values, np.nan)).rank(
        axis=1,
        method="average",
    ).to_numpy(dtype=float)

    factor_means = np.divide(
        np.nansum(factor_rank, axis=1),
        counts,
        out=np.full(len(counts), np.nan, dtype=float),
        where=counts > 0,
    )
    return_means = np.divide(
        np.nansum(return_rank, axis=1),
        counts,
        out=np.full(len(counts), np.nan, dtype=float),
        where=counts > 0,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        factor_centered = factor_rank - factor_means[:, None]
        return_centered = return_rank - return_means[:, None]
        numerator = np.nansum(factor_centered * return_centered, axis=1)
        denominator = np.sqrt(
            np.nansum(factor_centered**2, axis=1)
            * np.nansum(return_centered**2, axis=1)
        )
        correlations = numerator / denominator
    correlations[(counts < 3) | ~(denominator > 0)] = np.nan
    return correlations


def _iso_day(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.date().isoformat()


def ic_horizon_decay(
    factor: pd.DataFrame,
    open_prices: pd.DataFrame,
    *,
    min_horizon: int = IC_HORIZON_DECAY_MIN_HORIZON,
    max_horizon: int = IC_HORIZON_DECAY_MAX_HORIZON,
    latest_allowed_open_day: object | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Average daily Rank IC on one common signal-date sample for every horizon.

    A factor observed after close on signal day ``t`` enters at ``open[t+1]``.
    For each holding horizon ``H``, its label is
    ``open[t+1+H] / open[t+1] - 1``.  Structurally eligible signal dates are
    selected using ``max_horizon`` first.  The final common sample is the
    intersection of dates with a finite cross-sectional Rank IC at *every*
    requested horizon, so all points on the curve average over exactly the
    same dates.
    """

    if not isinstance(factor, pd.DataFrame) or factor.empty:
        raise ValueError("factor must be a nonempty pandas DataFrame")
    if not isinstance(open_prices, pd.DataFrame) or open_prices.empty:
        raise ValueError("open_prices must be a nonempty pandas DataFrame")
    if factor.index.has_duplicates or factor.columns.has_duplicates:
        raise ValueError("factor axes must not contain duplicates")
    if open_prices.index.has_duplicates or open_prices.columns.has_duplicates:
        raise ValueError("open_prices axes must not contain duplicates")

    normalized_min = _positive_integer(min_horizon, "min_horizon")
    normalized_max = _positive_integer(max_horizon, "max_horizon")
    if normalized_min > normalized_max:
        raise ValueError("min_horizon must not exceed max_horizon")

    missing_columns = factor.columns.difference(open_prices.columns)
    if len(missing_columns):
        raise ValueError(
            "open_prices is missing factor securities: "
            f"{missing_columns.astype(str).tolist()[:5]}"
        )
    signal_positions = open_prices.index.get_indexer(factor.index)
    if (signal_positions < 0).any():
        raise ValueError("Every factor signal date must exist in open_prices")
    if len(signal_positions) > 1 and np.any(np.diff(signal_positions) <= 0):
        raise ValueError("factor signal dates must follow open_prices in ascending order")

    if latest_allowed_open_day is None:
        latest_exit_position = len(open_prices.index) - 1
    else:
        latest_exit_position = int(
            open_prices.index.get_indexer(pd.Index([latest_allowed_open_day]))[0]
        )
        if latest_exit_position < 0:
            raise ValueError("latest_allowed_open_day must exist in open_prices")

    structurally_valid = (
        signal_positions + 1 + normalized_max <= latest_exit_position
    )
    candidate_positions = signal_positions[structurally_valid]
    candidate_dates = factor.index[structurally_valid]
    if not len(candidate_positions):
        raise ValueError(
            "No signal date can support the maximum IC horizon; "
            f"H={normalized_max} needs open[t+1+H]"
        )

    aligned_factor = factor.loc[candidate_dates].to_numpy(dtype=float, copy=False)
    aligned_open = open_prices.reindex(columns=factor.columns).to_numpy(
        dtype=float,
        copy=False,
    )
    entry_values = aligned_open[candidate_positions + 1]
    horizons = np.arange(normalized_min, normalized_max + 1, dtype=int)
    daily_ic = np.full((len(candidate_positions), len(horizons)), np.nan, dtype=float)

    for horizon_column, horizon in enumerate(horizons):
        for start in range(0, len(candidate_positions), _RANK_CHUNK_ROWS):
            stop = min(start + _RANK_CHUNK_ROWS, len(candidate_positions))
            exit_values = aligned_open[
                candidate_positions[start:stop] + 1 + horizon
            ]
            with np.errstate(invalid="ignore", divide="ignore"):
                returns = exit_values / entry_values[start:stop] - 1.0
            returns[~np.isfinite(returns)] = np.nan
            daily_ic[start:stop, horizon_column] = _rowwise_spearman(
                aligned_factor[start:stop],
                returns,
            )

    common_mask = np.isfinite(daily_ic).all(axis=1)
    common_dates = candidate_dates[common_mask]
    if not len(common_dates):
        raise ValueError(
            "No common signal dates have a finite Rank IC at every requested horizon"
        )
    common_daily_ic = daily_ic[common_mask]
    mean_ic = common_daily_ic.mean(axis=0)
    detail = pd.DataFrame(
        {"mean_rank_ic": mean_ic},
        index=pd.Index(horizons, name="horizon"),
    )
    metrics: dict[str, object] = {
        "ic_horizon_decay_min_horizon": normalized_min,
        "ic_horizon_decay_max_horizon": normalized_max,
        "ic_horizon_decay_horizon_count": int(len(horizons)),
        "ic_horizon_decay_structural_signal_day_count": int(len(candidate_dates)),
        "ic_horizon_decay_common_signal_day_count": int(len(common_dates)),
        "ic_horizon_decay_dropped_signal_day_count": int(
            len(candidate_dates) - len(common_dates)
        ),
        "ic_horizon_decay_common_start_day": _iso_day(common_dates[0]),
        "ic_horizon_decay_common_end_day": _iso_day(common_dates[-1]),
        "ic_horizon_decay_h20_mean_ic": float(detail.loc[normalized_min, "mean_rank_ic"]),
        "ic_horizon_decay_h252_mean_ic": float(detail.loc[normalized_max, "mean_rank_ic"]),
    }
    return metrics, detail


def _latest_allowed_open_day(
    state: EvaluationState,
    open_prices: pd.DataFrame,
) -> object:
    """Infer the requested sample's final permissible open from context axes."""

    evaluation_dates = state.context.close.index
    if evaluation_dates.equals(open_prices.index):
        return open_prices.index[-1]
    positions = open_prices.index.get_indexer(evaluation_dates)
    if (positions < 0).any():
        raise ValueError("Evaluation dates must exist in the open-price matrix")
    inferred_requested_end = int(positions[-1]) + state.context.horizon + 1
    return open_prices.index[min(inferred_requested_end, len(open_prices.index) - 1)]


@evaluation_method(
    "ic_horizon_decay",
    required_data_symbols=("o",),
)
def evaluate_ic_horizon_decay(state: EvaluationState) -> None:
    """Persist the H=20..252 common-sample Rank IC decay curve."""

    market_data = state.context.market_data
    if market_data is None or "o" not in market_data:
        raise ValueError("ic_horizon_decay requires open-price market data 'o'")
    open_prices = market_data["o"]
    metrics, detail = ic_horizon_decay(
        state.factor,
        open_prices,
        min_horizon=IC_HORIZON_DECAY_MIN_HORIZON,
        max_horizon=IC_HORIZON_DECAY_MAX_HORIZON,
        latest_allowed_open_day=_latest_allowed_open_day(state, open_prices),
    )
    state.add_detail("ic_horizon_decay", detail)
    state.add_metrics(metrics)
