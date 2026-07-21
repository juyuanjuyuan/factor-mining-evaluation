"""Daily cross-sectional Rank IC and IC/IR summary methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman correlation for two finite one-dimensional arrays."""

    if len(x) < 3:
        return np.nan
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    if np.ptp(x_rank) == 0 or np.ptp(y_rank) == 0:
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


@evaluation_method("rank_ic")
def evaluate_rank_ic(state: EvaluationState) -> None:
    """Calculate the daily Rank IC series independently of other methods."""

    factor_values = state.factor.to_numpy(dtype=float, copy=False)
    return_values = state.context.forward_return.to_numpy(dtype=float, copy=False)
    dates: list[object] = []
    values: list[float] = []
    pair_count = 0

    for row_number, day in enumerate(state.context.close.index):
        factor_row = factor_values[row_number]
        return_row = return_values[row_number]
        valid = np.isfinite(factor_row) & np.isfinite(return_row)
        count = int(valid.sum())
        if count < 3:
            continue
        pair_count += count
        ic = spearman_rank_correlation(factor_row[valid], return_row[valid])
        if np.isfinite(ic):
            dates.append(day)
            values.append(ic)

    if not values:
        raise ValueError("No finite daily Rank IC values were produced")

    ic_series = pd.Series(
        values,
        index=pd.Index(dates, name="day"),
        name="ic",
        dtype=float,
    )
    state.add_detail("ic", ic_series)
    state.add_metrics(
        {
            "ic_count": int(ic_series.size),
            "pair_count": pair_count,
            "start_day": str(ic_series.index.min()),
            "end_day": str(ic_series.index.max()),
        }
    )


@evaluation_method("rank_icir", requires=("rank_ic",))
def evaluate_rank_icir(state: EvaluationState) -> None:
    """Summarize a previously calculated Rank IC series."""

    detail = state.require_detail("ic")
    if not isinstance(detail, pd.Series):
        raise TypeError("Rank IC detail must be a pandas Series")
    ic_series = detail.dropna()
    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else np.nan
    signed_ir = (
        ic_mean / ic_std
        if np.isfinite(ic_mean) and np.isfinite(ic_std) and ic_std > 0
        else np.nan
    )
    state.add_metrics(
        {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ir": signed_ir,
            "ic_positive_ratio": float((ic_series > 0).mean()),
        }
    )


def calc_ic(
    frame: pd.DataFrame,
    factor_name: str = "score",
    label_name: str = "f1",
) -> pd.Series:
    """Compatibility helper for long-form notebook data."""

    required = {"day", factor_name, label_name}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing IC columns: {sorted(missing)}")

    def daily_ic(group: pd.DataFrame) -> float:
        valid = group[[factor_name, label_name]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(valid) < 3:
            return np.nan
        return spearman_rank_correlation(
            valid[factor_name].to_numpy(dtype=float),
            valid[label_name].to_numpy(dtype=float),
        )

    return frame.groupby("day", sort=True, observed=True)[
        [factor_name, label_name]
    ].apply(daily_ic)


def get_ic_info(
    frame: pd.DataFrame,
    factor_name: str,
    label_name: str,
) -> tuple[float, float, float]:
    """Compatibility helper returning IC mean, std, and absolute IR."""

    ic_series = calc_ic(frame, factor_name, label_name).dropna()
    ic_mean = float(ic_series.mean()) if len(ic_series) else np.nan
    ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else np.nan
    signed_ir = (
        ic_mean / ic_std
        if np.isfinite(ic_mean) and np.isfinite(ic_std) and ic_std > 0
        else np.nan
    )
    return ic_mean, ic_std, abs(signed_ir) if np.isfinite(signed_ir) else np.nan
