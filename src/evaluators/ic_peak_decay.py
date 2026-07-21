"""Overlapping 60-day rolling IC mean diagnostics.

The method name is kept for compatibility with the evaluation-module plan.
For now it only produces the 60-day rolling mean-IC series with a 10-observation
step (50 observations of overlap); peak and half-life semantics are intentionally
left out until their definition is settled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


DEFAULT_IC_PEAK_DECAY_WINDOW = 60
DEFAULT_IC_PEAK_DECAY_STEP = 10


def ic_peak_decay(
    ic: pd.Series,
    *,
    window: int = DEFAULT_IC_PEAK_DECAY_WINDOW,
    step: int = DEFAULT_IC_PEAK_DECAY_STEP,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate rolling mean IC using ``window`` observations and ``step``.

    ``window`` counts valid daily IC observations, rather than calendar days.
    Each window starts ``step`` observations after the previous one, so the
    default 60-day window overlaps the previous window by 50 observations.
    The incomplete tail is excluded and each output row is indexed by the
    final observation date in that window.
    """

    if not isinstance(ic, pd.Series):
        raise TypeError("IC peak decay requires a pandas Series")
    normalized_window = int(window)
    if normalized_window != window or normalized_window < 2:
        raise ValueError("window must be an integer no smaller than 2")
    normalized_step = int(step)
    if normalized_step != step or not 1 <= normalized_step <= normalized_window:
        raise ValueError("step must be an integer between one and window")

    valid = (
        ic.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
        .sort_index()
    )
    if len(valid) < normalized_window:
        raise ValueError(
            "IC peak decay requires at least one complete rolling "
            f"window of {normalized_window} valid IC observations"
        )

    means: list[float] = []
    end_days: list[object] = []
    starts = range(0, len(valid) - normalized_window + 1, normalized_step)
    for start in starts:
        block = valid.iloc[start : start + normalized_window]
        means.append(float(block.mean()))
        end_days.append(block.index[-1])

    window_count = len(means)

    detail = pd.DataFrame(
        {"mean_ic": means},
        index=pd.Index(end_days, name=valid.index.name or "day"),
    )
    metrics: dict[str, Any] = {
        "ic_peak_decay_window_days": normalized_window,
        "ic_peak_decay_step_days": normalized_step,
        "ic_peak_decay_overlap_days": normalized_window - normalized_step,
        "ic_peak_decay_valid_ic_days": int(len(valid)),
        "ic_peak_decay_full_window_count": window_count,
    }
    return metrics, detail


@evaluation_method("ic_peak_decay", requires=("rank_ic",))
def evaluate_ic_peak_decay(state: EvaluationState) -> None:
    """Persist the 60-day rolling mean-IC detail."""

    ic = state.require_detail("ic")
    if not isinstance(ic, pd.Series):
        raise TypeError("Rank IC detail must be a pandas Series")
    metrics, detail = ic_peak_decay(ic)
    state.add_detail("ic_peak_decay", detail)
    state.add_metrics(metrics)
