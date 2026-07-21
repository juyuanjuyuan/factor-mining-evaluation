"""Optional chart-only A-share market-cycle background."""

from __future__ import annotations

import pandas as pd

from market_cycles import cycle_backgrounds_for_dates

from .base import EvaluationState, evaluation_method, latest_versioned_key
from .quantile_plot import _plot_cumulative_returns


@evaluation_method("cycle_context", requires=("quantile_plot",))
def apply_cycle_context(state: EvaluationState) -> None:
    """Redraw the existing cumulative-return PNG with market-cycle bands.

    This is deliberately presentation-only: it does not alter the factor,
    return label, metrics, details, or the artifact mapping.  The latest plot
    is overwritten in place so downstream consumers keep the normal ``plot``
    artifact contract.
    """

    cumulative = state.require_detail("cumulative_returns")
    if not isinstance(cumulative, pd.DataFrame):
        raise TypeError("Cumulative-return detail must be a pandas DataFrame")
    artifact_key = latest_versioned_key("plot", state.artifacts)
    if artifact_key is None:
        raise ValueError("cycle_context requires an existing cumulative-return plot")
    _plot_cumulative_returns(
        cumulative,
        f"{state.context.factor_name} quantile cumulative returns",
        state.artifacts[artifact_key],
        cycle_backgrounds=cycle_backgrounds_for_dates(cumulative.index),
    )
