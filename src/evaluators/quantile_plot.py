"""Q1..QN cumulative-return plotting as an independent evaluation method."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from market_cycles import MarketCycle

from .base import EvaluationState, evaluation_method, versioned_key


def _plot_cumulative_returns(
    cumulative: pd.DataFrame,
    title: str,
    output_path: str | Path | None,
    *,
    cycle_backgrounds: Sequence[MarketCycle] = (),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(14, 7))
    group_columns = list(cumulative.columns)
    if not group_columns:
        raise ValueError("Cumulative returns must contain quantile columns")
    for cycle in cycle_backgrounds:
        axis.axvspan(
            pd.Timestamp(cycle.start_day),
            pd.Timestamp(cycle.end_day),
            color="#d94841" if cycle.direction == "up" else "#3a8f63",
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(group_columns)))[::-1]
    for index, column in enumerate(group_columns):
        axis.plot(
            cumulative.index,
            cumulative[column],
            label=column,
            color=colors[index],
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )
    axis.axhline(0, color="gray", linestyle="--", alpha=0.5, zorder=3)
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative return")
    axis.set_title(title)
    axis.legend(loc="upper left", bbox_to_anchor=(1, 1))
    axis.grid(True, alpha=0.3, zorder=1)
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


@evaluation_method(
    "quantile_plot",
    requires=("quantile_cumulative",),
)
def plot_quantile_cumulative(state: EvaluationState) -> None:
    """Write the Q1..QN cumulative-return chart for the current factor."""

    detail = state.require_detail("cumulative_returns")
    if not isinstance(detail, pd.DataFrame):
        raise TypeError("Cumulative-return detail must be a pandas DataFrame")
    # Repeated executions must not overwrite the earlier chart file: mirror the
    # versioned artifact key (plot, plot__2, ...) in the PNG filename.
    version = 1
    while versioned_key("plot", version) in state.artifacts:
        version += 1
    file_suffix = "" if version == 1 else f"__{version}"
    output_path = (
        state.context.output_dir
        / "plots"
        / f"{state.context.artifact_name}{file_suffix}.png"
    )
    _plot_cumulative_returns(
        detail,
        f"{state.context.factor_name} quantile cumulative returns",
        output_path,
    )
    state.add_artifact("plot", output_path)


def plot_group_cumulative(
    group_return_pivot: pd.DataFrame,
    title: str,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compatibility helper that compounds and plots group returns."""

    cumulative = (1 + group_return_pivot.fillna(0)).cumprod() - 1
    _plot_cumulative_returns(cumulative, title, output_path)
    return cumulative
