"""Mask factor values whose next-open entry is likely untradeable.

The evaluation label enters at ``open[t+1]``. If day ``t+1`` opens at its
price-limit boundary, the stock cannot realistically be entered at that open:
limit-up names cannot be bought and limit-down names cannot be sold. ST names
are excluded entirely by policy.

This filter sets ``factor[t]`` to ``NaN`` wherever day ``t+1`` is a suspected
limit-open or ST sample, so downstream grouping and IC methods only see the
remaining tradable universe. Limitations:

- Limit-open detection uses the supplied ``limit`` matrix and adjusted
  ``open[t+1] / close[t] - 1`` gap with a tolerance. It is still a proxy when
  exact exchange limit-up/down prices are unavailable.
- Only the entry day is checked; exit-day (``open[t+1+horizon]``) liquidity
  is not modelled.
- Suspended stocks need no handling: the data contract keeps them ``NaN``,
  so their forward return is already ``NaN``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


def open_limit_entry_masks(
    open_prices: pd.DataFrame,
    close: pd.DataFrame,
    limit_ratio: pd.DataFrame,
    *,
    tolerance: float = 0.002,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return entry-day limit-up and limit-down masks aligned to ``close``."""

    open_prices = open_prices.reindex(index=close.index, columns=close.columns)
    limit_ratio = limit_ratio.reindex(index=close.index, columns=close.columns)
    open_values = open_prices.to_numpy(dtype=float, copy=False)
    previous_close_values = close.shift(1).to_numpy(dtype=float, copy=False)
    ratio_values = limit_ratio.to_numpy(dtype=float, copy=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        gap = open_values / previous_close_values - 1
    comparable = (
        np.isfinite(gap)
        & np.isfinite(ratio_values)
        & (ratio_values > 0)
    )
    limit_up = comparable & (gap >= ratio_values - tolerance)
    limit_down = comparable & (gap <= -ratio_values + tolerance)
    return (
        pd.DataFrame(limit_up, index=close.index, columns=close.columns),
        pd.DataFrame(limit_down, index=close.index, columns=close.columns),
    )


def mask_untradeable_entries(
    factor: pd.DataFrame,
    open_prices: pd.DataFrame,
    close: pd.DataFrame,
    limit_ratio: pd.DataFrame,
    st_status: pd.DataFrame,
    *,
    tolerance: float = 0.002,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Set factor[t] to NaN where day t+1 is limit-open or ST.

    Returns (masked_factor, daily_detail, summary_metrics). Limit-up,
    limit-down, and ST counts are diagnostic categories and may overlap.
    """

    # Keep the complete market timeline here.  A bounded train/test factor
    # window deliberately omits its final ``horizon + 1`` signal rows, but its
    # last retained signal still needs the following market-data row to inspect
    # the actual t+1 entry.  Reindexing market data to factor.index first would
    # silently skip that last entry check.
    close = close.reindex(columns=factor.columns)
    open_prices = open_prices.reindex(index=close.index, columns=factor.columns)
    limit_ratio = limit_ratio.reindex(index=close.index, columns=factor.columns)
    st_status = (
        st_status.reindex(index=close.index, columns=factor.columns)
        .fillna(False)
        .astype(bool)
    )

    limit_up, limit_down = open_limit_entry_masks(
        open_prices,
        close,
        limit_ratio,
        tolerance=tolerance,
    )
    limit_up_values = limit_up.to_numpy()
    limit_down_values = limit_down.to_numpy()
    st_values = st_status.to_numpy()
    factor_values = factor.to_numpy(dtype=float, copy=True)
    signal_positions = close.index.get_indexer(factor.index)
    if (signal_positions < 0).any():
        raise ValueError("Factor dates must exist on the complete close-price timeline")

    n_days = len(factor.index)
    rows: list[dict[str, Any]] = []
    total_masked = 0
    total_valid = 0
    masked_days = 0
    total_limit_up = 0
    total_limit_down = 0
    total_st = 0

    for t in range(n_days):
        valid_factor = np.isfinite(factor_values[t])
        n_valid = int(valid_factor.sum())
        entry = int(signal_positions[t]) + 1
        if entry >= len(close.index):
            rows.append(
                {
                    "day": factor.index[t],
                    "n_factor_valid": n_valid,
                    "n_masked": 0,
                    "masked_share": 0.0 if n_valid else np.nan,
                    "n_masked_limit_up": 0,
                    "n_masked_limit_down": 0,
                    "n_masked_st": 0,
                }
            )
            total_valid += n_valid
            continue

        limit_up_masked = valid_factor & limit_up_values[entry]
        limit_down_masked = valid_factor & limit_down_values[entry]
        st_masked = valid_factor & st_values[entry]
        masked = limit_up_masked | limit_down_masked | st_masked
        n_masked = int(masked.sum())
        n_limit_up = int(limit_up_masked.sum())
        n_limit_down = int(limit_down_masked.sum())
        n_st = int(st_masked.sum())
        if n_masked:
            factor_values[t][masked] = np.nan
            masked_days += 1
        total_masked += n_masked
        total_valid += n_valid
        total_limit_up += n_limit_up
        total_limit_down += n_limit_down
        total_st += n_st
        rows.append(
            {
                "day": factor.index[t],
                "n_factor_valid": n_valid,
                "n_masked": n_masked,
                "masked_share": (n_masked / n_valid) if n_valid else np.nan,
                "n_masked_limit_up": n_limit_up,
                "n_masked_limit_down": n_limit_down,
                "n_masked_st": n_st,
            }
        )

    masked_factor = pd.DataFrame(
        factor_values,
        index=factor.index,
        columns=factor.columns,
    )
    detail = pd.DataFrame(rows).set_index("day")
    summary = {
        "tradability_masked_obs": int(total_masked),
        "tradability_masked_share": (
            total_masked / total_valid if total_valid else np.nan
        ),
        "tradability_masked_days": int(masked_days),
        "tradability_masked_limit_up_obs": int(total_limit_up),
        "tradability_masked_limit_down_obs": int(total_limit_down),
        "tradability_masked_st_obs": int(total_st),
    }
    return masked_factor, detail, summary


@evaluation_method(
    "tradability_filter",
    required_data_symbols=("o", "limit", "st"),
)
def evaluate_tradability_filter(state: EvaluationState) -> None:
    """Replace the working factor with its entry-tradability-masked version.

    Select this method *before* ``quantile_returns`` (and after ``rank_ic``
    if the IC should stay full-universe) — like ``market_cap_neutralize``,
    it changes the working factor for every later method.
    """

    data = state.context.market_data
    if data is None or not {"c", "o", "limit", "st"} <= set(data):
        raise ValueError(
            "Tradability filter requires the c/o/limit/st matrices in market_data"
        )
    masked_factor, detail, summary = mask_untradeable_entries(
        state.factor,
        data["o"],
        # ``context.close`` is deliberately restricted to the signal window.
        # The entry check needs the full close timeline so that the final
        # retained signal can still inspect its t+1 open/ST status.
        data["c"],
        data["limit"],
        data["st"],
    )
    state.replace_factor(masked_factor)
    state.add_detail("tradability_filter", detail)
    state.add_metrics(summary)
