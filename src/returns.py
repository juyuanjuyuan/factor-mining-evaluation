"""Canonical return labels used by every factor evaluation method."""

from __future__ import annotations

import numpy as np
import pandas as pd


RETURN_DEFINITION = "open[t+1+horizon]/open[t+1]-1"


def calculate_forward_open_return(
    open_prices: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """Return the label for a close-of-day factor traded at the next open.

    A factor observed after day ``t`` closes enters at ``open[t + 1]`` and,
    for horizon ``H``, exits at ``open[t + 1 + H]``. Therefore ``H=1`` is
    exactly ``open[t + 2] / open[t + 1] - 1``.
    """

    if not isinstance(open_prices, pd.DataFrame) or open_prices.empty:
        raise ValueError("open_prices must be a nonempty pandas DataFrame")
    normalized_horizon = int(horizon)
    if normalized_horizon < 1 or normalized_horizon != horizon:
        raise ValueError(
            f"horizon must be a positive integer; received {horizon!r}"
        )
    entry_open = open_prices.shift(-1)
    exit_open = open_prices.shift(-(normalized_horizon + 1))
    return (exit_open / entry_open - 1).replace(
        [np.inf, -np.inf],
        np.nan,
    )
