"""Shared daily quantile-membership logic for return and holdings evaluators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_membership(
    factor_row: np.ndarray,
    forward_return_row: np.ndarray,
    *,
    n_quantiles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return valid column positions, ascending ranks, and zero-based groups.

    A finite forward return is deliberately part of eligibility.  This is the
    same portfolio-membership rule used by ``quantile_net_returns`` and keeps
    a holdings audit exactly reconcilable with the reported group return.
    ``rank(method='first')`` makes ties deterministic in the existing column
    order.
    """

    if n_quantiles < 1 or int(n_quantiles) != n_quantiles:
        raise ValueError("n_quantiles must be a positive integer")
    valid = np.isfinite(factor_row) & np.isfinite(forward_return_row)
    valid_positions = np.flatnonzero(valid)
    count = len(valid_positions)
    if count < n_quantiles:
        return None
    ranks = pd.Series(factor_row[valid]).rank(method="first").to_numpy(dtype=float)
    groups = np.floor((ranks - 1) * int(n_quantiles) / count).astype(int)
    return valid_positions, ranks, groups
