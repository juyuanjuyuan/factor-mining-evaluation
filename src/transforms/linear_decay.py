"""Causal linear smoothing for a completed wide factor score."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# A window block has one float per (day, security, lookback) cell.  Keep the
# source block bounded so a large A-share matrix does not require one enormous
# three-dimensional allocation.
_BLOCK_BYTES = 64 * 1024 * 1024


def _column_chunk(row_count: int, window: int) -> int:
    per_column_bytes = max(1, row_count * window * 8)
    return int(min(4096, max(1, _BLOCK_BYTES // per_column_bytes)))


def validate_linear_decay_window(value: Any) -> int:
    """Normalize the public run-level Decay setting to a whole trading day."""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"decay must be a positive integer; received {value!r}")
    window = int(value)
    if window < 1:
        raise ValueError(f"decay must be a positive integer; received {value!r}")
    return window


def apply_linear_decay(factor: pd.DataFrame, decay: int = 1) -> pd.DataFrame:
    """Apply the configured run-level linear-decay preprocessor.

    A decay of one is deliberately an identity operation, preserving the
    factor scores produced by evaluations before this configurable
    preprocessing stage existed.
    """

    window = validate_linear_decay_window(decay)
    if window == 1 or factor.empty:
        return factor
    return _linearly_decay_factor(factor, window)


def linearly_decay_factor(factor: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return a causal linearly weighted score using every available prefix.

    This public numerical helper validates ``window`` independently. Use
    :func:`apply_linear_decay` at the run boundary so the identity default and
    validation policy stay centralized.
    """

    normalized_window = validate_linear_decay_window(window)
    if normalized_window == 1 or factor.empty:
        return factor
    return _linearly_decay_factor(factor, normalized_window)


def _linearly_decay_factor(
    factor: pd.DataFrame,
    normalized_window: int,
) -> pd.DataFrame:
    """Return a causal linearly weighted score using every available prefix.

    The latest score has weight ``window`` and the oldest has weight ``1``.
    During the first ``window - 1`` days, the function renormalizes the
    available suffix rather than dropping the row. Missing score cells do not
    contribute either a value or its weight.
    """

    raw = factor.to_numpy(dtype=float, copy=False)
    values = np.where(np.isfinite(raw), raw, np.nan)
    result = np.empty(values.shape, dtype=float)
    weights = np.arange(1, normalized_window + 1, dtype=float)
    step = _column_chunk(len(values), normalized_window)
    for start in range(0, values.shape[1], step):
        stop = min(values.shape[1], start + step)
        chunk = values[:, start:stop]
        padded = np.full(
            (len(values) + normalized_window - 1, chunk.shape[1]),
            np.nan,
            dtype=float,
        )
        padded[normalized_window - 1 :] = chunk
        block = sliding_window_view(padded, normalized_window, axis=0)
        valid = np.isfinite(block)
        weighted_values = np.where(valid, block * weights, 0.0)
        weight_sum = np.where(valid, weights, 0.0).sum(axis=2)
        numerator = weighted_values.sum(axis=2)
        result[:, start:stop] = np.divide(
            numerator,
            weight_sum,
            out=np.full_like(numerator, np.nan),
            where=weight_sum > 0,
        )
    return pd.DataFrame(result, index=factor.index, columns=factor.columns)
