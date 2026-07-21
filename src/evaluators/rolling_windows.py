"""Shared configuration and segmentation for risk evaluators."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

DEFAULT_WINDOWS = (20, 60, 252)


def complete_non_overlapping_windows(
    values: pd.DataFrame,
    *,
    window: int,
) -> Iterator[tuple[object, pd.DataFrame]]:
    """Yield full, non-overlapping windows labelled by their ending day.

    A 60-day input is partitioned as ``[t1...t60]``, ``[t61...t120]``, and
    so on.  The final incomplete slice is deliberately excluded so every
    reported value represents the same observation span.
    """

    if window < 2:
        raise ValueError("Risk-evaluation windows must be at least two")
    full_window_count = len(values) // window
    for window_number in range(full_window_count):
        start_position = window_number * window
        end_position = start_position + window
        yield values.index[end_position - 1], values.iloc[start_position:end_position]
