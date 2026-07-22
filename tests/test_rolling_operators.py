#!/usr/bin/env python3
"""Equivalence test for the vectorized rolling operators in ``engine``.

``ts_product``, ``ts_argmax``, ``ts_argmin``, ``decay_linear``, and
``ts_linear_reg_slope`` were rewritten from a per-window Python callback
(``rolling().apply``) to strided NumPy blocks. The callback forms are kept
here as the reference definition: every vectorized result must reproduce
them element for element, including the ramp-up rows where pandas passes a
shorter window, missing values, infinities, and ties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from engine import (
    decay_linear,
    roll,
    ts_argmax,
    ts_argmin,
    ts_linear_reg_slope,
    ts_product,
)


def reference_arg_extreme(values: np.ndarray, *, find_maximum: bool) -> float:
    if len(values) == 0 or not np.isfinite(values).any():
        return np.nan
    position = np.nanargmax(values) if find_maximum else np.nanargmin(values)
    return float(position + 1)


def reference_ts_product(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll(x, window).apply(np.prod, raw=True)


def reference_ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll(x, window).apply(
        lambda values: reference_arg_extreme(values, find_maximum=True), raw=True
    )


def reference_ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll(x, window).apply(
        lambda values: reference_arg_extreme(values, find_maximum=False), raw=True
    )


def reference_decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    def weighted(values: np.ndarray) -> float:
        valid = np.isfinite(values)
        if not valid.any():
            return np.nan
        weights = np.arange(1, len(values) + 1, dtype=float)
        weights = weights[valid]
        return float(np.dot(values[valid], weights) / weights.sum())

    return roll(x, window).apply(weighted, raw=True)


def reference_ts_linear_reg_slope(x: pd.DataFrame, window: int) -> pd.DataFrame:
    def slope(values: np.ndarray) -> float:
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return np.nan
        time = np.arange(1, len(values) + 1, dtype=float)[valid]
        observed = values[valid]
        centered_time = time - time.mean()
        denominator = np.dot(centered_time, centered_time)
        if denominator == 0:
            return np.nan
        return float(np.dot(centered_time, observed - observed.mean()) / denominator)

    return roll(x, window).apply(slope, raw=True)


OPERATOR_PAIRS = (
    ("ts_product", ts_product, reference_ts_product),
    ("ts_argmax", ts_argmax, reference_ts_argmax),
    ("ts_argmin", ts_argmin, reference_ts_argmin),
    ("decay_linear", decay_linear, reference_decay_linear),
    ("ts_linear_reg_slope", ts_linear_reg_slope, reference_ts_linear_reg_slope),
)


def build_adversarial_frame() -> pd.DataFrame:
    """Columns that exercise every branch of the rolling kernels."""

    rows = 40
    index = pd.date_range("2024-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(20260722)
    columns: dict[str, np.ndarray] = {
        "plain": rng.normal(loc=1.0, scale=0.3, size=rows),
        "positive": rng.uniform(0.5, 1.5, size=rows),
    }

    leading_nan = rng.normal(size=rows)
    leading_nan[:12] = np.nan
    columns["leading_nan"] = leading_nan

    interior_nan = rng.normal(size=rows)
    interior_nan[[3, 4, 5, 17, 28, 29, 30, 31]] = np.nan
    columns["interior_nan"] = interior_nan

    columns["all_nan"] = np.full(rows, np.nan)

    sparse = np.full(rows, np.nan)
    # A single observation per window keeps min_periods and the n < 2 slope
    # guard live.
    sparse[::7] = rng.normal(size=len(sparse[::7]))
    columns["sparse"] = sparse

    with_infinity = rng.normal(size=rows)
    with_infinity[[6, 19]] = np.inf
    with_infinity[[9, 25]] = -np.inf
    with_infinity[[10, 26]] = np.nan
    columns["with_infinity"] = with_infinity

    only_infinity = np.full(rows, np.nan)
    only_infinity[1::3] = np.inf
    only_infinity[2::6] = -np.inf
    columns["only_infinity"] = only_infinity

    # Repeated values force the oldest-occurrence tie rule.
    columns["ties"] = np.tile([2.0, 2.0, 2.0, 5.0, 5.0], rows // 5)

    zeros_and_signs = rng.normal(size=rows)
    zeros_and_signs[::5] = 0.0
    zeros_and_signs[1::5] *= -1.0
    columns["zeros_and_signs"] = zeros_and_signs

    columns["constant"] = np.full(rows, 3.5)
    columns["monotone"] = np.arange(rows, dtype=float)
    return pd.DataFrame(columns, index=index)


def assert_matches(name: str, window: int, produced, expected) -> None:
    assert produced.index.equals(expected.index), f"{name} w={window}: index drifted"
    assert produced.columns.equals(
        expected.columns
    ), f"{name} w={window}: columns drifted"
    left = produced.to_numpy(dtype=float)
    right = expected.to_numpy(dtype=float)
    missing_mismatch = np.isnan(left) != np.isnan(right)
    if missing_mismatch.any():
        row, column = np.argwhere(missing_mismatch)[0]
        raise AssertionError(
            f"{name} w={window}: missing-value mask differs at row {row}, "
            f"column {expected.columns[column]}: {left[row, column]!r} vs "
            f"{right[row, column]!r}"
        )
    both = ~np.isnan(left)
    if not both.any():
        return
    if not np.allclose(left[both], right[both], rtol=1e-9, atol=1e-9, equal_nan=False):
        difference = np.abs(left[both] - right[both])
        raise AssertionError(
            f"{name} w={window}: max absolute difference {difference.max():.3e}"
        )


def test_matches_callback_reference() -> None:
    frame = build_adversarial_frame()
    for window in (1, 2, 3, 5, 20, 41):
        for name, vectorized, reference in OPERATOR_PAIRS:
            assert_matches(name, window, vectorized(frame, window), reference(frame, window))
    print(f"operator equivalence: {len(OPERATOR_PAIRS)} operators x 6 windows")


def test_short_and_narrow_frames() -> None:
    index = pd.date_range("2024-01-02", periods=3, freq="B")
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]}, index=index)
    for window in (1, 2, 5):
        for name, vectorized, reference in OPERATOR_PAIRS:
            assert_matches(name, window, vectorized(frame, window), reference(frame, window))

    empty = pd.DataFrame(index=index)
    for name, vectorized, _ in OPERATOR_PAIRS:
        result = vectorized(empty, 3)
        assert result.shape == empty.shape, f"{name}: empty frame shape drifted"
    print("short frames and zero-column frames preserved")


def test_column_chunking_is_transparent() -> None:
    """A frame wider than one block must equal its per-column evaluation."""

    import engine

    frame = build_adversarial_frame()
    wide = pd.concat([frame] * 3, axis=1)
    wide.columns = [f"{name}_{position}" for position, name in enumerate(wide.columns)]
    original_budget = engine._ROLLING_BLOCK_BYTES
    try:
        # Force several blocks per call regardless of frame size.
        engine._ROLLING_BLOCK_BYTES = 1
        for name, vectorized, reference in OPERATOR_PAIRS:
            assert_matches(name, 5, vectorized(wide, 5), reference(wide, 5))
    finally:
        engine._ROLLING_BLOCK_BYTES = original_budget
    print(f"chunked evaluation matches on {wide.shape[1]} columns")


def test_alpha101_expression_still_matches() -> None:
    """The operators compose unchanged inside a real expression."""

    from engine import evaluate_expression

    frame = build_adversarial_frame().abs() + 0.5
    data = {
        "c": frame,
        "o": frame.shift(1).bfill(),
        "h": frame * 1.02,
        "l": frame * 0.98,
        "vol": frame * 1000.0,
    }
    expression = (
        "decay_linear(ts_argmax(c, 5) - ts_argmin(l, 5), 3) "
        "* ts_product(c / h, 4)"
    )
    produced = evaluate_expression(expression, data)
    expected = reference_decay_linear(
        reference_ts_argmax(data["c"], 5) - reference_ts_argmin(data["l"], 5), 3
    ) * reference_ts_product(data["c"] / data["h"], 4)
    assert_matches("composed expression", 0, produced, expected)
    print("composed Alpha101-style expression matches the reference")


def main() -> None:
    test_matches_callback_reference()
    test_short_and_narrow_frames()
    test_column_chunking_is_transparent()
    test_alpha101_expression_still_matches()
    print("OK: vectorized rolling operators match the callback reference")


if __name__ == "__main__":
    main()
