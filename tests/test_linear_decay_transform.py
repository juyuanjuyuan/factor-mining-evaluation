#!/usr/bin/env python3
"""Focused contract tests for the reusable run-level linear-decay transform."""

from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from transforms.linear_decay import (
    apply_linear_decay,
    validate_linear_decay_window,
)


def main() -> None:
    scores = pd.DataFrame(
        {"000001": [1.0, 2.0, 4.0], "000002": [3.0, 6.0, 12.0]},
        index=pd.date_range("2024-01-02", periods=3, freq="B"),
    )
    assert apply_linear_decay(scores, 1) is scores
    decayed = apply_linear_decay(scores, 3)
    assert decayed.index.equals(scores.index)
    assert decayed.columns.equals(scores.columns)
    np.testing.assert_allclose(
        decayed["000001"].to_numpy(),
        [1.0, 8 / 5, 17 / 6],
    )
    np.testing.assert_allclose(
        decayed["000002"].to_numpy(),
        [3.0, 24 / 5, 17 / 2],
    )

    with_gap = pd.DataFrame({"000001": [1.0, np.nan, 4.0]})
    np.testing.assert_allclose(
        apply_linear_decay(with_gap, 3)["000001"].to_numpy(),
        [1.0, 1.0, 13 / 4],
    )

    for invalid_decay in (0, -1, 1.0, 1.5, True):
        try:
            validate_linear_decay_window(invalid_decay)
        except ValueError:
            pass
        else:
            raise AssertionError(f"decay={invalid_decay!r} must be rejected")

    print("linear decay transform passed")


if __name__ == "__main__":
    main()
