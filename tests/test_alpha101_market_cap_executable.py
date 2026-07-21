#!/usr/bin/env python3
"""Validate the market-cap Alpha101 set on synthetic wide data."""

from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from engine import evaluate_expression, parse_and_validate_expression
from factor_registry import select_registered_factors


def main() -> None:
    rng = np.random.default_rng(56)
    days = pd.date_range("2023-01-02", periods=40, freq="B")
    codes = [f"{number:06d}" for number in range(20)]
    returns = rng.normal(0.0003, 0.018, size=(len(days), len(codes)))
    close = pd.DataFrame(
        30 * np.cumprod(1 + returns, axis=0),
        index=days,
        columns=codes,
    )
    cap = pd.DataFrame(
        rng.lognormal(mean=24, sigma=1.2, size=close.shape),
        index=days,
        columns=codes,
    )
    factors = select_registered_factors(
        "all",
        implementation_set="market_cap",
    )
    assert len(factors) == 1
    factor = factors[0]
    parse_and_validate_expression(factor.expression)
    result = evaluate_expression(factor.expression, {"c": close, "cap": cap})
    assert result.shape == close.shape
    assert result.index.equals(close.index)
    assert result.columns.equals(close.columns)
    print("1 market-cap Alpha101 expression passed")


if __name__ == "__main__":
    main()
