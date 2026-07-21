#!/usr/bin/env python3
"""Validate all 30 VWAP-proxy Alpha101 expressions on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from engine import evaluate_expression, parse_and_validate_expression
from factor_registry import select_registered_factors


def main() -> None:
    rng = np.random.default_rng(101)
    days = pd.date_range("2023-01-02", periods=300, freq="B")
    codes = [f"{number:06d}" for number in range(20)]
    returns = rng.normal(0.0003, 0.018, size=(len(days), len(codes)))
    close = pd.DataFrame(
        30 * np.cumprod(1 + returns, axis=0),
        index=days,
        columns=codes,
    )
    open_ = close * pd.DataFrame(
        1 + rng.normal(0, 0.006, size=close.shape),
        index=days,
        columns=codes,
    )
    high = pd.DataFrame(
        np.maximum(open_, close)
        * (1 + rng.uniform(0.001, 0.02, size=close.shape)),
        index=days,
        columns=codes,
    )
    low = pd.DataFrame(
        np.minimum(open_, close)
        * (1 - rng.uniform(0.001, 0.02, size=close.shape)),
        index=days,
        columns=codes,
    )
    volume = pd.DataFrame(
        rng.integers(50_000, 20_000_000, size=close.shape),
        index=days,
        columns=codes,
        dtype=float,
    )
    amount = volume * close
    data = {
        "c": close,
        "o": open_,
        "h": high,
        "l": low,
        "vol": volume,
        "amt": amount,
        "vwap": (high + low) / 2,
    }

    factors = select_registered_factors(
        "all",
        implementation_set="vwap_proxy",
    )
    assert len(factors) == 30
    for factor in factors:
        parse_and_validate_expression(factor.expression)
        result = evaluate_expression(factor.expression, data)
        assert result.shape == close.shape, factor.name
        assert result.index.equals(close.index), factor.name
        assert result.columns.equals(close.columns), factor.name

    assert [
        factor.number
        for factor in select_registered_factors(
            "5,11,98",
            implementation_set="vwap_proxy",
        )
    ] == [5, 11, 98]
    print("30 VWAP-proxy Alpha101 expressions passed")


if __name__ == "__main__":
    main()
