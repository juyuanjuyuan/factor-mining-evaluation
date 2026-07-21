#!/usr/bin/env python3
"""Validate the canonical JSON-backed runnable Alpha101 registry."""

from __future__ import annotations

from _bootstrap import PROJECT_ROOT

from engine import parse_and_validate_expression
from factor_registry import (
    DEFAULT_FACTOR_BATCH,
    load_factor_batch,
    load_registered_factors,
    select_registered_factors,
)


def main() -> None:
    assert DEFAULT_FACTOR_BATCH == (
        PROJECT_ROOT / "factor_registry" / "alpha101_runnable_factors.json"
    )
    payload = load_factor_batch(DEFAULT_FACTOR_BATCH)
    factors = load_registered_factors()
    assert payload["factor_count"] == 83
    assert len(factors) == 83
    assert len({factor.number for factor in factors}) == 83
    assert len({factor.factor_name for factor in factors}) == 83
    assert [
        factor.number
        for factor in select_registered_factors("5,56,101")
    ] == [
        5,
        56,
        101,
    ]
    assert sum(factor.uses_proxy for factor in factors) == 30
    assert all(factor.entered_at for factor in factors)
    assert {
        implementation_set: len(
            select_registered_factors(
                "all",
                implementation_set=implementation_set,
            )
        )
        for implementation_set in ("exact", "vwap_proxy", "market_cap")
    } == {"exact": 52, "vwap_proxy": 30, "market_cap": 1}
    for factor in factors:
        parse_and_validate_expression(factor.expression)
    print("83-factor runnable Alpha101 registry passed")


if __name__ == "__main__":
    main()
