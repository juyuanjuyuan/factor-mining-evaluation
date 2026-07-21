#!/usr/bin/env python3
"""Causality whitelist checks for expression operators."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from engine import expression_operator_names
from operator_causality import (
    OPERATOR_SPECS,
    causal_operator_whitelist,
    operator_causality_results,
)


def main() -> None:
    results = operator_causality_results()
    assert len(results) == len(OPERATOR_SPECS)
    failed = [result for result in results if not result.passed]
    assert not failed, [
        (result.name, result.expression, result.reason) for result in failed
    ]
    assert causal_operator_whitelist() == frozenset(
        spec.name for spec in OPERATOR_SPECS
    )
    assert expression_operator_names(
        "where(c > o, np.log(ts_mean(c, 5)), abs(delta(c, 2)))"
    ) == {"where", "np.log", "ts_mean", "abs", "delta"}
    print("operator causality whitelist passed")


if __name__ == "__main__":
    main()
