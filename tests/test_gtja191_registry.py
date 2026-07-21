"""Validate and execute every locally runnable GTJA191 definition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import evaluate_expression, expression_data_symbols, parse_and_validate_expression
from factor_registry import load_factor_batch, load_registered_factors
from paths import PROJECT_ROOT


REGISTRY_PATH = PROJECT_ROOT / "factor_registry" / "gtja191_runnable_factors.json"
EXTERNAL_NUMBERS = {30, 75, 149, 181, 182}


def _synthetic_data() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(191)
    index = pd.date_range("2020-01-01", periods=320, freq="B")
    columns = [f"{number:06d}" for number in range(1, 13)]
    close = pd.DataFrame(rng.lognormal(4.0, 0.3, size=(len(index), len(columns))), index=index, columns=columns)
    spread = pd.DataFrame(rng.uniform(0.002, 0.08, size=close.shape), index=index, columns=columns)
    open_ = close * pd.DataFrame(1 + rng.normal(0, 0.01, size=close.shape), index=index, columns=columns)
    high = pd.DataFrame(np.maximum(close.to_numpy(), open_.to_numpy()) * (1 + spread.to_numpy()), index=index, columns=columns)
    low = pd.DataFrame(np.minimum(close.to_numpy(), open_.to_numpy()) * (1 - spread.to_numpy()), index=index, columns=columns)
    volume = pd.DataFrame(rng.lognormal(13.0, 0.5, size=close.shape), index=index, columns=columns)
    return {"c": close, "o": open_, "h": high, "l": low, "vol": volume, "vwap": (high + low) / 2}


def main() -> None:
    payload = load_factor_batch(REGISTRY_PATH)
    factors = load_registered_factors(REGISTRY_PATH)
    assert payload["factor_count"] == 186
    assert {factor.number for factor in factors} == set(range(1, 192)) - EXTERNAL_NUMBERS
    assert {item["number"] for item in payload["excluded_factors"]} == EXTERNAL_NUMBERS
    assert payload["set_counts"]["vwap_proxy"] == sum(factor.uses_proxy for factor in factors)

    data = _synthetic_data()
    for factor in factors:
        parse_and_validate_expression(factor.expression)
        assert set(factor.required_symbols) == expression_data_symbols(factor.expression)
        result = evaluate_expression(factor.expression, data)
        assert result.index.equals(data["c"].index), factor.factor_name
        assert list(result.columns) == list(data["c"].columns), factor.factor_name
    print("186 runnable GTJA191 factor definitions passed synthetic execution")


if __name__ == "__main__":
    main()
