#!/usr/bin/env python3
"""Registered model-training methods and rank-Ridge numerical contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from engine import rank_cs, winsorize_cs, zscore_cs
from model_training import (
    DEFAULT_MODEL_TRAINING_METHOD,
    ModelTerm,
    ModelTrainingContext,
    available_model_training_methods,
    normalize_training_parameters,
    run_model_training,
)
from returns import RETURN_DEFINITION


def synthetic_context(*, alpha: float = 0.0, raw_amplitude: bool = False) -> ModelTrainingContext:
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-02", periods=32, freq="B")
    columns = [f"{code:06d}" for code in range(1, 13)]
    factor_one = pd.DataFrame(
        rng.normal(size=(len(index), len(columns))), index=index, columns=columns
    )
    factor_two = pd.DataFrame(
        rng.normal(size=(len(index), len(columns))), index=index, columns=columns
    )
    factor_one.iloc[:, 0] *= 50
    if raw_amplitude:
        feature_one = zscore_cs(winsorize_cs(factor_one, 0.01, 0.99))
        feature_two = zscore_cs(winsorize_cs(factor_two, 0.01, 0.99))
    else:
        feature_one = rank_cs(factor_one)
        feature_two = rank_cs(factor_two)
    target = 0.02 * (feature_one - feature_one.mean(axis=1).to_numpy()[:, None])
    target -= 0.01 * (feature_two - feature_two.mean(axis=1).to_numpy()[:, None])

    open_prices = pd.DataFrame(100.0, index=index, columns=columns)
    for position in range(len(index) - 2):
        open_prices.iloc[position + 2] = (
            open_prices.iloc[position + 1].to_numpy()
            * (1.0 + target.iloc[position].to_numpy())
        )
    terms = (
        ModelTerm("synthetic", "factor_one", "c", 1.0),
        ModelTerm("synthetic", "factor_two", "vol", 1.0),
    )
    return ModelTrainingContext(
        terms=terms,
        market_data={"c": factor_one, "o": open_prices, "vol": factor_two},
        signal_start=index[0].date().isoformat(),
        signal_end=index[-1].date().isoformat(),
        horizon=1,
        parameters=(
            {
                "ridge_alpha": alpha,
                "winsor_lower_quantile": 0.01,
                "winsor_upper_quantile": 0.99,
            }
            if raw_amplitude
            else {"ridge_alpha": alpha}
        ),
    )


def main() -> None:
    metadata = {item.name: item for item in available_model_training_methods()}
    assert set(metadata) == {"manual_weights", "rank_ridge", "winsorized_zscore_ridge"}
    assert DEFAULT_MODEL_TRAINING_METHOD == "winsorized_zscore_ridge"
    assert metadata["manual_weights"].term_weight_editable
    assert not metadata["rank_ridge"].term_weight_editable
    assert metadata["rank_ridge"].requires_fitting
    assert metadata["winsorized_zscore_ridge"].requires_fitting
    assert normalize_training_parameters("rank_ridge", {}) == {"ridge_alpha": 1e-6}
    assert normalize_training_parameters("winsorized_zscore_ridge", {}) == {
        "ridge_alpha": 1e-6,
        "winsor_lower_quantile": 0.01,
        "winsor_upper_quantile": 0.99,
    }

    context = synthetic_context()
    result = run_model_training("rank_ridge", context)
    np.testing.assert_allclose(result.weights, (0.02, -0.01), atol=1e-10)
    assert "rank_cs((c))" in result.expression
    assert result.diagnostics["return_definition"] == RETURN_DEFINITION
    assert result.diagnostics["trading_day_count"] == 30
    assert result.diagnostics["observation_count"] == 30 * 12
    assert result.diagnostics["training_r_squared"] > 0.999999

    raw_context = synthetic_context(raw_amplitude=True)
    raw_result = run_model_training("winsorized_zscore_ridge", raw_context)
    np.testing.assert_allclose(raw_result.weights, (0.02, -0.01), atol=1e-10)
    assert "zscore_cs(winsorize_cs((c), 0.01, 0.99))" in raw_result.expression
    assert raw_result.diagnostics["feature_transform"].startswith("daily_winsorize")
    assert raw_result.diagnostics["training_r_squared"] > 0.999999

    manual = run_model_training(
        "manual_weights",
        ModelTrainingContext(
            terms=context.terms,
            market_data=context.market_data,
            signal_start=context.signal_start,
            signal_end=context.signal_end,
            horizon=context.horizon,
            parameters={},
        ),
    )
    assert manual.weights == (1.0, 1.0)

    try:
        normalize_training_parameters("rank_ridge", {"unknown": 1})
    except ValueError as exc:
        assert "does not accept" in str(exc)
    else:
        raise AssertionError("unknown training parameters must fail")

    print("model training methods passed")


if __name__ == "__main__":
    main()
