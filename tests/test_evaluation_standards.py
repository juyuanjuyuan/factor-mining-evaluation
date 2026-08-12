#!/usr/bin/env python3
"""Evaluation-standard registry, gate, and synthetic execution contracts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from evaluation_standards import (
    IC_METHOD_NAMES,
    PROFITABILITY_METHOD_NAMES,
    evaluate_factor_standards,
    evaluate_ic_gate,
    evaluate_profitability_gate,
    resolve_evaluation_standards,
)
from returns import RETURN_DEFINITION


def synthetic_market_data() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(20260721)
    days = pd.date_range("2023-01-03", periods=320, freq="B")
    codes = [f"{number:06d}" for number in range(1, 31)]
    innovations = rng.normal(0.0004, 0.012, size=(len(days), len(codes)))
    close = pd.DataFrame(
        20 * np.cumprod(1 + innovations, axis=0),
        index=days,
        columns=codes,
    )
    open_prices = close * pd.DataFrame(
        1 + rng.normal(0, 0.004, close.shape),
        index=days,
        columns=codes,
    )
    industry = pd.DataFrame(
        np.tile(np.repeat(np.array(["A", "B", "C"], dtype=object), 10), (len(days), 1)),
        index=days,
        columns=codes,
        dtype="string",
    )
    return {
        "c": close,
        "o": open_prices,
        "cap": pd.DataFrame(
            rng.lognormal(22, 0.7, close.shape),
            index=days,
            columns=codes,
        ),
        "limit": pd.DataFrame(0.10, index=days, columns=codes),
        "st": pd.DataFrame(False, index=days, columns=codes),
        "delisting": pd.DataFrame(False, index=days, columns=codes),
        "amt": pd.DataFrame(1e8, index=days, columns=codes),
        "industry": industry,
    }


def test_registry_matches_webapp_templates() -> None:
    standards = resolve_evaluation_standards("all")
    assert standards[0].method_names == IC_METHOD_NAMES
    assert standards[1].method_names == PROFITABILITY_METHOD_NAMES
    assert IC_METHOD_NAMES[1] == "industry_market_cap_neutralize"
    assert PROFITABILITY_METHOD_NAMES[0] == "industry_market_cap_neutralize"
    assert "fitness" in PROFITABILITY_METHOD_NAMES


def test_gate_logic() -> None:
    ic = evaluate_ic_gate(
        {
            "prefix_truncation_passed": True,
            "ic_mean": 0.01,
            "nw_ic_p_value": 0.049,
        }
    )
    assert ic.passed
    assert ic.conditions[1].name == "市值+行业联合中性 Rank IC 均值为正"
    assert not evaluate_ic_gate(
        {
            "prefix_truncation_passed": True,
            "ic_mean": -0.01,
            "nw_ic_p_value": 0.001,
        }
    ).passed

    by_sharpe = evaluate_profitability_gate(
        {
            "top_group_above_every_lower_group": True,
            "gn_rolling_sharpe_60_median": 1.0,
            "top_group_annualized_return": 0.10,
        }
    )
    assert by_sharpe.passed
    by_return = evaluate_profitability_gate(
        {
            "top_group_above_every_lower_group": True,
            "gn_rolling_sharpe_60_median": 0.2,
            "top_group_annualized_return": 0.300001,
        }
    )
    assert by_return.passed
    assert not evaluate_profitability_gate(
        {
            "top_group_above_every_lower_group": False,
            "gn_rolling_sharpe_60_median": 3.0,
            "top_group_annualized_return": 1.0,
        }
    ).passed
    # User specified annualized return strictly greater than 30%.
    assert not evaluate_profitability_gate(
        {
            "top_group_above_every_lower_group": True,
            "gn_rolling_sharpe_60_median": 0.9,
            "top_group_annualized_return": 0.30,
        }
    ).passed


def test_synthetic_standard_execution() -> None:
    data = synthetic_market_data()
    with tempfile.TemporaryDirectory() as temporary:
        result = evaluate_factor_standards(
            factor_name="standard_synthetic",
            expression="rank_cs(delta(c, 5))",
            data_dir=Path(temporary),
            output_dir=Path(temporary) / "outputs",
            signal_start="2023-03-01",
            signal_end="2024-03-15",
            standards="all",
            horizon=1,
            n_quantiles=5,
            preloaded_data=data,
        )
        assert result["return_definition"] == RETURN_DEFINITION
        assert set(result["standards"]) == {"ic_test", "profitability_test"}
        assert result["standards"]["ic_test"]["methods"] == list(IC_METHOD_NAMES)
        assert result["standards"]["profitability_test"]["methods"] == list(
            PROFITABILITY_METHOD_NAMES
        )
        assert "fitness" in result["standards"]["profitability_test"]["metrics"]
        assert Path(result["summary_path"]).is_file()
        for standard in result["standards"].values():
            assert Path(standard["metrics_path"]).is_file()
            assert Path(standard["code_path"]).is_file()
            assert standard["metrics"]["return_definition"] == RETURN_DEFINITION
            assert standard["metrics"]["sample_start_day"] >= "2023-03-01"
            assert standard["metrics"]["sample_end_day"] < "2024-03-15"


def main() -> None:
    test_registry_matches_webapp_templates()
    test_gate_logic()
    test_synthetic_standard_execution()
    print("evaluation standards contracts passed")


if __name__ == "__main__":
    main()
