#!/usr/bin/env python3
"""Unit-level checks for composable factor evaluation methods."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from evaluators import (
    DEFAULT_EVALUATION_METHODS,
    EvaluationContext,
    evaluate_rank_ic,
    evaluate_rank_icir,
    evaluation_method_names,
    resolve_evaluation_methods,
    run_evaluation_methods,
)
from evaluators.quantile_net_returns import calculate_quantile_net_returns
from batch import _matching_completed
from returns import (
    RETURN_DEFINITION,
    calculate_forward_open_return,
)


def assert_default_quantile_net_costs() -> None:
    days = pd.date_range("2024-01-02", periods=2, freq="B")
    factor = pd.DataFrame(
        {"000001": [1.0, 2.0], "000002": [2.0, 1.0]},
        index=days,
    )
    forward_return = pd.DataFrame(0.0, index=days, columns=factor.columns)

    _, turnover, costs, metrics = calculate_quantile_net_returns(
        factor,
        forward_return,
        n_quantiles=2,
    )

    assert not {
        "quantile_net_buy_cost_rate",
        "quantile_net_sell_cost_rate",
        "quantile_net_round_trip_cost_rate",
    } & metrics.keys()
    assert np.isclose(turnover.iloc[0]["G1_buy_turnover"], 1.0)
    assert np.isclose(turnover.iloc[0]["G1_sell_turnover"], 0.0)
    assert np.isclose(costs.iloc[0]["G1"], 0.00125)
    assert np.isclose(turnover.iloc[1]["G1_buy_turnover"], 1.0)
    assert np.isclose(turnover.iloc[1]["G1_sell_turnover"], 1.0)
    assert np.isclose(costs.iloc[1]["G1"], 0.003)


def main() -> None:
    assert_default_quantile_net_costs()
    rng = np.random.default_rng(42)
    days = pd.date_range("2024-01-02", periods=18, freq="B")
    codes = [f"{number:06d}" for number in range(12)]
    daily_returns = rng.normal(0.0005, 0.015, size=(len(days), len(codes)))
    close = pd.DataFrame(
        100 * np.cumprod(1 + daily_returns, axis=0),
        index=days,
        columns=codes,
    )
    open_prices = close * pd.DataFrame(
        1 + rng.normal(0, 0.002, size=close.shape),
        index=days,
        columns=codes,
    )
    forward_return = calculate_forward_open_return(open_prices)
    # The synthetic factor deliberately equals its label so Rank IC is exact.
    factor = forward_return.copy()

    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        context = EvaluationContext(
            factor_name="modular_test",
            artifact_name="modular_test",
            factor=factor,
            close=close,
            forward_return=forward_return,
            horizon=1,
            n_quantiles=4,
            output_dir=output_dir,
        )

        rank_methods = resolve_evaluation_methods("rank_icir")
        assert evaluation_method_names(rank_methods) == ("rank_ic", "rank_icir")
        rank_state = run_evaluation_methods(context, rank_methods)
        assert set(rank_state.details) == {"ic"}
        assert np.isclose(rank_state.metrics["ic_mean"], 1.0)
        assert "gn_final_cumulative" not in rank_state.metrics
        assert not rank_state.artifacts

        plot_methods = resolve_evaluation_methods("quantile_plot")
        assert evaluation_method_names(plot_methods) == (
            "quantile_returns",
            "quantile_cumulative",
            "quantile_plot",
        )
        plot_state = run_evaluation_methods(context, plot_methods)
        assert set(plot_state.details) == {
            "group_returns",
            "cumulative_returns",
        }
        assert plot_state.artifacts["plot"].is_file()
        assert "ic_mean" not in plot_state.metrics

        cycle_methods = resolve_evaluation_methods("cycle_context")
        assert evaluation_method_names(cycle_methods) == (
            "quantile_returns",
            "quantile_cumulative",
            "quantile_plot",
            "cycle_context",
        )
        cycle_state = run_evaluation_methods(context, cycle_methods)
        assert cycle_state.executed_methods[-1] == "cycle_context"
        assert set(cycle_state.details) == set(plot_state.details)
        assert set(cycle_state.artifacts) == {"plot"}
        assert cycle_state.metrics == plot_state.metrics
        assert cycle_state.artifacts["plot"].is_file()

        assert evaluation_method_names(
            resolve_evaluation_methods("quantile_cumulative")
        ) == ("quantile_returns", "quantile_cumulative")
        net_methods = resolve_evaluation_methods(
            "quantile_net_returns,quantile_cumulative"
        )
        assert evaluation_method_names(net_methods) == (
            "quantile_net_returns",
            "quantile_cumulative",
        )
        assert evaluation_method_names(
            resolve_evaluation_methods(
                "quantile_net_returns,quantile_returns,quantile_cumulative"
            )
        ) == (
            "quantile_net_returns",
            "quantile_returns",
            "quantile_cumulative",
        )
        net_state = run_evaluation_methods(context, net_methods)
        assert set(net_state.details) == {
            "group_returns",
            "quantile_turnover",
            "quantile_transaction_cost",
            "cumulative_returns",
        }
        gross_returns = plot_state.details["group_returns"]
        net_returns = net_state.details["group_returns"]
        assert isinstance(gross_returns, pd.DataFrame)
        assert isinstance(net_returns, pd.DataFrame)
        assert (net_returns <= gross_returns).all().all()
        turnover = net_state.details["quantile_turnover"]
        costs = net_state.details["quantile_transaction_cost"]
        assert isinstance(turnover, pd.DataFrame)
        assert isinstance(costs, pd.DataFrame)
        assert np.isclose(turnover.iloc[0]["G4_buy_turnover"], 1.0)
        assert np.isclose(turnover.iloc[0]["G4_sell_turnover"], 0.0)
        assert (costs >= 0).all().all()
        assert "gn_mean_daily_one_way_turnover" in net_state.metrics
        assert "gn_mean_daily_transaction_cost" in net_state.metrics

        full_state = run_evaluation_methods(
            context,
            DEFAULT_EVALUATION_METHODS,
        )
        assert evaluation_method_names(DEFAULT_EVALUATION_METHODS) == (
            "rank_ic",
            "rank_icir",
            "quantile_returns",
            "quantile_cumulative",
            "quantile_plot",
        )
        assert {"ic", "group_returns", "cumulative_returns"} == set(
            full_state.details
        )

        # A method may run more than once (e.g. IC before and after
        # neutralization): later outputs get versioned keys, earlier outputs
        # are preserved, and dependents read the most recent version.
        repeated_state = run_evaluation_methods(
            context,
            (evaluate_rank_ic, evaluate_rank_icir, evaluate_rank_ic, evaluate_rank_icir),
        )
        assert repeated_state.executed_methods == [
            "rank_ic",
            "rank_icir",
            "rank_ic",
            "rank_icir",
        ]
        assert {"ic", "ic__2"} == set(repeated_state.details)
        assert "ic_mean" in repeated_state.metrics
        assert "ic_mean__2" in repeated_state.metrics
        assert np.isclose(
            repeated_state.metrics["ic_mean"],
            repeated_state.metrics["ic_mean__2"],
        )

        try:
            run_evaluation_methods(context, (evaluate_rank_icir,))
        except ValueError as exc:
            assert "requires methods to run first" in str(exc)
        else:
            raise AssertionError("Rank ICIR must require Rank IC")

        try:
            resolve_evaluation_methods("not_a_method")
        except ValueError as exc:
            assert "Unknown evaluation method" in str(exc)
        else:
            raise AssertionError("Unknown evaluation methods must be rejected")

        metrics_path = output_dir / "legacy_metrics.csv"
        pd.DataFrame(
            [
                {
                    "factor_name": "legacy_factor",
                    "expression": "c",
                    "horizon": 1,
                    "n_quantiles": 10,
                    "evaluation_methods": np.nan,
                }
            ]
        ).to_csv(metrics_path, index=False)
        factors = (SimpleNamespace(name="legacy_factor", expression="c"),)
        default_names = evaluation_method_names(DEFAULT_EVALUATION_METHODS)
        # A result without an explicit return definition used the legacy
        # close-to-close label and must be rerun.
        assert not _matching_completed(
            metrics_path,
            factors,
            1,
            10,
            default_names,
        )
        current_metrics = pd.read_csv(metrics_path)
        current_metrics["return_definition"] = RETURN_DEFINITION
        current_metrics.to_csv(metrics_path, index=False)
        assert _matching_completed(
            metrics_path,
            factors,
            1,
            10,
            default_names,
        ) == {"legacy_factor"}
        assert not _matching_completed(
            metrics_path,
            factors,
            1,
            10,
            ("rank_ic", "rank_icir"),
        )

    print("composable evaluation methods passed")


if __name__ == "__main__":
    main()
