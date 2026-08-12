#!/usr/bin/env python3
"""Synthetic tests for the optional extended evaluation modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from engine import evaluate_factor_expression
from market_cycles import cycle_backgrounds_for_dates, market_cycle_backgrounds
from evaluators import (
    DEFAULT_EVALUATION_METHODS,
    EvaluationContext,
    EvaluationState,
    HOLDING_AUDIT_COLUMNS,
    HOLDING_AUDIT_SUMMARY,
    calculate_yearly_fitness,
    compare_ic_trend_filters,
    evaluate_fitness,
    evaluate_holding_audit,
    evaluate_ic_horizon_decay,
    evaluate_market_cap_neutralization,
    evaluate_ic_peak_decay,
    evaluate_ic_trend_filter,
    evaluate_quantile_net_returns,
    evaluate_quantile_returns,
    evaluate_rolling_drawdown,
    evaluate_rolling_sharpe,
    evaluate_top_quantile_performance,
    evaluate_tradability_filter,
    evaluation_method_names,
    future_data_perturbation_test,
    fourier_low_pass_ic,
    ic_peak_decay,
    ic_horizon_decay,
    kalman_ic_trend,
    mask_untradeable_entries,
    newey_west_mean_test,
    non_overlapping_top_quantile_leadership,
    overlapping_filtered_ic_mean,
    prefix_truncation_consistency_test,
    resolve_evaluation_methods,
    run_evaluation_methods,
    select_newey_west_lag,
    second_order_low_pass_ic,
    top_quantile_performance,
)
from evaluators.base import method_metadata
from evaluators.rolling_drawdown import (
    evaluate_rolling_drawdown as direct_evaluate_rolling_drawdown,
)
from evaluators.rolling_sharpe import (
    evaluate_rolling_sharpe as direct_evaluate_rolling_sharpe,
)
from transforms import neutralize_factor_by_market_cap
from transforms.delisting_status import normalize_delisting_status_frame
from transforms.price_limits import (
    infer_price_limit_ratio_frame,
    normalize_st_status_frame,
    price_limit_ratio_for_code,
    reconcile_incremental_st_status,
)


def test_market_cycle_backgrounds() -> None:
    backgrounds = market_cycle_backgrounds()
    assert [item["direction"] for item in backgrounds] == [
        "up",
        "down",
        "up",
        "down",
        "up",
        "down",
        "up",
    ]
    assert backgrounds[-1]["provisional"]
    assert backgrounds[-1]["end_day"] is None

    dates = pd.DatetimeIndex(["2015-06-11", "2015-06-12", "2015-06-15"])
    clipped = cycle_backgrounds_for_dates(dates)
    assert [(item.start_day, item.end_day, item.direction) for item in clipped] == [
        ("2015-06-11", "2015-06-12", "up"),
        ("2015-06-12", "2015-06-15", "down"),
    ]


def test_future_perturbation() -> None:
    rng = np.random.default_rng(11)
    days = pd.date_range("2024-01-02", periods=40, freq="B")
    codes = [f"{number:06d}" for number in range(12)]
    close = pd.DataFrame(
        rng.lognormal(4.0, 0.3, size=(len(days), len(codes))),
        index=days,
        columns=codes,
    )
    data = {"c": close}

    trailing_builder = lambda inputs: inputs["c"].rolling(3, min_periods=1).mean()
    trailing_factor = trailing_builder(data)
    detail, summary = future_data_perturbation_test(
        trailing_factor,
        data,
        trailing_builder,
        checkpoint_positions=(5, 15, 30),
    )
    assert "future_perturbation_passed" not in summary
    assert summary["future_perturbation_changed_values"] == 0
    assert detail["changed_values"].sum() == 0

    future_builder = lambda inputs: inputs["c"].shift(-1)
    future_factor = future_builder(data)
    _, leaking_summary = future_data_perturbation_test(
        future_factor,
        data,
        future_builder,
        checkpoint_positions=(5, 15, 30),
    )
    assert leaking_summary["future_perturbation_changed_values"] > 0
    assert leaking_summary["future_perturbation_changed_checkpoints"] == 3


def test_prefix_truncation_consistency() -> None:
    rng = np.random.default_rng(12)
    days = pd.date_range("2024-01-02", periods=40, freq="B")
    codes = [f"{number:06d}" for number in range(12)]
    close = pd.DataFrame(
        rng.lognormal(4.0, 0.3, size=(len(days), len(codes))),
        index=days,
        columns=codes,
    )
    data = {"c": close}

    trailing_builder = lambda inputs: inputs["c"].rolling(3, min_periods=1).mean()
    trailing_factor = trailing_builder(data)
    detail, summary = prefix_truncation_consistency_test(
        trailing_factor,
        data,
        trailing_builder,
        checkpoint_positions=(5, 15, 30),
    )
    assert summary["prefix_truncation_passed"]
    assert detail["changed_values"].sum() == 0

    future_builder = lambda inputs: inputs["c"].shift(-1)
    future_factor = future_builder(data)
    _, leaking_summary = prefix_truncation_consistency_test(
        future_factor,
        data,
        future_builder,
        checkpoint_positions=(5, 15, 30),
    )
    assert not leaking_summary["prefix_truncation_passed"]
    assert leaking_summary["prefix_truncation_changed_checkpoints"] == 3


def test_newey_west() -> None:
    assert select_newey_west_lag(
        [1.0, 0.30, -0.20, 0.04, 0.03, 0.02],
        significance_bound=0.05,
    ) == 2
    assert select_newey_west_lag(
        [1.0, 0.04, 0.03, 0.02, 0.01],
        significance_bound=0.05,
        min_lag=3,
    ) == 3

    rng = np.random.default_rng(22)
    values = np.empty(600)
    values[0] = 0.04
    for index in range(1, len(values)):
        values[index] = (
            0.04
            + 0.65 * (values[index - 1] - 0.04)
            + rng.normal(0, 0.02)
        )
    series = pd.Series(values)
    metrics, detail = newey_west_mean_test(series, horizon=5)
    assert metrics["nw_ic_mean"] > 0
    assert metrics["nw_ic_p_value"] < 0.05
    assert metrics["nw_ic_significant_5pct"]
    assert metrics["nw_ic_lag"] >= 4
    assert metrics["nw_ic_lag_min"] == 4
    assert np.isclose(metrics["nw_ic_autocorr_bound"], 2.0 / np.sqrt(600))
    assert np.isclose(detail.loc[0, "autocorrelation"], 1.0)
    assert not detail.loc[0, "within_significance_band"]
    assert detail["selected_lag"].sum() == 1


def test_ic_peak_decay_is_60_day_rolling_mean_ic() -> None:
    days = pd.date_range("2024-01-02", periods=125, freq="B")
    values = np.linspace(-0.05, 0.05, len(days))
    series = pd.Series(values, index=days, name="ic")

    metrics, detail = ic_peak_decay(series)

    assert metrics == {
        "ic_peak_decay_window_days": 60,
        "ic_peak_decay_step_days": 10,
        "ic_peak_decay_overlap_days": 50,
        "ic_peak_decay_valid_ic_days": 125,
        "ic_peak_decay_full_window_count": 7,
    }
    assert detail.index.equals(days[[59, 69, 79, 89, 99, 109, 119]])
    assert detail.columns.tolist() == ["mean_ic"]
    assert np.allclose(
        detail["mean_ic"].to_numpy(),
        [series.iloc[start : start + 60].mean() for start in range(0, 66, 10)],
    )

    context = EvaluationContext(
        factor_name="ic_peak_decay",
        artifact_name="ic_peak_decay",
        factor=pd.DataFrame(0.0, index=days, columns=["000001"]),
        close=pd.DataFrame(0.0, index=days, columns=["000001"]),
        forward_return=pd.DataFrame(0.0, index=days, columns=["000001"]),
        horizon=1,
        n_quantiles=1,
        output_dir=Path("."),
    )
    state = EvaluationState(context)
    state.add_detail("ic", series)
    evaluate_ic_peak_decay(state)
    assert state.details["ic_peak_decay"].equals(detail)
    assert evaluation_method_names(resolve_evaluation_methods("ic_peak_decay")) == (
        "rank_ic",
        "ic_peak_decay",
    )


def test_ic_horizon_decay_uses_one_common_signal_sample() -> None:
    days = pd.date_range("2024-01-02", periods=260, freq="B")
    codes = [f"{number:06d}" for number in range(5)]
    growth = np.linspace(0.0001, 0.0005, len(codes))
    open_prices = pd.DataFrame(
        np.exp(np.arange(len(days))[:, None] * growth[None, :]),
        index=days,
        columns=codes,
    )
    factor = pd.DataFrame(
        np.tile(growth, (len(days), 1)),
        index=days,
        columns=codes,
    )

    metrics, detail = ic_horizon_decay(factor, open_prices)
    assert detail.index.equals(pd.Index(range(20, 253), name="horizon"))
    assert detail.columns.tolist() == ["mean_rank_ic"]
    assert np.allclose(detail["mean_rank_ic"], 1.0)
    assert metrics["ic_horizon_decay_horizon_count"] == 233
    assert metrics["ic_horizon_decay_structural_signal_day_count"] == 7
    assert metrics["ic_horizon_decay_common_signal_day_count"] == 7
    assert metrics["ic_horizon_decay_common_end_day"] == days[6].date().isoformat()

    # open[t+1+20] is unavailable only for the first candidate signal date.
    # Exact common-sample semantics must therefore remove that date from every H.
    missing_one_horizon = open_prices.copy()
    missing_one_horizon.iloc[21] = np.nan
    dropped_metrics, dropped_detail = ic_horizon_decay(
        factor,
        missing_one_horizon,
    )
    assert np.allclose(dropped_detail["mean_rank_ic"], 1.0)
    assert dropped_metrics["ic_horizon_decay_structural_signal_day_count"] == 7
    assert dropped_metrics["ic_horizon_decay_common_signal_day_count"] == 6
    assert dropped_metrics["ic_horizon_decay_dropped_signal_day_count"] == 1
    assert dropped_metrics["ic_horizon_decay_common_start_day"] == days[1].date().isoformat()

    bounded_factor = factor.loc[days[2] : days[257]]
    context = EvaluationContext(
        factor_name="ic_horizon_decay",
        artifact_name="ic_horizon_decay",
        factor=bounded_factor,
        close=open_prices.loc[bounded_factor.index],
        forward_return=pd.DataFrame(0.0, index=bounded_factor.index, columns=codes),
        horizon=1,
        n_quantiles=5,
        output_dir=Path("."),
        market_data={"o": open_prices},
        source_factor=factor,
    )
    state = EvaluationState(context)
    evaluate_ic_horizon_decay(state)
    assert state.details["ic_horizon_decay"].equals(detail)
    assert state.metrics["ic_horizon_decay_common_signal_day_count"] == 5
    assert state.metrics["ic_horizon_decay_common_start_day"] == days[2].date().isoformat()
    assert state.metrics["ic_horizon_decay_common_end_day"] == days[6].date().isoformat()

    resolved = resolve_evaluation_methods("ic_horizon_decay")
    assert evaluation_method_names(resolved) == ("ic_horizon_decay",)
    assert method_metadata(resolved[0]).requires == ()
    assert method_metadata(resolved[0]).required_data_symbols == ("o",)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        open_prices.to_parquet(root / "open_df.pq")
        open_prices.to_parquet(root / "close_df.pq")
        result = evaluate_factor_expression(
            factor_name="ic_horizon_decay_pipeline",
            expression="rank_cs(c)",
            data_dir=root,
            output_dir=root / "result",
            evaluation_methods=resolved,
        )
        assert result["evaluation_methods"] == ["ic_horizon_decay"]
        assert set(result["detail_paths"]) == {"ic_horizon_decay"}
        persisted = pd.read_csv(
            result["detail_paths"]["ic_horizon_decay"],
            index_col=0,
        )
        assert persisted.index.tolist() == list(range(20, 253))
        assert persisted.columns.tolist() == ["mean_rank_ic"]
        assert result["metrics"]["ic_horizon_decay_max_horizon"] == 252
        assert '"ic_horizon_decay":"details/' in result["metrics"][
            "evaluation_details"
        ]


def test_ic_trend_filter() -> None:
    days = pd.date_range("2024-01-02", periods=5, freq="B")
    series = pd.Series([0.0, np.nan, 0.20, 0.20, 0.20], index=days, name="ic")

    metrics, detail = kalman_ic_trend(series)

    assert detail.index.equals(days[[0, 2, 3, 4]])
    assert detail.columns.tolist() == [
        "daily_ic",
        "filtered_ic",
        "second_order_low_pass_ic",
        "fourier_low_pass_ic",
    ]
    assert np.allclose(detail["daily_ic"].to_numpy(), [0.0, 0.20, 0.20, 0.20])
    assert detail.iloc[0]["filtered_ic"] == 0.0
    assert 0.0 < detail.iloc[1]["filtered_ic"] < 0.20
    assert detail.iloc[-1]["filtered_ic"] < 0.20
    assert np.isclose(metrics["ic_trend_filter_process_to_observation_ratio"], 0.01)
    assert 0.09 < metrics["ic_trend_filter_kalman_gain"] < 0.10
    assert np.isclose(metrics["ic_trend_filter_start"], 0.0)
    assert np.isclose(metrics["ic_trend_filter_final"], detail.iloc[-1]["filtered_ic"])
    assert np.isclose(
        metrics["ic_trend_filter_change"],
        detail.iloc[-1]["filtered_ic"],
    )
    assert metrics["ic_trend_filter_valid_ic_days"] == 4
    assert metrics["ic_trend_filter_second_order_period"] == 31
    assert metrics["ic_trend_filter_fourier_min_period"] == 31
    assert metrics["ic_trend_filter_fourier_retained_frequency_count"] == 1
    assert np.isclose(
        metrics["ic_trend_filter_second_order_final"],
        detail.iloc[-1]["second_order_low_pass_ic"],
    )
    assert np.isclose(
        metrics["ic_trend_filter_fourier_final"],
        detail.iloc[-1]["fourier_low_pass_ic"],
    )

    constant = np.full(40, 0.07)
    assert np.allclose(second_order_low_pass_ic(constant), constant)
    constant_fourier, retained_count = fourier_low_pass_ic(constant)
    assert np.allclose(constant_fourier, constant)
    assert retained_count == 2

    observations = np.array([0.01, 0.03, -0.02, 0.04, 0.00])
    second_order = second_order_low_pass_ic(observations)
    alpha = 2.0 / 31.0
    expected_at_three = (
        2.0 * (1.0 - alpha) * observations[2]
        - (1.0 - alpha) ** 2 * observations[1]
        + (alpha - 0.25 * alpha**2) * observations[2]
        + 0.5 * alpha**2 * observations[1]
        - (alpha - 0.75 * alpha**2) * observations[0]
    )
    assert np.isclose(second_order[3], expected_at_three)

    causal_input = np.linspace(-0.05, 0.05, 50)
    causal_perturbed = causal_input.copy()
    causal_perturbed[30:] += 0.5
    assert np.allclose(
        second_order_low_pass_ic(causal_input)[:31],
        second_order_low_pass_ic(causal_perturbed)[:31],
    )

    positions = np.arange(128)
    low_frequency = np.sin(2.0 * np.pi * positions / 64.0)
    high_frequency = 0.4 * np.sin(2.0 * np.pi * positions / 4.0)
    fourier_filtered, retained_count = fourier_low_pass_ic(
        low_frequency + high_frequency,
        min_period=31,
    )
    assert np.allclose(fourier_filtered, low_frequency, atol=1e-12)
    assert retained_count == 5
    fourier_perturbed_input = low_frequency + high_frequency
    fourier_perturbed_input[-1] += 1.0
    fourier_perturbed, _ = fourier_low_pass_ic(
        fourier_perturbed_input,
        min_period=31,
    )
    assert not np.allclose(fourier_filtered[:20], fourier_perturbed[:20])

    comparison_metrics, comparison_detail = compare_ic_trend_filters(series)
    assert comparison_metrics == metrics
    assert comparison_detail.equals(detail)

    mean_days = pd.date_range("2024-02-01", periods=25, freq="B")
    mean_series = pd.Series(
        np.linspace(-0.08, 0.12, len(mean_days)),
        index=mean_days,
        name="ic",
    )
    mean_source_metrics, mean_source_detail = compare_ic_trend_filters(mean_series)
    mean_metrics, mean_detail = overlapping_filtered_ic_mean(mean_source_detail)
    assert mean_source_metrics["ic_trend_filter_valid_ic_days"] == 25
    assert mean_detail.index.equals(mean_days[[9, 14, 19, 24]])
    assert mean_detail.columns.tolist() == [
        "kalman_filtered_ic_mean",
        "second_order_low_pass_ic_mean",
        "fourier_low_pass_ic_mean",
    ]
    expected_mean_columns = [
        "filtered_ic",
        "second_order_low_pass_ic",
        "fourier_low_pass_ic",
    ]
    expected_means = np.vstack(
        [
            mean_source_detail.iloc[start : start + 10][expected_mean_columns]
            .mean(axis=0)
            .to_numpy()
            for start in (0, 5, 10, 15)
        ]
    )
    assert np.allclose(mean_detail.to_numpy(), expected_means)
    assert mean_metrics == {
        "ic_trend_filter_mean_window_days": 10,
        "ic_trend_filter_mean_step_days": 5,
        "ic_trend_filter_mean_overlap_days": 5,
        "ic_trend_filter_mean_full_window_count": 4,
    }

    try:
        overlapping_filtered_ic_mean(mean_source_detail.iloc[:9])
    except ValueError as exc:
        assert "complete window of 10" in str(exc)
    else:
        raise AssertionError("Nine filtered IC observations must not form a window")

    try:
        kalman_ic_trend(series.iloc[:1])
    except ValueError as exc:
        assert "at least four finite daily IC values" in str(exc)
    else:
        raise AssertionError("One IC observation must not produce a trend filter")

    context = EvaluationContext(
        factor_name="ic_trend_filter",
        artifact_name="ic_trend_filter",
        factor=pd.DataFrame(0.0, index=mean_days, columns=["000001"]),
        close=pd.DataFrame(0.0, index=mean_days, columns=["000001"]),
        forward_return=pd.DataFrame(0.0, index=mean_days, columns=["000001"]),
        horizon=1,
        n_quantiles=1,
        output_dir=Path("."),
    )
    state = EvaluationState(context)
    state.add_detail("ic", mean_series)
    evaluate_ic_trend_filter(state)
    assert state.details["ic_trend_filter"].equals(mean_source_detail)
    assert state.details["ic_trend_filter_mean_10"].equals(mean_detail)
    assert state.metrics["ic_trend_filter_mean_full_window_count"] == 4
    assert evaluation_method_names(resolve_evaluation_methods("ic_trend_filter")) == (
        "rank_ic",
        "ic_trend_filter",
    )


def test_market_cap_neutralization() -> None:
    rng = np.random.default_rng(33)
    days = pd.date_range("2023-01-02", periods=20, freq="B")
    codes = [f"{number:06d}" for number in range(40)]
    market_cap = pd.DataFrame(
        rng.lognormal(24, 1.0, size=(len(days), len(codes))),
        index=days,
        columns=codes,
    )
    factor = 2.0 + 3.0 * np.log(market_cap) + pd.DataFrame(
        rng.normal(0, 0.1, size=market_cap.shape),
        index=days,
        columns=codes,
    )
    residuals, diagnostics = neutralize_factor_by_market_cap(
        factor,
        market_cap,
    )
    for day in days:
        correlation = np.corrcoef(
            residuals.loc[day],
            np.log(market_cap.loc[day]),
        )[0, 1]
        assert abs(correlation) < 1e-10
        assert abs(float(residuals.loc[day].mean())) < 1e-10
    assert diagnostics["r_squared"].mean() > 0.99

    context = EvaluationContext(
        factor_name="cap_neutralization",
        artifact_name="cap_neutralization",
        factor=factor,
        close=factor,
        forward_return=factor * 0,
        horizon=1,
        n_quantiles=10,
        output_dir=Path("."),
        market_data={"c": factor, "cap": market_cap},
    )
    state = run_evaluation_methods(
        context,
        (evaluate_market_cap_neutralization,),
    )
    assert state.metrics["market_cap_neutralized_days"] == len(days)
    assert "market_cap_neutralization" in state.details


def test_top_quantiles_and_rolling_evaluators() -> None:
    assert evaluate_rolling_sharpe is direct_evaluate_rolling_sharpe
    assert evaluate_rolling_drawdown is direct_evaluate_rolling_drawdown
    temporary_output = tempfile.TemporaryDirectory()
    rng = np.random.default_rng(44)
    days = pd.date_range("2022-01-03", periods=320, freq="B")
    codes = [f"{number:06d}" for number in range(50)]
    base = np.arange(len(codes), dtype=float)
    factor_values = np.vstack(
        [np.roll(base, day_number * 3) for day_number in range(len(days))]
    )
    factor = pd.DataFrame(factor_values, index=days, columns=codes)
    normalized_factor = factor.rank(axis=1, pct=True)
    forward_return = 0.004 * normalized_factor + pd.DataFrame(
        rng.normal(0, 0.003, size=factor.shape),
        index=days,
        columns=codes,
    )
    close = pd.DataFrame(100.0, index=days, columns=codes)
    context = EvaluationContext(
        factor_name="top_and_risk",
        artifact_name="top_and_risk",
        factor=factor,
        close=close,
        forward_return=forward_return,
        horizon=1,
        n_quantiles=10,
        output_dir=Path(temporary_output.name),
    )
    state = run_evaluation_methods(
        context,
        (
            evaluate_quantile_returns,
            evaluate_top_quantile_performance,
            evaluate_rolling_sharpe,
            evaluate_rolling_drawdown,
        ),
    )
    assert state.metrics["top_group_above_every_lower_group"]
    legacy_top_pair_marker = "top" + "2"
    assert not any(legacy_top_pair_marker in key for key in state.metrics)
    assert "top_group_minus_other_mean_return" not in state.metrics
    top_detail = state.details["top_quantile_performance"]
    assert not any(legacy_top_pair_marker in column for column in top_detail.columns)
    assert top_detail["top_group_minus_other"].mean() > 0
    expected_daily_best = (
        top_detail["top_group_return"] >= top_detail["best_lower_group_return"]
    ).mean()
    assert np.isclose(
        state.metrics["top_group_outperformance_ratio"],
        expected_daily_best,
    )
    leadership = state.details["top_quantile_leadership_60"]
    assert len(leadership) == len(days) // 60
    assert "g8_cumulative" in leadership
    assert "top_group_is_best" in leadership
    assert not any(legacy_top_pair_marker in column for column in leadership.columns)
    assert state.metrics["top_group_60d_window_count"] == len(leadership)
    assert 0.0 <= state.metrics["top_group_60d_best_window_ratio"] <= 1.0
    expected_top_group_cumulative = float(
        (1 + top_detail["top_group_return"]).cumprod().iloc[-1] - 1
    )
    assert np.isclose(
        state.metrics["top_group_final_cumulative"],
        expected_top_group_cumulative,
    )
    assert np.isclose(
        top_detail["top_group_cumulative_return"].iloc[-1],
        expected_top_group_cumulative,
    )
    expected_top_group_annualized = float(
        (1.0 + expected_top_group_cumulative) ** (252.0 / len(top_detail)) - 1.0
    )
    assert np.isclose(
        state.metrics["top_group_annualized_return"],
        expected_top_group_annualized,
    )
    assert "turnover" not in top_detail
    assert "top_group_mean_daily_turnover" not in state.metrics
    assert "top_group_annualized_turnover" not in state.metrics

    net_state = run_evaluation_methods(
        context,
        (
            evaluate_quantile_net_returns,
            evaluate_top_quantile_performance,
        ),
    )
    net_returns = net_state.details["group_returns"]
    assert isinstance(net_returns, pd.DataFrame)
    expected_net_top_group = net_returns["G10"]
    expected_net_cumulative = float(
        (1 + expected_net_top_group).cumprod().iloc[-1] - 1
    )
    net_top_detail = net_state.details["top_quantile_performance"]
    assert np.allclose(
        net_top_detail["top_group_return"].to_numpy(),
        expected_net_top_group.to_numpy(),
    )
    assert np.isclose(
        net_state.metrics["top_group_final_cumulative"],
        expected_net_cumulative,
    )
    assert np.isclose(
        net_state.metrics["top_group_annualized_return"],
        (1.0 + expected_net_cumulative) ** (252.0 / len(net_top_detail)) - 1.0,
    )
    assert net_state.metrics["top_group_final_cumulative"] < expected_top_group_cumulative
    group_returns = state.details["group_returns"]
    assert isinstance(group_returns, pd.DataFrame)
    sharpe_plot = state.artifacts["rolling_sharpe_60_plot"]
    assert sharpe_plot.name == "top_and_risk__rolling_sharpe_60.png"
    assert sharpe_plot.is_file() and sharpe_plot.stat().st_size > 0
    for window in (20, 60, 252):
        sharpe = state.details[f"rolling_sharpe_{window}"]
        drawdown = state.details[f"rolling_drawdown_{window}"]
        full_window_count = len(group_returns) // window
        expected_end_days = group_returns.index[
            window - 1 : full_window_count * window : window
        ]
        assert len(sharpe) == full_window_count
        assert len(drawdown) == full_window_count
        # A point at t is the full block ending at t: [t-(window-1), ..., t].
        assert sharpe.index[0] == group_returns.index[window - 1]
        assert sharpe.index.equals(expected_end_days)
        assert drawdown.index.equals(expected_end_days)
        assert np.isfinite(sharpe.iloc[-1]["G10"])
        assert drawdown.iloc[-1]["G10"] <= 0
        expected_first_sharpe = (
            group_returns.iloc[:window].mean()
            / group_returns.iloc[:window].std(ddof=1)
            * np.sqrt(252)
        )
        assert np.allclose(
            sharpe.iloc[0].to_numpy(),
            expected_first_sharpe.to_numpy(),
            equal_nan=True,
        )
        first_window_top = group_returns["G10"].iloc[:window].to_numpy()
        first_window_wealth = np.cumprod(1.0 + first_window_top)
        first_window_peaks = np.maximum.accumulate(
            np.concatenate(([1.0], first_window_wealth))
        )[1:]
        expected_first_drawdown = float(
            np.min(first_window_wealth / first_window_peaks - 1.0)
        )
        assert np.isclose(drawdown.iloc[0]["G10"], expected_first_drawdown)
        pos_share = state.metrics[
            f"gn_rolling_sharpe_{window}_pos_share"
        ]
        assert 0.0 <= pos_share <= 1.0
        assert (
            state.metrics[f"gn_rolling_sharpe_{window}_min"]
            <= state.metrics[f"gn_rolling_sharpe_{window}_median"]
        )
        assert state.metrics[
            f"gn_rolling_drawdown_{window}_worst"
        ] <= 0
        assert (
            state.metrics[f"gn_rolling_drawdown_{window}_worst"]
            <= state.metrics[
                f"gn_rolling_drawdown_{window}_median"
            ]
        )
        assert f"gn_rolling_sharpe_{window}_latest" not in state.metrics
    # The strong synthetic signal should be positive in most windows.
    assert state.metrics["gn_rolling_sharpe_252_pos_share"] > 0.9

    deterministic_returns = pd.DataFrame(
        {
            "G1": np.full(185, 0.0000),
            "G2": np.full(185, 0.0002),
            "G3": np.full(185, 0.0004),
            "G4": np.full(185, 0.0030),
            "G5": np.full(185, 0.0032),
        },
        index=pd.date_range("2023-01-02", periods=185, freq="B"),
    )
    deterministic_returns.iloc[60:120, 2] = 0.0040
    daily_detail, daily_metrics = top_quantile_performance(
        deterministic_returns,
        n_quantiles=5,
    )
    assert daily_detail["top_group_is_daily_best"].iloc[0] == 1.0
    assert daily_detail["top_group_is_daily_best"].iloc[60] == 0.0
    assert np.isclose(
        daily_metrics["top_group_outperformance_ratio"],
        125.0 / 185.0,
    )
    assert np.isclose(
        daily_metrics["top_group_annualized_return"],
        (1.0 + daily_metrics["top_group_final_cumulative"])
        ** (252.0 / len(daily_detail))
        - 1.0,
    )
    leadership_detail, missed_detail, leadership_metrics = (
        non_overlapping_top_quantile_leadership(
            deterministic_returns,
            n_quantiles=5,
            window=60,
        )
    )
    assert len(leadership_detail) == 3
    assert len(missed_detail) == 1
    assert leadership_metrics["top_group_60d_window_count"] == 3
    assert leadership_metrics["top_group_60d_best_window_count"] == 2
    assert "top_group_60d_missed_window_count" not in leadership_metrics
    assert np.isclose(leadership_metrics["top_group_60d_best_window_ratio"], 2.0 / 3.0)
    assert leadership_detail["top_group_is_best"].tolist() == [1.0, 0.0, 1.0]
    assert leadership_detail.iloc[1]["best_group_number"] == 3.0
    assert missed_detail.index[0] == deterministic_returns.index[60]

    second_highest_beats_top = pd.DataFrame(
        {
            "G1": np.full(60, 0.0000),
            "G2": np.full(60, 0.0001),
            "G3": np.full(60, 0.0002),
            "G4": np.full(60, 0.0040),
            "G5": np.full(60, 0.0030),
        },
        index=pd.date_range("2024-01-02", periods=60, freq="B"),
    )
    second_context = EvaluationContext(
        factor_name="top_lower_check",
        artifact_name="top_lower_check",
        factor=second_highest_beats_top,
        close=second_highest_beats_top,
        forward_return=second_highest_beats_top * 0,
        horizon=1,
        n_quantiles=5,
        output_dir=Path("."),
    )
    second_state = EvaluationState(second_context)
    second_state.add_detail("group_returns", second_highest_beats_top)
    evaluate_top_quantile_performance(second_state)
    manual_detail, manual_metrics = top_quantile_performance(
        second_highest_beats_top,
        n_quantiles=5,
    )
    assert manual_detail["top_group_is_daily_best"].eq(0.0).all()
    assert manual_metrics["top_group_outperformance_ratio"] == 0.0
    assert not second_state.metrics["top_group_above_every_lower_group"]
    assert np.isclose(second_state.metrics["best_lower_group_mean_return"], 0.004)


def test_yearly_fitness_uses_annualized_partial_years_turnover_and_drawdown_penalty() -> None:
    days = pd.DatetimeIndex(
        [
            "2022-12-29",
            "2022-12-30",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
        ]
    )
    group_returns = pd.DataFrame(
        {
            "G1": 0.0,
            "G2": 0.0,
            "G3": [0.0008, -0.0002, 0.0005, 0.0011, -0.0004],
        },
        index=days,
    )
    turnover = pd.DataFrame(
        {
            "G3_one_way_turnover": [0.05, 0.05, 0.20, 0.25, 0.30],
        },
        index=days,
    )
    detail, metrics = calculate_yearly_fitness(
        group_returns,
        turnover,
        n_quantiles=3,
    )
    assert detail.index.tolist() == [2022, 2023]
    assert detail["trading_days"].tolist() == [2.0, 3.0]

    first_year_returns = group_returns.loc["2022", "G3"]
    first_year_cumulative = float((1 + first_year_returns).prod() - 1)
    first_year_annualized = float(
        (1 + first_year_cumulative) ** (252 / len(first_year_returns)) - 1
    )
    first_year_sharpe = float(
        first_year_returns.mean() / first_year_returns.std(ddof=1) * np.sqrt(252)
    )
    first_year_net_value = (1 + first_year_returns).cumprod()
    first_year_drawdown = float(
        (1 - first_year_net_value / first_year_net_value.cummax().clip(lower=1.0)).max()
    )
    first_year_turnover = float(turnover.loc["2022", "G3_one_way_turnover"].mean())
    first_year_lambda = float(1 / (1 - abs(first_year_drawdown)) ** 2)
    first_year_radicand = float(abs(first_year_annualized) / max(first_year_turnover, 0.125))
    first_year_fitness = float(
        first_year_sharpe * np.sqrt(first_year_radicand)
        - first_year_lambda * first_year_drawdown
    )
    assert np.isclose(detail.loc[2022, "cumulative_net_return"], first_year_cumulative)
    assert np.isclose(detail.loc[2022, "annualized_net_return"], first_year_annualized)
    assert np.isclose(detail.loc[2022, "annualized_sharpe"], first_year_sharpe)
    assert np.isclose(detail.loc[2022, "mean_daily_one_way_turnover"], first_year_turnover)
    assert np.isclose(detail.loc[2022, "annual_max_drawdown"], first_year_drawdown)
    assert np.isclose(detail.loc[2022, "drawdown_penalty_lambda"], first_year_lambda)
    assert np.isclose(detail.loc[2022, "fitness_radicand"], first_year_radicand)
    assert np.isclose(detail.loc[2022, "fitness"], first_year_fitness)

    second_year_returns = group_returns.loc["2023", "G3"]
    second_year_cumulative = float((1 + second_year_returns).prod() - 1)
    second_year_annualized = float(
        (1 + second_year_cumulative) ** (252 / len(second_year_returns)) - 1
    )
    second_year_sharpe = float(
        second_year_returns.mean() / second_year_returns.std(ddof=1) * np.sqrt(252)
    )
    second_year_net_value = (1 + second_year_returns).cumprod()
    second_year_drawdown = float(
        (1 - second_year_net_value / second_year_net_value.cummax().clip(lower=1.0)).max()
    )
    second_year_turnover = float(turnover.loc["2023", "G3_one_way_turnover"].mean())
    second_year_lambda = float(1 / (1 - abs(second_year_drawdown)) ** 2)
    second_year_radicand = float(abs(second_year_annualized) / max(second_year_turnover, 0.125))
    second_year_fitness = float(
        second_year_sharpe * np.sqrt(second_year_radicand)
        - second_year_lambda * second_year_drawdown
    )
    assert np.isclose(detail.loc[2023, "annualized_net_return"], second_year_annualized)
    assert np.isclose(detail.loc[2023, "annualized_sharpe"], second_year_sharpe)
    assert np.isclose(detail.loc[2023, "mean_daily_one_way_turnover"], second_year_turnover)
    assert np.isclose(detail.loc[2023, "annual_max_drawdown"], second_year_drawdown)
    assert np.isclose(detail.loc[2023, "drawdown_penalty_lambda"], second_year_lambda)
    assert np.isclose(detail.loc[2023, "fitness_radicand"], second_year_radicand)
    assert np.isclose(detail.loc[2023, "fitness"], second_year_fitness)
    assert np.isclose(metrics["fitness"], (first_year_fitness + second_year_fitness) / 2)
    assert metrics["fitness_year_count"] == 2
    assert metrics["fitness_valid_year_count"] == 2

    context = EvaluationContext(
        factor_name="yearly_fitness",
        artifact_name="yearly_fitness",
        factor=group_returns,
        close=group_returns,
        forward_return=group_returns,
        horizon=1,
        n_quantiles=3,
        output_dir=Path("."),
    )
    state = EvaluationState(context)
    state.add_detail("group_returns", group_returns)
    state.add_detail("quantile_turnover", turnover)
    evaluate_fitness(state)
    assert state.details["yearly_fitness"].equals(detail)
    assert state.metrics == metrics
    assert evaluation_method_names(resolve_evaluation_methods("fitness")) == (
        "quantile_net_returns",
        "fitness",
    )

    underwater_returns = pd.DataFrame(
        {"G3": [-0.2, 0.25]},
        index=pd.date_range("2024-01-02", periods=2, freq="B"),
    )
    underwater_turnover = pd.DataFrame(
        {"G3_one_way_turnover": [0.1, 0.1]},
        index=underwater_returns.index,
    )
    underwater_detail, underwater_metrics = calculate_yearly_fitness(
        underwater_returns,
        underwater_turnover,
        n_quantiles=3,
    )
    underwater_drawdown = 0.2
    underwater_lambda = 1 / (1 - underwater_drawdown) ** 2
    assert np.isclose(underwater_detail.loc[2024, "fitness_radicand"], 0.0)
    assert np.isclose(underwater_detail.loc[2024, "fitness"], -underwater_lambda * underwater_drawdown)
    assert np.isclose(underwater_metrics["fitness"], -underwater_lambda * underwater_drawdown)
    assert underwater_metrics["fitness_valid_year_count"] == 1


def test_tradability_filter() -> None:
    rng = np.random.default_rng(66)
    days = pd.date_range("2024-01-02", periods=30, freq="B")
    codes = [
        "000001",
        "000002",
        "300001",
        "688001",
        "600001",
        "002001",
    ]
    close = pd.DataFrame(
        20
        * np.cumprod(
            1 + rng.normal(0.0005, 0.01, size=(len(days), len(codes))),
            axis=0,
        ),
        index=days,
        columns=codes,
    )
    open_prices = close.shift(1) * 1.001
    open_prices.iloc[0] = close.iloc[0] * 1.001
    limit_ratio = pd.DataFrame(
        [price_limit_ratio_for_code(code) for code in codes],
        index=codes,
    ).T
    limit_ratio = pd.concat([limit_ratio] * len(days), ignore_index=True)
    limit_ratio.index = days
    st_status = pd.DataFrame(False, index=days, columns=codes)
    delisting_status = pd.DataFrame(False, index=days, columns=codes)
    traded_amount = pd.DataFrame(1e8, index=days, columns=codes)
    factor = close.rank(axis=1, pct=True)

    # Entry-day open limit boundaries are checked against the factor day close.
    up_day, down_day, st_day, delisting_day, no_trade_day = 10, 20, 25, 15, 28
    up_code, down_code, st_code, delisting_code, no_trade_code = (
        "000001",
        "300001",
        "688001",
        "002001",
        "600001",
    )
    open_prices.loc[days[up_day], up_code] = (
        close.loc[days[up_day - 1], up_code] * 1.10
    )
    open_prices.loc[days[down_day], down_code] = (
        close.loc[days[down_day - 1], down_code] * 0.80
    )
    # This would have been a wide gap under a hard-coded 10% rule, but it is
    # still inside the 20% ChiNext limit and must remain tradable.
    open_prices.loc[days[up_day], down_code] = (
        close.loc[days[up_day - 1], down_code] * 1.15
    )
    st_status.loc[days[st_day], st_code] = True
    delisting_status.loc[days[delisting_day], delisting_code] = True
    traded_amount.loc[days[no_trade_day], no_trade_code] = 0.0

    masked, detail, summary = mask_untradeable_entries(
        factor,
        open_prices,
        close,
        limit_ratio,
        st_status,
        delisting_status,
        traded_amount,
    )
    # factor[t] is masked when day t+1 is untradeable at entry.
    assert np.isnan(masked.iloc[up_day - 1][up_code])
    assert np.isnan(masked.iloc[down_day - 1][down_code])
    assert np.isnan(masked.iloc[st_day - 1][st_code])
    assert np.isnan(masked.iloc[st_day][st_code])
    assert np.isnan(masked.iloc[delisting_day - 1][delisting_code])
    assert np.isnan(masked.iloc[delisting_day][delisting_code])
    assert np.isnan(masked.iloc[no_trade_day - 1][no_trade_code])
    # Once the formal ST state has been withdrawn, later signal and entry days
    # are eligible again.
    assert not np.isnan(masked.iloc[st_day + 1][st_code])
    assert not np.isnan(masked.iloc[up_day - 1][down_code])
    assert summary["tradability_masked_obs"] == 7
    assert summary["tradability_masked_days"] == 7
    assert summary["tradability_masked_limit_up_obs"] == 1
    assert summary["tradability_masked_limit_down_obs"] == 1
    assert summary["tradability_masked_st_obs"] == 2
    assert summary["tradability_masked_delisting_obs"] == 2
    assert summary["tradability_masked_signal_st_obs"] == 1
    assert summary["tradability_masked_entry_st_obs"] == 1
    assert summary["tradability_masked_signal_delisting_obs"] == 1
    assert summary["tradability_masked_entry_delisting_obs"] == 1
    assert summary["tradability_masked_entry_no_trade_obs"] == 1
    # Everything else is untouched.
    untouched = masked.drop(
        columns=[up_code, down_code, st_code, delisting_code, no_trade_code]
    )
    expected = factor.drop(
        columns=[up_code, down_code, st_code, delisting_code, no_trade_code]
    )
    assert untouched.equals(expected)
    assert detail["n_masked"].sum() == 7
    assert detail["n_masked_entry_no_trade"].sum() == 1

    # A bounded train/test factor window must still inspect the market row just
    # after its final retained signal.  The old implementation first truncated
    # market data to factor.index and silently skipped this last t+1 entry.
    bounded_factor = factor.iloc[up_day - 3 : up_day]
    bounded_masked, _, bounded_summary = mask_untradeable_entries(
        bounded_factor,
        open_prices,
        close,
        limit_ratio,
        st_status,
        delisting_status,
        traded_amount,
    )
    assert np.isnan(bounded_masked.iloc[-1][up_code])
    assert bounded_summary["tradability_masked_limit_up_obs"] == 1

    # Signal-day policy checks still apply on a timeline tail without t+1.
    tail_st = st_status.copy()
    tail_st.loc[days[-1], up_code] = True
    tail_masked, tail_detail, tail_summary = mask_untradeable_entries(
        factor.iloc[-1:],
        open_prices,
        close,
        limit_ratio,
        tail_st,
        delisting_status,
        traded_amount,
    )
    assert np.isnan(tail_masked.iloc[0][up_code])
    assert tail_summary["tradability_masked_signal_st_obs"] == 1
    assert tail_summary["tradability_masked_entry_st_obs"] == 0
    assert tail_detail.iloc[0]["n_masked_entry_no_trade"] == 0

    context = EvaluationContext(
        factor_name="tradability",
        artifact_name="tradability",
        factor=factor,
        close=close,
        forward_return=factor * 0,
        horizon=1,
        n_quantiles=5,
        output_dir=Path("."),
        market_data={
            "c": close,
            "o": open_prices,
            "limit": limit_ratio,
            "st": st_status,
            "delisting": delisting_status,
            "amt": traded_amount,
        },
    )
    state = run_evaluation_methods(context, (evaluate_tradability_filter,))
    assert state.metrics["tradability_masked_obs"] == 7
    assert np.isnan(state.factor.iloc[up_day - 1][up_code])
    assert "tradability_filter" in state.details
    resolved = resolve_evaluation_methods("tradability_filter,quantile_returns")
    assert evaluation_method_names(resolved) == (
        "tradability_filter",
        "quantile_returns",
    )

    inferred = infer_price_limit_ratio_frame(close)
    assert inferred["000001"].dropna().eq(0.10).all()
    assert inferred["300001"].dropna().eq(0.20).all()
    assert inferred["688001"].dropna().eq(0.20).all()

    st_long = pd.DataFrame(
        {
            "day": [days[3].strftime("%Y-%m-%d")],
            "code": [st_code],
            "是否st": [True],
        }
    )
    st_wide = normalize_st_status_frame(st_long, close)
    assert bool(st_wide.loc[days[3], st_code])
    assert not bool(st_wide.loc[days[3], up_code])

    delisting_long = pd.DataFrame(
        {
            "day": [days[4].strftime("%Y-%m-%d")],
            "code": [delisting_code],
            "is_delisting_period": [True],
        }
    )
    delisting_wide = normalize_delisting_status_frame(delisting_long, close)
    assert bool(delisting_wide.loc[days[4], delisting_code])
    assert not bool(delisting_wide.loc[days[4], up_code])

    for normalizer, column in (
        (normalize_st_status_frame, "是否st"),
        (normalize_delisting_status_frame, "is_delisting_period"),
    ):
        invalid = pd.DataFrame(
            {"day": [days[4]], "code": [up_code], column: ["False"]}
        )
        try:
            normalizer(invalid, close)
        except ValueError:
            pass
        else:
            raise AssertionError("string False must not pass a true-only status loader")


def test_incremental_st_status_reconciliation() -> None:
    historical = pd.DataFrame(
        {
            "day": pd.to_datetime(["2026-06-26", "2026-06-26"]),
            "code": ["600365", "600999"],
            "是否st": [True, True],
        }
    )
    incoming = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-06-29",
                    "2026-06-29",
                    "2026-06-29",
                    "2026-07-06",
                    "2026-07-06",
                    "2026-07-06",
                ]
            ),
            "code": [
                "600365",
                "600999",
                "300087",
                "600365",
                "600999",
                "300087",
            ],
            "pre_close_raw": [2.39, 10.0, 10.0, 2.57, 10.0, 10.0],
            "limit_up_price": [2.51, 11.0, 12.0, 2.83, 11.0, 12.0],
            "limit_down_price": [2.27, 9.0, 8.0, 2.31, 9.0, 8.0],
            "st_effective_date": [
                "2012-08-16",
                "2010-01-01",
                "2026-06-29",
                "2012-08-16",
                "2010-01-01",
                "2026-06-29",
            ],
            "st_party_state": [1, 2, 2, 1, 2, 2],
        }
    )
    reconciled, diagnostics = reconcile_incremental_st_status(
        historical,
        incoming,
    )
    tail = reconciled[reconciled["day"] >= pd.Timestamp("2026-06-29")]
    wide = tail.pivot(index="day", columns="code", values="是否st")

    # ST通葡's stale 2012 removal event must not override the trustworthy
    # boundary state; after 2026-07-06 its 10% limit is no longer evidence of
    # removal, so the state continues.
    assert bool(wide.at[pd.Timestamp("2026-06-29"), "600365"])
    assert bool(wide.at[pd.Timestamp("2026-07-06"), "600365"])
    # A main-board 10% limit before the rule change is direct evidence that a
    # previously carried ST flag should be cleared.
    assert not bool(wide.at[pd.Timestamp("2026-06-29"), "600999"])
    assert not bool(wide.at[pd.Timestamp("2026-07-06"), "600999"])
    # A genuinely new event inside the increment is applied and carried.
    assert bool(wide.at[pd.Timestamp("2026-06-29"), "300087"])
    assert bool(wide.at[pd.Timestamp("2026-07-06"), "300087"])
    assert diagnostics["boundary_day"] == "2026-06-26"
    assert diagnostics["explicit_event_observations"] == 1


def test_holding_audit() -> None:
    """The persisted G_N rows must be the same rows used for net returns."""

    days = pd.date_range("2024-01-02", periods=12, freq="B")
    codes = [f"{number:06d}" for number in range(1, 11)]
    daily_growth = np.linspace(0.0005, 0.005, len(codes))
    open_prices = pd.DataFrame(
        100 * np.exp(np.arange(len(days))[:, None] * daily_growth[None, :]),
        index=days,
        columns=codes,
    )
    close = open_prices.copy()
    factor = pd.DataFrame(
        np.tile(np.arange(1, len(codes) + 1, dtype=float), (len(days), 1)),
        index=days,
        columns=codes,
    )
    forward_return = open_prices.shift(-2) / open_prices.shift(-1) - 1
    limit_ratio = pd.DataFrame(0.10, index=days, columns=codes)
    st_status = pd.DataFrame(False, index=days, columns=codes)
    delisting_status = pd.DataFrame(False, index=days, columns=codes)
    traded_amount = pd.DataFrame(1e8, index=days, columns=codes)
    # The highest-ranked code is ST at the first signal day's next-open entry,
    # so the audit proves it uses the factor after tradability masking.
    st_status.loc[days[1], "000010"] = True
    # A different high-ranked name is in a formal delisting period on the
    # signal day and must also disappear from the first basket.
    delisting_status.loc[days[0], "000009"] = True
    cap = pd.DataFrame(
        np.tile(np.arange(1, len(codes) + 1) * 1e8, (len(days), 1)),
        index=days,
        columns=codes,
    )
    industry = pd.DataFrame(
        np.tile(["01031701"] * 5 + ["01031702"] * 5, (len(days), 1)),
        index=days,
        columns=codes,
    )

    with tempfile.TemporaryDirectory() as temporary:
        context = EvaluationContext(
            factor_name="holding_audit",
            artifact_name="holding_audit",
            factor=factor,
            close=close,
            forward_return=forward_return,
            horizon=1,
            n_quantiles=5,
            output_dir=Path(temporary),
            market_data={
                "c": close,
                "o": open_prices,
                "cap": cap,
                "industry": industry,
                "limit": limit_ratio,
                "st": st_status,
                "delisting": delisting_status,
                "amt": traded_amount,
            },
        )
        methods = resolve_evaluation_methods("holding_audit")
        assert evaluation_method_names(methods) == (
            "industry_market_cap_neutralize",
            "tradability_filter",
            "quantile_net_returns",
            "holding_audit",
        )
        state = run_evaluation_methods(context, methods)
        artifact_path = state.artifacts["holding_audit"]
        assert artifact_path.is_file()
        holdings = pd.read_parquet(artifact_path)

    assert tuple(holdings.columns) == HOLDING_AUDIT_COLUMNS
    assert set(holdings["group"]) == {"G5"}
    assert not holdings["entry_is_st"].any()
    assert not holdings["signal_is_st"].any()
    assert not holdings["signal_is_delisting_period"].any()
    assert not holdings["entry_is_delisting_period"].any()
    # The ST high-rank name on the first entry date cannot remain in the G5 basket.
    first_day = days[0].date().isoformat()
    assert "000010" not in set(
        holdings.loc[holdings["signal_day"] == first_day, "security_code"]
    )
    assert "000009" not in set(
        holdings.loc[holdings["signal_day"] == first_day, "security_code"]
    )
    summary = state.details[HOLDING_AUDIT_SUMMARY]
    group_returns = state.details["group_returns"]
    transaction_costs = state.details["quantile_transaction_cost"]
    assert summary.index.equals(group_returns.index)
    assert summary.index[-1] == days[-3]
    assert len(summary) == len(days) - 2
    assert np.allclose(summary["top_group_net_return"], group_returns["G5"])
    assert np.allclose(
        summary["top_group_gross_return"] - summary["top_group_transaction_cost"],
        group_returns["G5"],
    )
    assert np.allclose(summary["top_group_transaction_cost"], transaction_costs["G5"])
    assert int(summary["top_group_position_count"].sum()) == len(holdings)
    assert state.metrics["holding_audit_signal_day_count"] == len(summary)
    assert state.metrics["holding_audit_position_count"] == len(holdings)

    no_dependency_context = EvaluationContext(
        factor_name="holding_audit_dependency",
        artifact_name="holding_audit_dependency",
        factor=factor,
        close=close,
        forward_return=forward_return,
        horizon=1,
        n_quantiles=5,
        output_dir=Path("."),
        market_data={
            "c": close,
            "o": open_prices,
            "cap": cap,
            "industry": industry,
            "limit": limit_ratio,
            "st": st_status,
            "delisting": delisting_status,
            "amt": traded_amount,
        },
    )
    try:
        run_evaluation_methods(no_dependency_context, (evaluate_holding_audit,))
    except ValueError as exc:
        assert "industry_market_cap_neutralize" in str(exc)
    else:  # pragma: no cover - documents that declared dependencies stay enforced.
        raise AssertionError("holding_audit must require prior portfolio construction")


def test_engine_integration() -> None:
    rng = np.random.default_rng(55)
    days = pd.date_range("2024-01-02", periods=45, freq="B")
    codes = [f"{number:06d}" for number in range(20)]
    returns = rng.normal(0.0005, 0.01, size=(len(days), len(codes)))
    close = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        index=days,
        columns=codes,
    )
    open_prices = close * pd.DataFrame(
        1 + rng.normal(0, 0.002, size=close.shape),
        index=days,
        columns=codes,
    )
    # A stable singularity must not become a false future-data failure merely
    # because the engine sanitizes the original +/-inf to NaN.
    close.iloc[:, 0] = 0.0
    cap = pd.DataFrame(
        rng.lognormal(24, 1.0, size=close.shape),
        index=days,
        columns=codes,
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        close.to_parquet(root / "close_df.pq")
        open_prices.to_parquet(root / "open_df.pq")
        cap.to_parquet(root / "market_cap_df.pq")

        neutralized = evaluate_factor_expression(
            factor_name="neutralized_pipeline",
            expression="rank_cs(c)",
            data_dir=root,
            output_dir=root / "neutralized",
            evaluation_methods=resolve_evaluation_methods(
                "market_cap_neutralize,rank_icir"
            ),
        )
        assert neutralized["evaluation_methods"] == [
            "market_cap_neutralize",
            "rank_ic",
            "rank_icir",
        ]
        assert "market_cap_neutralization" in neutralized["detail_paths"]

        perturbation = evaluate_factor_expression(
            factor_name="perturbation_pipeline",
            expression="ts_mean(c, 3)",
            data_dir=root,
            output_dir=root / "perturbation",
            evaluation_methods=resolve_evaluation_methods(
                "future_data_perturbation"
            ),
        )
        assert perturbation["metrics"]["future_perturbation_changed_values"] == 0

        stable_singularity = evaluate_factor_expression(
            factor_name="stable_singularity",
            expression="1.0 / c",
            data_dir=root,
            output_dir=root / "stable_singularity",
            evaluation_methods=resolve_evaluation_methods(
                "future_data_perturbation"
            ),
        )
        assert stable_singularity["metrics"]["future_perturbation_changed_values"] == 0

        prefix = evaluate_factor_expression(
            factor_name="prefix_pipeline",
            expression="ts_mean(c, 3)",
            data_dir=root,
            output_dir=root / "prefix",
            evaluation_methods=resolve_evaluation_methods(
                "prefix_truncation_consistency"
            ),
        )
        assert prefix["metrics"]["prefix_truncation_passed"]
        assert "prefix_truncation_consistency" in prefix["detail_paths"]
        assert prefix["evaluation_methods"] == ["prefix_truncation_consistency"]

        rolling_sharpe_result = evaluate_factor_expression(
            factor_name="rolling_sharpe_pipeline",
            expression="rank_cs(c)",
            data_dir=root,
            output_dir=root / "rolling_sharpe",
            evaluation_methods=resolve_evaluation_methods("rolling_sharpe"),
        )
        sharpe_plot = rolling_sharpe_result["artifact_paths"][
            "rolling_sharpe_60_plot"
        ]
        assert Path(sharpe_plot).is_file() and Path(sharpe_plot).stat().st_size > 0
        assert "rolling_sharpe_60" in rolling_sharpe_result["detail_paths"]
        assert '"rolling_sharpe_60_plot":"plots/' in rolling_sharpe_result[
            "metrics"
        ]["evaluation_artifacts"]

        fitness_result = evaluate_factor_expression(
            factor_name="fitness_pipeline",
            expression="rank_cs(c)",
            data_dir=root,
            output_dir=root / "fitness",
            evaluation_methods=resolve_evaluation_methods("fitness"),
        )
        assert fitness_result["evaluation_methods"] == [
            "quantile_net_returns",
            "fitness",
        ]
        fitness_path = Path(fitness_result["detail_paths"]["yearly_fitness"])
        assert fitness_path.is_file()
        yearly = pd.read_csv(fitness_path, encoding="utf-8-sig", index_col=0)
        assert {
            "trading_days",
            "annualized_net_return",
            "annualized_sharpe",
            "mean_daily_one_way_turnover",
            "annual_max_drawdown",
            "drawdown_penalty_lambda",
            "fitness_radicand",
            "fitness",
        } <= set(yearly)
        assert "fitness" in fitness_result["metrics"]

        # A bounded run keeps its final signal only when its next-open entry
        # and exit price are observable.  That final t+1 must still use the
        # *full* close/ST timeline for the entry filter; otherwise an ST name
        # can leak solely on the final audit date.
        audit_close = pd.DataFrame(
            np.tile(np.arange(1, len(codes) + 1, dtype=float), (len(days), 1)),
            index=days,
            columns=codes,
        )
        audit_open = audit_close.copy()
        audit_close.to_parquet(root / "audit_close_df.pq")
        audit_open.to_parquet(root / "audit_open_df.pq")
        pd.DataFrame(1e8, index=days, columns=codes).to_parquet(
            root / "audit_market_cap_df.pq"
        )
        pd.DataFrame(0.10, index=days, columns=codes).to_parquet(
            root / "audit_limit_ratio_df.pq"
        )
        pd.DataFrame(1e8, index=days, columns=codes).to_parquet(
            root / "audit_amount_df.pq"
        )
        pd.DataFrame(
            {
                "trade_date": np.repeat(days, len(codes)),
                "security_code": codes * len(days),
                "industry_l1_code": np.tile(
                    ["01031701"] * 10 + ["01031702"] * 10,
                    len(days),
                ),
            }
        ).to_parquet(root / "audit_industry.parquet")
        # signal_end=days[25] keeps days[23] as the final signal for H=1.
        # Its next-open entry is days[24], where the otherwise top-residual
        # code must be eliminated rather than leaking into the audit.
        pd.DataFrame(
            {
                "day": [days[24]],
                "code": ["000020"],
                "是否st": [True],
            }
        ).to_parquet(root / "audit_st_status_df.pq")
        pd.DataFrame(
            {
                "day": [days[23]],
                "code": ["000019"],
                "is_delisting_period": [True],
            }
        ).to_parquet(root / "audit_delisting_status_df.pq")
        audit_result = evaluate_factor_expression(
            factor_name="bounded_holding_audit",
            expression="rank_cs(c)",
            data_dir=root,
            output_dir=root / "bounded_holding_audit",
            file_names={
                "c": "audit_close_df.pq",
                "o": "audit_open_df.pq",
                "cap": "audit_market_cap_df.pq",
                "limit": "audit_limit_ratio_df.pq",
                "st": "audit_st_status_df.pq",
                "delisting": "audit_delisting_status_df.pq",
                "amt": "audit_amount_df.pq",
                "industry": "audit_industry.parquet",
            },
            evaluation_methods=resolve_evaluation_methods("holding_audit"),
            signal_start=days[10],
            signal_end=days[25],
        )
        assert audit_result["evaluation_methods"] == [
            "industry_market_cap_neutralize",
            "tradability_filter",
            "quantile_net_returns",
            "holding_audit",
        ]
        assert '"holding_audit":"details/' in audit_result["metrics"][
            "evaluation_artifacts"
        ]
        audited_holdings = pd.read_parquet(audit_result["artifact_paths"]["holding_audit"])
        assert not audited_holdings["entry_is_st"].any()
        assert not audited_holdings["signal_is_st"].any()
        assert not audited_holdings["signal_is_delisting_period"].any()
        assert not audited_holdings["entry_is_delisting_period"].any()
        final_signal = days[23].date().isoformat()
        assert "000020" not in set(
            audited_holdings.loc[
                audited_holdings["signal_day"] == final_signal,
                "security_code",
            ]
        )
        assert "000019" not in set(
            audited_holdings.loc[
                audited_holdings["signal_day"] == final_signal,
                "security_code",
            ]
        )

        # A full-history run keeps the source factor's natural unlabelled tail,
        # while quantile_net_returns omits those dates.  The audit must follow
        # the emitted portfolio dates instead of demanding nonexistent future
        # entry/exit opens for the final horizon+1 factor rows.
        full_audit_result = evaluate_factor_expression(
            factor_name="full_history_holding_audit",
            expression="rank_cs(c)",
            data_dir=root,
            output_dir=root / "full_history_holding_audit",
            file_names={
                "c": "audit_close_df.pq",
                "o": "audit_open_df.pq",
                "cap": "audit_market_cap_df.pq",
                "limit": "audit_limit_ratio_df.pq",
                "st": "audit_st_status_df.pq",
                "delisting": "audit_delisting_status_df.pq",
                "amt": "audit_amount_df.pq",
                "industry": "audit_industry.parquet",
            },
            evaluation_methods=resolve_evaluation_methods("holding_audit"),
        )
        full_audited_holdings = pd.read_parquet(
            full_audit_result["artifact_paths"]["holding_audit"]
        )
        assert full_audited_holdings["signal_day"].max() == days[-3].date().isoformat()


def main() -> None:
    assert evaluation_method_names(DEFAULT_EVALUATION_METHODS) == (
        "rank_ic",
        "rank_icir",
        "quantile_returns",
        "quantile_cumulative",
        "quantile_plot",
    )
    test_future_perturbation()
    test_market_cycle_backgrounds()
    test_prefix_truncation_consistency()
    test_newey_west()
    test_ic_horizon_decay_uses_one_common_signal_sample()
    test_ic_peak_decay_is_60_day_rolling_mean_ic()
    test_ic_trend_filter()
    test_market_cap_neutralization()
    test_top_quantiles_and_rolling_evaluators()
    test_yearly_fitness_uses_annualized_partial_years_turnover_and_drawdown_penalty()
    test_tradability_filter()
    test_incremental_st_status_reconciliation()
    test_holding_audit()
    test_engine_integration()
    print("extended evaluation modules passed")


if __name__ == "__main__":
    main()
