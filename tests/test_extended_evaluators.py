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
    compare_ic_trend_filters,
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
from evaluators.rolling_drawdown import (
    evaluate_rolling_drawdown as direct_evaluate_rolling_drawdown,
)
from evaluators.rolling_sharpe import (
    evaluate_rolling_sharpe as direct_evaluate_rolling_sharpe,
)
from transforms import neutralize_factor_by_market_cap
from transforms.price_limits import (
    infer_price_limit_ratio_frame,
    normalize_st_status_frame,
    price_limit_ratio_for_code,
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
    factor = close.rank(axis=1, pct=True)

    # Entry-day open limit boundaries are checked against the factor day close.
    up_day, down_day, st_day = 10, 20, 25
    up_code, down_code, st_code = "000001", "300001", "688001"
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

    masked, detail, summary = mask_untradeable_entries(
        factor,
        open_prices,
        close,
        limit_ratio,
        st_status,
    )
    # factor[t] is masked when day t+1 is untradeable at entry.
    assert np.isnan(masked.iloc[up_day - 1][up_code])
    assert np.isnan(masked.iloc[down_day - 1][down_code])
    assert np.isnan(masked.iloc[st_day - 1][st_code])
    assert not np.isnan(masked.iloc[up_day - 1][down_code])
    assert summary["tradability_masked_obs"] == 3
    assert summary["tradability_masked_days"] == 3
    assert summary["tradability_masked_limit_up_obs"] == 1
    assert summary["tradability_masked_limit_down_obs"] == 1
    assert summary["tradability_masked_st_obs"] == 1
    # Everything else is untouched.
    untouched = masked.drop(columns=[up_code, down_code, st_code])
    expected = factor.drop(columns=[up_code, down_code, st_code])
    assert untouched.equals(expected)
    assert detail["n_masked"].sum() == 3

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
    )
    assert np.isnan(bounded_masked.iloc[-1][up_code])
    assert bounded_summary["tradability_masked_limit_up_obs"] == 1

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
        },
    )
    state = run_evaluation_methods(context, (evaluate_tradability_filter,))
    assert state.metrics["tradability_masked_obs"] == 3
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
    test_ic_peak_decay_is_60_day_rolling_mean_ic()
    test_ic_trend_filter()
    test_market_cap_neutralization()
    test_top_quantiles_and_rolling_evaluators()
    test_tradability_filter()
    test_engine_integration()
    print("extended evaluation modules passed")


if __name__ == "__main__":
    main()
