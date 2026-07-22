#!/usr/bin/env python3
"""Synthetic end-to-end smoke test for the factor evaluation artifact contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from engine import (
    _artifact_name,
    adv,
    assign_quantile,
    calc_group_returns,
    decay_linear,
    elementwise_max,
    evaluate_factor_expression,
    evaluate_expression,
    evaluate_wide,
    get_ic_info,
    scale_cs,
    winsorize_cs,
    zscore_cs,
    ts_argmax,
    ts_corr,
    ts_sum,
)
from evaluators import resolve_evaluation_methods
from reporting.dashboard import validate_artifacts
from returns import (
    RETURN_DEFINITION,
    calculate_forward_open_return,
)


def main() -> None:
    assert _artifact_name("alpha101_001") == "alpha101_001"
    assert _artifact_name('bad/name:*?') == "bad_name___"

    rng = np.random.default_rng(7)
    days = pd.date_range("2024-01-01", periods=35, freq="B")
    codes = [f"{number:06d}" for number in range(30)]
    returns = rng.normal(0.0005, 0.02, size=(len(days), len(codes)))
    close = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0), index=days, columns=codes
    )
    open_prices = close * pd.DataFrame(
        1 + rng.normal(0, 0.003, size=close.shape),
        index=days,
        columns=codes,
    )
    high = close * (1 + rng.uniform(0, 0.02, size=close.shape))
    low = close * (1 - rng.uniform(0, 0.02, size=close.shape))
    amount = pd.DataFrame(
        rng.lognormal(12, 0.7, size=close.shape), index=days, columns=codes
    )
    volume = pd.DataFrame(
        rng.integers(10_000, 1_000_000, size=close.shape),
        index=days,
        columns=codes,
    )
    vwap = (high + low) / 2

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        close.to_parquet(root / "close_df.pq")
        open_prices.to_parquet(root / "open_df.pq")
        high.to_parquet(root / "high_df.pq")
        low.to_parquet(root / "low_df.pq")
        amount.to_parquet(root / "amount_df.pq")
        volume.to_parquet(root / "volume_df.pq")
        vwap.to_parquet(root / "vwap_proxy_df.pq")
        expression = (
            "ts_mean(np.abs((h-l)/(h+l+1e-6)), 5) / "
            "(ts_std(amt,5) + 1e-6)"
        )
        result = evaluate_factor_expression(
            factor_name="factor_test1",
            expression=expression,
            data_dir=root,
            output_dir=root / "results",
            n_quantiles=5,
        )
        required_paths = [
            "metrics_path",
            "metrics_history_path",
            "plot_path",
            "code_path",
            "code_history_path",
            "ic_path",
            "group_returns_path",
            "cumulative_returns_path",
        ]
        for key in required_paths:
            path = Path(result[key])
            assert path.is_file() and path.stat().st_size > 0, (key, path)
        assert result["metrics"]["ic_count"] > 0
        assert np.isfinite(result["metrics"]["ic_mean"])
        assert result["metrics"]["return_definition"] == RETURN_DEFINITION
        assert result["evaluation_methods"] == [
            "rank_ic",
            "rank_icir",
            "quantile_returns",
            "quantile_cumulative",
            "quantile_plot",
        ]
        assert set(json.loads(result["metrics"]["evaluation_details"])) == {
            "ic",
            "group_returns",
            "cumulative_returns",
        }
        assert set(json.loads(result["metrics"]["evaluation_artifacts"])) == {
            "plot"
        }

        rank_only = evaluate_factor_expression(
            factor_name="factor_rank_only",
            expression=expression,
            data_dir=root,
            output_dir=root / "results",
            n_quantiles=5,
            evaluation_methods=resolve_evaluation_methods("rank_icir"),
        )
        assert set(rank_only["detail_paths"]) == {"ic"}
        assert set(
            json.loads(rank_only["metrics"]["evaluation_details"])
        ) == {"ic"}
        assert json.loads(rank_only["metrics"]["evaluation_artifacts"]) == {}
        assert "plot_path" not in rank_only
        assert "group_returns_path" not in rank_only
        assert np.isfinite(rank_only["metrics"]["ic_mean"])
        assert "gn_final_cumulative" not in rank_only["metrics"]
        assert "EVALUATION_METHODS = ('rank_ic', 'rank_icir')" in Path(
            rank_only["code_path"]
        ).read_text(encoding="utf-8")
        assert (
            f"EXPECTED_RETURN_DEFINITION = {RETURN_DEFINITION!r}"
            in Path(rank_only["code_path"]).read_text(encoding="utf-8")
        )
        compile(
            Path(rank_only["code_path"]).read_text(encoding="utf-8"),
            str(rank_only["code_path"]),
            "exec",
        )
        history = pd.read_csv(rank_only["metrics_history_path"])
        assert len(history) == 2
        assert "gn_final_cumulative" in history
        assert pd.isna(
            history.loc[
                history["factor_name"] == "factor_rank_only",
                "gn_final_cumulative",
            ]
        ).all()
        validate_artifacts(
            root / "results",
            pd.read_csv(rank_only["metrics_path"]),
        )

        trend_filter = evaluate_factor_expression(
            factor_name="factor_ic_trend_filter",
            expression=expression,
            data_dir=root,
            output_dir=root / "results",
            n_quantiles=5,
            evaluation_methods=resolve_evaluation_methods("ic_trend_filter"),
        )
        assert trend_filter["evaluation_methods"] == ["rank_ic", "ic_trend_filter"]
        assert set(trend_filter["detail_paths"]) == {
            "ic",
            "ic_trend_filter",
            "ic_trend_filter_mean_10",
        }
        assert "ic_mean" not in trend_filter["metrics"]
        trend_filter_detail = pd.read_csv(
            trend_filter["detail_paths"]["ic_trend_filter"],
            index_col=0,
        )
        assert trend_filter_detail.columns.tolist() == [
            "daily_ic",
            "filtered_ic",
            "second_order_low_pass_ic",
            "fourier_low_pass_ic",
        ]
        assert len(trend_filter_detail) == trend_filter["metrics"]["ic_count"]
        assert np.isfinite(trend_filter["metrics"]["ic_trend_filter_final"])
        assert trend_filter["metrics"]["ic_trend_filter_second_order_period"] == 31
        assert trend_filter["metrics"]["ic_trend_filter_fourier_min_period"] == 31
        assert np.isfinite(
            trend_filter["metrics"]["ic_trend_filter_second_order_final"]
        )
        assert np.isfinite(
            trend_filter["metrics"]["ic_trend_filter_fourier_final"]
        )
        trend_filter_mean_detail = pd.read_csv(
            trend_filter["detail_paths"]["ic_trend_filter_mean_10"],
            index_col=0,
        )
        assert trend_filter_mean_detail.columns.tolist() == [
            "kalman_filtered_ic_mean",
            "second_order_low_pass_ic_mean",
            "fourier_low_pass_ic_mean",
        ]
        expected_mean_window_count = (
            trend_filter["metrics"]["ic_count"] - 10
        ) // 5 + 1
        assert len(trend_filter_mean_detail) == expected_mean_window_count
        assert (
            trend_filter["metrics"]["ic_trend_filter_mean_full_window_count"]
            == expected_mean_window_count
        )
        assert trend_filter["metrics"]["ic_trend_filter_mean_window_days"] == 10
        assert trend_filter["metrics"]["ic_trend_filter_mean_step_days"] == 5
        assert trend_filter["metrics"]["ic_trend_filter_mean_overlap_days"] == 5
        assert "EVALUATION_METHODS = ('rank_ic', 'ic_trend_filter')" in Path(
            trend_filter["code_path"]
        ).read_text(encoding="utf-8")

        # Bounded model samples use signal dates, but exclude the final entry
        # and holding days so every open-to-open label is contained inside the
        # requested training or testing period.
        split_result = evaluate_factor_expression(
            factor_name="factor_train_split",
            expression=expression,
            data_dir=root,
            output_dir=root / "results",
            horizon=1,
            n_quantiles=5,
            evaluation_methods=resolve_evaluation_methods("rank_icir"),
            signal_start=days[5].date().isoformat(),
            signal_end=days[22].date().isoformat(),
        )
        split_metrics = split_result["metrics"]
        assert split_metrics["signal_start"] == days[5].date().isoformat()
        assert split_metrics["signal_end"] == days[22].date().isoformat()
        assert split_metrics["sample_start_day"] == days[5].date().isoformat()
        # H=1 requires open[t+2], so the final two signal dates are excluded.
        assert split_metrics["sample_end_day"] == days[20].date().isoformat()
        split_ic = pd.read_csv(split_result["ic_path"], index_col=0)
        assert split_ic.index[0] == days[5].date().isoformat()
        assert split_ic.index[-1] == days[20].date().isoformat()
        generated_split_code = Path(split_result["code_path"]).read_text(encoding="utf-8")
        assert f"SIGNAL_START = {days[5].date().isoformat()!r}" in generated_split_code
        assert f"SIGNAL_END = {days[22].date().isoformat()!r}" in generated_split_code

        # Causality diagnostics still use the corresponding positions in the
        # full data history when a model is evaluated on a bounded split.
        split_causality = evaluate_factor_expression(
            factor_name="factor_split_causality",
            expression=expression,
            data_dir=root,
            output_dir=root / "results",
            horizon=1,
            n_quantiles=5,
            evaluation_methods=resolve_evaluation_methods("future_data_perturbation"),
            signal_start=days[5].date().isoformat(),
            signal_end=days[22].date().isoformat(),
        )
        assert split_causality["metrics"]["future_perturbation_changed_values"] == 0
        causality_detail = pd.read_csv(
            split_causality["detail_paths"]["future_data_perturbation"], index_col=0
        )
        assert causality_detail.index[0] >= days[5].date().isoformat()
        assert causality_detail.index[-1] <= days[20].date().isoformat()

        data = {"c": close, "h": high, "l": low, "amt": amount}
        factor = evaluate_expression(expression, data)
        future = calculate_forward_open_return(open_prices, horizon=1)
        long_factor = factor.stack().rename("score")
        long_return = future.stack().rename("f1")
        merged = pd.concat([long_factor, long_return], axis=1).dropna().reset_index()
        merged.columns = ["day", "code", "score", "f1"]
        ic_mean, ic_std, abs_ir = get_ic_info(merged, "score", "f1")
        grouped = assign_quantile(merged, "score", 5)
        expected_group_returns = calc_group_returns(
            grouped,
            n_quantiles=5,
        )
        actual_group_returns = pd.read_csv(
            result["group_returns_path"],
            index_col="day",
            parse_dates=True,
        )
        pd.testing.assert_frame_equal(
            actual_group_returns,
            expected_group_returns,
            check_freq=False,
        )
        wide_ic, wide_groups, wide_pairs = evaluate_wide(
            factor,
            open_prices,
            horizon=1,
            n_quantiles=5,
        )
        pd.testing.assert_frame_equal(
            wide_groups,
            expected_group_returns,
            check_freq=False,
        )
        assert np.isclose(wide_ic.mean(), ic_mean)
        assert wide_pairs == int(merged.shape[0])
        assert grouped["group"].dropna().between(1, 5).all()
        assert np.isclose(result["metrics"]["ic_mean"], ic_mean)
        assert np.isclose(result["metrics"]["ic_std"], ic_std)
        assert "abs_ir" not in result["metrics"]
        assert np.isfinite(ic_mean)
        assert np.isfinite(ic_std)
        assert np.isfinite(abs_ir)

        known_open = pd.DataFrame(
            {"asset": [10.0, 11.0, 13.0, 16.0]},
            index=pd.date_range("2024-04-01", periods=4, freq="B"),
        )
        expected_h1 = pd.DataFrame(
            {"asset": [13 / 11 - 1, 16 / 13 - 1, np.nan, np.nan]},
            index=known_open.index,
        )
        expected_h2 = pd.DataFrame(
            {"asset": [16 / 11 - 1, np.nan, np.nan, np.nan]},
            index=known_open.index,
        )
        pd.testing.assert_frame_equal(
            calculate_forward_open_return(known_open, horizon=1),
            expected_h1,
        )
        pd.testing.assert_frame_equal(
            calculate_forward_open_return(known_open, horizon=2),
            expected_h2,
        )

        operator_expression = (
            "where(vol > ts_mean(vol, 5), "
            "scale_cs(decay_linear(ts_corr(c, vol, 5), 3)), "
            "elementwise_min(ts_rank(c, 5), ts_argmax(h, 5)))"
        )
        operator_data = {
            "c": close,
            "h": high,
            "vol": volume,
        }
        operator_factor = evaluate_expression(
            operator_expression, operator_data
        )
        assert operator_factor.shape == close.shape
        assert np.isfinite(operator_factor.to_numpy()).any()

        deterministic = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]},
            index=pd.date_range("2024-03-01", periods=3),
        )
        assert ts_sum(deterministic, 2.9).iloc[-1].to_dict() == {
            "a": 5.0,
            "b": 3.0,
        }
        assert ts_argmax(deterministic, 3).iloc[-1].to_dict() == {
            "a": 3.0,
            "b": 1.0,
        }
        assert np.isclose(decay_linear(deterministic, 3).iloc[-1]["a"], 14 / 6)
        assert np.allclose(scale_cs(deterministic).abs().sum(axis=1), 1.0)
        outlier = pd.DataFrame(
            {"a": [0.0], "b": [1.0], "c": [2.0], "d": [100.0]},
            index=pd.date_range("2024-03-08", periods=1),
        )
        clipped = winsorize_cs(outlier, 0.25, 0.75)
        assert clipped.iloc[0].to_dict() == {"a": 0.75, "b": 1.0, "c": 2.0, "d": 26.5}
        standardized = zscore_cs(clipped)
        assert np.isclose(standardized.iloc[0].mean(), 0.0)
        assert np.isclose(standardized.iloc[0].std(ddof=0), 1.0)
        assert elementwise_max(
            deterministic, deterministic.iloc[::-1].set_axis(deterministic.index)
        ).shape == deterministic.shape
        assert np.isfinite(ts_corr(deterministic, deterministic, 3).iloc[-1]).all()
        assert adv(deterministic, 2).equals(
            deterministic.rolling(2, min_periods=1).mean()
        )

    print("factor evaluation smoke test passed")


if __name__ == "__main__":
    main()
