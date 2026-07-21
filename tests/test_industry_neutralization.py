#!/usr/bin/env python3
"""Industry-neutralization numerical, loader, and pipeline contracts."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from engine import (
    DEFAULT_FILES,
    evaluate_factor_expression,
    normalize_industry_classification_frame,
    parse_and_validate_expression,
)
from evaluators import resolve_evaluation_methods
from transforms import (
    neutralize_factor_by_industry,
    neutralize_factor_by_industry_and_market_cap,
)


def test_daily_industry_residuals_and_exclusions() -> None:
    day = pd.Timestamp("2024-01-02")
    codes = [f"{number:06d}" for number in range(1, 11)]
    factor = pd.DataFrame(
        [[10.0, 11.0, 12.0, 100.0, 103.0, 106.0, 5.0, 6.0, 7.0, np.nan]],
        index=[day],
        columns=codes,
    )
    industry = pd.DataFrame(
        [["A", "A", "A", "B", "B", "B", "C", "C", pd.NA, "A"]],
        index=[day],
        columns=codes,
        dtype="string",
    )

    residuals, diagnostics = neutralize_factor_by_industry(factor, industry)
    np.testing.assert_allclose(
        residuals.loc[day, codes[:6]].to_numpy(dtype=float),
        [-1.0, 0.0, 1.0, -3.0, 0.0, 3.0],
    )
    assert residuals.loc[day, codes[6:]].isna().all()
    assert abs(float(residuals.loc[day, codes[:3]].mean())) < 1e-12
    assert abs(float(residuals.loc[day, codes[3:6]].mean())) < 1e-12
    detail = diagnostics.loc[day]
    assert detail["n_factor_obs"] == 9
    assert detail["n_classified_obs"] == 8
    assert detail["n_used_obs"] == 6
    assert detail["n_unclassified_obs"] == 1
    assert detail["n_small_industry_obs"] == 2
    assert detail["industry_count"] == 3
    assert detail["eligible_industry_count"] == 2


def test_long_industry_data_is_same_day_only() -> None:
    days = pd.date_range("2024-01-02", periods=3, freq="B")
    codes = pd.Index(["000001", "000002", "000003"])
    close = pd.DataFrame(1.0, index=days, columns=codes)
    long = pd.DataFrame(
        {
            "trade_date": [days[0]] * 3,
            "security_code": ["1", "2", "3"],
            "industry_l1_code": ["A", "A", "B"],
        }
    )
    normalized = normalize_industry_classification_frame(long, close)
    assert normalized.loc[days[0], "000001"] == "A"
    assert normalized.loc[days[1]].isna().all()
    try:
        parse_and_validate_expression("industry")
    except ValueError as exc:
        assert "Unknown expression name" in str(exc)
    else:
        raise AssertionError("industry must remain evaluator-only")


def test_joint_industry_market_cap_residual_is_joint_ols() -> None:
    day = pd.Timestamp("2024-01-02")
    codes = [f"{number:06d}" for number in range(1, 13)]
    log_cap = np.array(
        [20.1, 20.4, 20.8, 21.0, 19.9, 20.3, 20.7, 21.2, 20.0, 20.5, 20.9, 21.3]
    )
    labels = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
    industry = pd.DataFrame([labels], index=[day], columns=codes, dtype="string")
    market_cap = pd.DataFrame([np.exp(log_cap)], index=[day], columns=codes)
    dummies = np.column_stack((labels == "A", labels == "B")).astype(float)
    design = np.column_stack((np.ones(len(codes)), log_cap, dummies))
    raw_noise = np.array(
        [2.0, -1.0, 3.0, -2.0, -3.0, 4.0, -4.0, 1.0, 5.0, -2.0, 0.0, -3.0]
    )
    expected_residual = raw_noise - design @ np.linalg.lstsq(
        design, raw_noise, rcond=None
    )[0]
    factor = pd.DataFrame(
        [design @ np.array([1.5, 0.7, 3.0, -2.0]) + expected_residual],
        index=[day],
        columns=codes,
    )

    residuals, diagnostics = neutralize_factor_by_industry_and_market_cap(
        factor, industry, market_cap
    )
    np.testing.assert_allclose(
        residuals.loc[day].to_numpy(dtype=float), expected_residual
    )
    np.testing.assert_allclose(
        design.T @ residuals.loc[day].to_numpy(dtype=float), 0.0, atol=1e-10
    )
    detail = diagnostics.loc[day]
    assert detail["n_used_obs"] == 12
    assert detail["n_model_parameters"] == 4
    assert detail["model_rank"] == 4
    assert detail["residual_degrees_of_freedom"] == 8
    assert abs(float(detail["log_cap_beta"]) - 0.7) < 1e-12


def test_joint_transform_excludes_invalid_cap_and_small_industries() -> None:
    day = pd.Timestamp("2024-01-02")
    codes = [f"{number:06d}" for number in range(1, 11)]
    factor = pd.DataFrame([np.arange(10, dtype=float)], index=[day], columns=codes)
    industry = pd.DataFrame(
        [["A"] * 4 + ["B"] * 3 + ["C"] * 3],
        index=[day],
        columns=codes,
        dtype="string",
    )
    market_cap = pd.DataFrame(
        [[10.0, 11.0, 12.0, 13.0, 0.0, 20.0, 21.0, 30.0, 31.0, 32.0]],
        index=[day],
        columns=codes,
    )

    residuals, diagnostics = neutralize_factor_by_industry_and_market_cap(
        factor, industry, market_cap
    )
    assert residuals.loc[day, codes[:4]].notna().all()
    assert residuals.loc[day, codes[4:7]].isna().all()
    assert residuals.loc[day, codes[7:]].notna().all()
    detail = diagnostics.loc[day]
    assert detail["n_invalid_cap_obs"] == 1
    assert detail["n_small_industry_obs"] == 2
    assert detail["n_used_obs"] == 7


def test_industry_pipeline_loads_long_input_and_persists_detail() -> None:
    rng = np.random.default_rng(31)
    days = pd.date_range("2024-01-02", periods=18, freq="B")
    codes = [f"{number:06d}" for number in range(1, 13)]
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 1, (len(days), len(codes))), axis=0),
        index=days,
        columns=codes,
    ).clip(lower=1)
    open_prices = close * pd.DataFrame(
        1 + rng.normal(0, 0.002, close.shape), index=days, columns=codes
    )
    long = pd.DataFrame(
        {
            "trade_date": np.repeat(days, len(codes)),
            "security_code": codes * len(days),
            "industry_l1_code": np.tile(["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3, len(days)),
        }
    )
    cap = pd.DataFrame(
        rng.lognormal(20, 1, close.shape), index=days, columns=codes
    )

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        close.to_parquet(root / DEFAULT_FILES["c"])
        open_prices.to_parquet(root / DEFAULT_FILES["o"])
        cap.to_parquet(root / DEFAULT_FILES["cap"])
        long.to_parquet(root / DEFAULT_FILES["industry"])
        result = evaluate_factor_expression(
            factor_name="industry_pipeline",
            expression="rank_cs(c)",
            data_dir=root,
            output_dir=root / "output",
            n_quantiles=3,
            evaluation_methods=resolve_evaluation_methods(
                "industry_neutralize,rank_icir"
            ),
        )
        assert result["evaluation_methods"] == [
            "industry_neutralize",
            "rank_ic",
            "rank_icir",
        ]
        assert "industry_neutralization" in result["detail_paths"]
        assert result["metrics"]["industry_neutralized_days"] > 0
        detail = pd.read_csv(result["detail_paths"]["industry_neutralization"], index_col=0)
        assert {"n_used_obs", "r_squared", "residual_std"} <= set(detail.columns)

        joint = evaluate_factor_expression(
            factor_name="industry_market_cap_pipeline",
            expression="rank_cs(c)",
            data_dir=root,
            output_dir=root / "joint_output",
            n_quantiles=3,
            evaluation_methods=resolve_evaluation_methods(
                "industry_market_cap_neutralize,rank_icir"
            ),
        )
        assert joint["evaluation_methods"] == [
            "industry_market_cap_neutralize",
            "rank_ic",
            "rank_icir",
        ]
        assert "industry_market_cap_neutralization" in joint["detail_paths"]
        assert joint["metrics"]["industry_market_cap_neutralized_days"] > 0
        joint_detail = pd.read_csv(
            joint["detail_paths"]["industry_market_cap_neutralization"], index_col=0
        )
        assert {"n_model_parameters", "log_cap_beta", "r_squared"} <= set(
            joint_detail.columns
        )


def main() -> None:
    test_daily_industry_residuals_and_exclusions()
    test_long_industry_data_is_same_day_only()
    test_joint_industry_market_cap_residual_is_joint_ols()
    test_joint_transform_excludes_invalid_cap_and_small_industries()
    test_industry_pipeline_loads_long_input_and_persists_detail()
    print("Industry neutralization tests passed.")


if __name__ == "__main__":
    main()
