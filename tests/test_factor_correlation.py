#!/usr/bin/env python3
"""Incremental submitted-library factor correlation contract checks."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401 - imports src onto sys.path

from engine import evaluate_expression
from factor_correlation import (
    FactorCorrelationService,
    FactorCorrelationThresholdError,
    non_overlapping_pair_correlations,
)


def write_data(data_dir: Path) -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-02", periods=125, freq="B")
    columns = [f"{code:06d}" for code in range(1, 9)]
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(size=(len(index), len(columns))), axis=0),
        index=index,
        columns=columns,
    )
    volume = pd.DataFrame(
        rng.lognormal(mean=12, sigma=0.7, size=close.shape),
        index=index,
        columns=columns,
    )
    data_dir.mkdir()
    close.to_parquet(data_dir / "close_df.pq")
    (close * 1.001).to_parquet(data_dir / "open_df.pq")
    volume.to_parquet(data_dir / "volume_df.pq")


def factor(name: str, expression: str) -> dict[str, str]:
    return {"factor_name": name, "expression": expression}


def test_non_overlapping_pair_correlations() -> None:
    days = pd.date_range("2024-01-02", periods=125, freq="B")
    columns = ["000001", "000002", "000003"]
    values = np.arange(len(days) * len(columns), dtype=float).reshape(
        len(days), len(columns)
    )
    left = pd.DataFrame(values, index=days, columns=columns)
    right = left.copy()
    right.iloc[60:120] = -left.iloc[60:120]

    windows = non_overlapping_pair_correlations(left, right)
    assert len(windows) == 2
    assert windows[0]["start_day"] == "2024-01-02"
    assert windows[0]["end_day"] == str(days[59].date())
    assert np.isclose(windows[0]["correlation"], 1.0)
    assert np.isclose(windows[1]["correlation"], -1.0)
    # Both +1 and -1 violate because the threshold comparison uses abs(corr).
    assert all(row["exceeds_threshold"] for row in windows)
    assert windows[-1]["end_day"] == str(days[119].date())


def main() -> None:
    test_non_overlapping_pair_correlations()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        write_data(data_dir)
        settings = SimpleNamespace(data_dir=data_dir, state_dir=root / "state")
        service = FactorCorrelationService(settings)

        # Viewing an empty formal library must not poison the later first
        # submission with an unparsable empty CSV cache.
        assert service.matrix([])["matrix"] == []
        first = factor("close_rank", "rank_cs(c)")
        first_prepared = service.prepare_candidate(first, [])
        assert first_prepared.matrix.loc["close_rank", "close_rank"] == 1.0
        service.commit_candidate(first_prepared)
        assert (settings.state_dir / "factor_correlation" / "matrix.csv").is_file()

        second = factor("volume_rank", "rank_cs(vol)")
        with patch("factor_correlation.evaluate_expression", wraps=evaluate_expression) as evaluate:
            second_prepared = service.prepare_candidate(second, [first])
        # The old exposure came from the cache; only the candidate expression
        # is evaluated while adding its one new row and column.
        assert evaluate.call_count == 1
        assert evaluate.call_args.args[0] == "rank_cs(vol)"
        assert abs(second_prepared.matrix.loc["close_rank", "volume_rank"]) <= 0.75
        service.commit_candidate(second_prepared)

        payload = service.matrix([first, second])
        assert payload["factor_names"] == ["close_rank", "volume_rank"]
        assert len(payload["matrix"]) == 2
        assert payload["passed"]
        assert payload["max_abs_off_diagonal"] <= 0.75

        pair = service.pair_detail("close_rank", "volume_rank", [first, second])
        assert pair["window_size"] == 60
        assert pair["step_size"] == 60
        assert len(pair["windows"]) == 2
        assert pair["windows"][-1]["end_day"] == str(
            pd.date_range("2024-01-02", periods=125, freq="B")[119].date()
        )
        assert pair["max_abs_correlation"] == max(
            abs(row["correlation"])
            for row in pair["windows"]
            if row["correlation"] is not None
        )
        assert pair["violation_window_count"] == sum(
            abs(row["correlation"]) > 0.75
            for row in pair["windows"]
            if row["correlation"] is not None
        )

        duplicate = factor("duplicate_close_rank", "rank_cs(c * 2)")
        try:
            service.prepare_candidate(duplicate, [first, second])
        except FactorCorrelationThresholdError as exc:
            assert exc.violations[0]["factor_a"] == "close_rank"
            assert exc.violations[0]["factor_b"] == "duplicate_close_rank"
        else:
            raise AssertionError("a perfectly correlated candidate must be rejected")

        # A rejected candidate never expands the persisted n x n matrix.
        assert service.matrix([first, second])["factor_names"] == [
            "close_rank",
            "volume_rank",
        ]

    print("factor correlation matrix contract passed")


if __name__ == "__main__":
    main()
