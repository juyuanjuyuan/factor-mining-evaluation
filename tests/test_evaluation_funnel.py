#!/usr/bin/env python3
"""Unit checks for staged Alpha101 funnel gates and resume validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import _bootstrap  # noqa: F401
from funnel import (
    FUNNEL_STAGES,
    _summary_record,
    evaluate_gate,
    matching_stage_record,
)
from returns import RETURN_DEFINITION


def _write_complete_result(
    root: Path,
    factor: SimpleNamespace,
    stage_number: int,
) -> None:
    stage = FUNNEL_STAGES[stage_number]
    artifact_name = factor.name
    (root / "code").mkdir(parents=True)
    (root / "details").mkdir()
    (root / "plots").mkdir()
    (root / "code" / f"{artifact_name}.py").write_text(
        "print('rerun')\n",
        encoding="utf-8",
    )
    details = {}
    for name in stage.required_details:
        path = f"details/{artifact_name}__{name}.csv"
        (root / path).write_text("day,value\n2024-01-02,1\n", encoding="utf-8")
        details[name] = path
    artifacts = {}
    for name in stage.required_artifacts:
        path = f"plots/{artifact_name}.png"
        (root / path).write_bytes(b"not-empty")
        artifacts[name] = path
    pd.DataFrame(
        [
            {
                "factor_name": factor.name,
                "artifact_name": artifact_name,
                "expression": factor.expression,
                "horizon": 1,
                "n_quantiles": 10,
                "return_definition": RETURN_DEFINITION,
                "evaluation_methods": ",".join(stage.method_names),
                "evaluation_details": json.dumps(details),
                "evaluation_artifacts": json.dumps(artifacts),
            }
        ]
    ).to_csv(root / "metrics.csv", index=False)


def main() -> None:
    assert [stage.name for stage in FUNNEL_STAGES] == [
        "stage1_validity",
        "stage2_ic",
        "stage2b_neutral",
        "stage3_portfolio",
    ]
    assert FUNNEL_STAGES[2].method_names == (
        "market_cap_neutralize",
        "rank_ic",
        "rank_icir",
        "newey_west_ic_significance",
    )
    assert FUNNEL_STAGES[3].method_names == (
        "tradability_filter",
        "quantile_returns",
        "quantile_cumulative",
        "quantile_plot",
        "top_quantile_performance",
        "rolling_sharpe",
        "rolling_drawdown",
    )

    outcome, _, _ = evaluate_gate(
        FUNNEL_STAGES[0],
        {"future_perturbation_changed_values": 0},
        significance_level=0.05,
    )
    assert outcome == "passed"
    outcome, _, _ = evaluate_gate(
        FUNNEL_STAGES[1],
        {"nw_ic_p_value": 0.049},
        significance_level=0.05,
    )
    assert outcome == "passed"
    outcome, _, _ = evaluate_gate(
        FUNNEL_STAGES[1],
        {"nw_ic_p_value": 0.05},
        significance_level=0.05,
    )
    assert outcome == "eliminated"
    outcome, _, _ = evaluate_gate(
        FUNNEL_STAGES[3],
        {},
        significance_level=0.05,
    )
    assert outcome == "completed"

    summary_factor = SimpleNamespace(
        number=1,
        name="alpha001",
        implementation_set="exact",
        uses_proxy=False,
    )
    summary = _summary_record(
        summary_factor,
        {"stage3_portfolio": {"top_group_final_cumulative": 0.25}},
        {"stage3_portfolio": "completed"},
        run_id="test-run",
    )
    assert summary["top_group_final_cumulative"] == 0.25

    factor = SimpleNamespace(name="alpha001", expression="rank_cs(c)")
    with tempfile.TemporaryDirectory() as temporary:
        stage_dir = Path(temporary)
        _write_complete_result(stage_dir, factor, 1)
        assert matching_stage_record(
            stage_dir,
            factor,
            FUNNEL_STAGES[1],
            horizon=1,
            n_quantiles=10,
        )
        detail = next((stage_dir / "details").iterdir())
        detail.write_text("", encoding="utf-8")
        assert matching_stage_record(
            stage_dir,
            factor,
            FUNNEL_STAGES[1],
            horizon=1,
            n_quantiles=10,
        ) is None

    print("evaluation funnel passed")


if __name__ == "__main__":
    main()
