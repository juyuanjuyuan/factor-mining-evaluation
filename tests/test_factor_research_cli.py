#!/usr/bin/env python3
"""Synthetic parity checks for the agent-oriented factor research CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_factor_research.py"
BASIC_SCRIPT = PROJECT_ROOT / "scripts" / "run_factor_evaluation.py"


def write_market_data(data_dir: Path) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from engine import DEFAULT_FILES

    rng = np.random.default_rng(23)
    index = pd.date_range("2024-01-02", periods=100, freq="B")
    columns = [f"{number:06d}" for number in range(1, 9)]
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 1, (len(index), len(columns))), axis=0),
        index=index,
        columns=columns,
    ).clip(lower=5)
    frames = {
        "c": close,
        "o": close * (1 + rng.normal(0, 0.003, close.shape)),
        "h": close * 1.01,
        "l": close * 0.99,
        "vol": pd.DataFrame(rng.uniform(1e5, 1e6, close.shape), index=index, columns=columns),
        "amt": pd.DataFrame(rng.uniform(1e7, 1e8, close.shape), index=index, columns=columns),
        "vwap": close,
        "cap": pd.DataFrame(rng.uniform(1e9, 1e10, close.shape), index=index, columns=columns),
        "limit": pd.DataFrame(0.10, index=index, columns=columns),
        "st": pd.DataFrame(False, index=index, columns=columns),
        "delisting": pd.DataFrame(False, index=index, columns=columns),
    }
    frames["industry"] = pd.DataFrame(
        {
            "trade_date": np.repeat(index, len(columns)),
            "security_code": list(columns) * len(index),
            "industry_l1_code": np.tile(["A"] * 4 + ["B"] * 4, len(index)),
        }
    )
    data_dir.mkdir()
    for symbol, filename in DEFAULT_FILES.items():
        frames[symbol].to_parquet(data_dir / filename)


def run_cli(root: Path, *arguments: str, expected_code: int = 0) -> dict:
    command = [
        sys.executable,
        str(SCRIPT),
        "--data-dir",
        str(root / "data"),
        "--state-dir",
        str(root / "state"),
        "--registry-dir",
        str(root / "registry"),
        *arguments,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == expected_code, (
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )
    output = completed.stdout if expected_code == 0 else completed.stderr
    return json.loads(output)


def run_basic_evaluation(root: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(BASIC_SCRIPT),
            "--data-dir",
            str(root / "data"),
            "--output-dir",
            str(root / "basic-output"),
            "--factor-name",
            "basic_window_candidate",
            "--expression",
            "rank_cs(delta(c, 2))",
            "--methods",
            "rank_ic,rank_icir",
            "--signal-start",
            "2024-02-01",
            "--signal-end",
            "2024-04-30",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    return json.loads(completed.stdout)


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "state").mkdir()
        (root / "registry").mkdir()
        write_market_data(root / "data")

        methods = run_cli(root, "catalog", "methods")
        assert [item["name"] for item in methods if item["is_default"]] == [
            "rank_ic",
            "rank_icir",
            "quantile_returns",
            "quantile_cumulative",
            "quantile_plot",
        ]
        validation = run_cli(
            root,
            "validate",
            "--expression",
            "rank_cs(delta(c, 2))",
        )
        assert validation == {
            "valid": True,
            "symbols": ["c"],
            "operators": ["delta", "rank_cs"],
        }

        basic_result = run_basic_evaluation(root)
        assert basic_result["metrics"]["signal_start"] == "2024-02-01"
        assert basic_result["metrics"]["signal_end"] == "2024-04-30"
        assert basic_result["metrics"]["sample_start_day"] == "2024-02-01"
        assert basic_result["metrics"]["sample_end_day"] == "2024-04-26"

        created_factor = run_cli(
            root,
            "factor",
            "add",
            "--factor-name",
            "cli_candidate",
            "--expression",
            "rank_cs(delta(c, 2))",
            "--project",
            "CLI smoke",
            "--tag",
            "agent-candidate",
        )
        assert created_factor["batch_id"] == "webapp_test_factors"
        assert created_factor["tags"] == ["agent-candidate"]

        research_id = "short-term-reversal-v1"
        note = run_cli(
            root,
            "journal",
            "note",
            "--research-id",
            research_id,
            "--research-phase",
            "exploration",
            "--research-direction",
            "短期价格反转",
            "--hypothesis",
            "近期收益冲击会在下一持有期部分反转",
            "--note",
            "从两日收盘价变化开始，先在训练期检查截面 IC。",
        )
        assert note["event"] == "research_note"
        assert note["research"]["research_id"] == research_id

        methods_value = (
            "rank_ic,rank_icir,market_cap_neutralize,rank_ic,rank_icir"
        )
        template = run_cli(
            root,
            "template",
            "create",
            "--name",
            "cli-repeated-ic",
            "--kind",
            "methods",
            "--methods",
            methods_value,
        )
        assert template["methods"] == methods_value.split(",")

        job = run_cli(
            root,
            "run",
            "--factor-name",
            "cli_inline_candidate",
            "--expression",
            "rank_cs(delta(c, 2))",
            "--template",
            "cli-repeated-ic",
            "--gate",
            "ic_count__2:gte:1",
            "--signal-start",
            "2024-02-01",
            "--signal-end",
            "2024-04-30",
            "--research-id",
            research_id,
            "--research-phase",
            "training",
            "--research-direction",
            "短期价格反转",
            "--hypothesis",
            "近期收益冲击会在下一持有期部分反转",
            "--iteration-note",
            "以二日变化作为训练期的首个可运行版本。",
            "--timeout",
            "90",
        )
        assert job["status"] == "succeeded"
        assert len(job["runs"]) == 1
        run = job["runs"][0]
        assert run["status"] == "succeeded"
        assert run["methods"] == methods_value.split(",")
        assert run["gate_outcome"] == "passed"
        assert run["run_params"] == {
            "decay": 1,
            "signal_start": "2024-02-01",
            "signal_end": "2024-04-30",
        }
        assert run["result"]["signal_start"] == "2024-02-01"
        assert run["result"]["signal_end"] == "2024-04-30"
        assert run["result"]["sample_end_day"] == "2024-04-26"
        assert "ic_mean__2" in run["result"]
        assert Path(run["output_dir"], "metrics.csv").is_file()

        testing_job = run_cli(
            root,
            "run",
            "--factor-name",
            "cli_testing_candidate",
            "--expression",
            "rank_cs(delta(c, 3))",
            "--methods",
            "rank_ic,rank_icir",
            "--signal-start",
            "2024-05-01",
            "--signal-end",
            "2024-05-17",
            "--research-id",
            research_id,
            "--research-phase",
            "testing",
            "--research-direction",
            "短期价格反转",
            "--hypothesis",
            "近期收益冲击会在下一持有期部分反转",
            "--parent-candidate",
            "cli_inline_candidate",
            "--iteration-note",
            "冻结训练期方向后，用三日变化在独立 testing 窗口复核。",
            "--timeout",
            "90",
        )
        assert testing_job["status"] == "succeeded"
        assert testing_job["runs"][0]["run_params"] == {
            "decay": 1,
            "signal_start": "2024-05-01",
            "signal_end": "2024-05-17",
        }

        funnel = run_cli(
            root,
            "run",
            "--kind",
            "funnel",
            "--factor",
            "webapp_test_factors/cli_candidate",
            "--significance-level",
            "0.1",
            "--signal-start",
            "2024-02-01",
            "--signal-end",
            "2024-04-30",
            "--research-id",
            research_id,
            "--research-phase",
            "testing",
            "--research-direction",
            "短期价格反转",
            "--hypothesis",
            "近期收益冲击会在下一持有期部分反转",
            "--parent-candidate",
            "cli_inline_candidate",
            "--iteration-note",
            "保留一次异步 funnel 取消事件，确认未完成实验也可回看。",
            "--submit-only",
        )
        assert funnel["kind"] == "funnel"
        assert funnel["status"] == "queued"
        assert funnel["runs"][0]["stage"] == "stage1_validity"
        assert funnel["runs"][0]["run_params"] == {
            "decay": 1,
            "signal_start": "2024-02-01",
            "signal_end": "2024-04-30",
        }
        cancelled = run_cli(
            root,
            "job",
            "cancel",
            "--job-id",
            funnel["id"],
        )
        assert cancelled["status"] == "cancelled"

        history = run_cli(
            root,
            "journal",
            "history",
            "--research-id",
            research_id,
            "--limit",
            "20",
        )
        events = [record["event"] for record in history["records"]]
        assert events.count("evaluation_submitted") == 3
        assert events.count("evaluation_finished") == 3
        assert "research_note" in events
        assert any(
            record["job"]["status"] == "cancelled"
            for record in history["records"]
            if record["event"] == "evaluation_finished"
        )

        summary = run_cli(
            root,
            "journal",
            "summary",
            "--research-id",
            research_id,
        )
        assert summary["records_considered"] == len(history["records"])
        assert summary["recent_notes"][0]["note"].startswith("从两日收盘价变化")
        candidates = {item["factor_name"]: item for item in summary["candidates"]}
        training_evaluation = candidates["cli_inline_candidate"]["evaluations"][0]
        assert training_evaluation["research_phase"] == "training"
        assert training_evaluation["status"] == "succeeded"
        assert training_evaluation["requested_signal_start"] == "2024-02-01"
        assert "ic_mean__2" in training_evaluation["metrics"]
        testing_candidate = candidates["cli_testing_candidate"]
        assert testing_candidate["parent_candidates"] == ["cli_inline_candidate"]
        assert testing_candidate["evaluations"][0]["research_phase"] == "testing"

    print("factor research CLI test passed")


if __name__ == "__main__":
    main()
