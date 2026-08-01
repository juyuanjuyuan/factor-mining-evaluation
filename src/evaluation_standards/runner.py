"""Execute code-owned evaluation standards on a frozen test window."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from engine import (
    DEFAULT_FILES,
    evaluate_factor_expression,
    load_market_data,
)
from evaluators import evaluation_required_data_symbols
from returns import RETURN_DEFINITION

from .registry import evaluate_standard_gate, resolve_evaluation_standards


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate_factor_standards(
    *,
    factor_name: str,
    expression: str,
    data_dir: str | Path,
    output_dir: str | Path,
    signal_start: str,
    signal_end: str,
    standards: str | Iterable[str] | None = None,
    horizon: int = 1,
    n_quantiles: int = 10,
    decay: int = 1,
    file_names: Mapping[str, str] | None = None,
    preloaded_data: Mapping[str, pd.DataFrame] | None = None,
    significance_level: float = 0.05,
    minimum_ic_mean: float = 0.0,
    minimum_rolling_sharpe_60_median: float = 1.0,
    minimum_annualized_return: float = 0.30,
) -> dict[str, Any]:
    """Run selected standards only on the bounded, frozen test sample."""

    selected = resolve_evaluation_standards(standards)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_files = dict(DEFAULT_FILES)
    if file_names:
        resolved_files.update(file_names)

    required_symbols: set[str] = set()
    for standard in selected:
        required_symbols.update(evaluation_required_data_symbols(standard.methods))
    shared_data = (
        load_market_data(
            data_dir,
            expression,
            resolved_files,
            extra_symbols=required_symbols,
        )
        if preloaded_data is None
        else preloaded_data
    )

    standard_results: dict[str, Any] = {}
    for standard in selected:
        standard_dir = output_root / standard.name
        result = evaluate_factor_expression(
            factor_name=factor_name,
            expression=expression,
            data_dir=data_dir,
            output_dir=standard_dir,
            horizon=horizon,
            n_quantiles=n_quantiles,
            decay=decay,
            file_names=resolved_files,
            preloaded_data=shared_data,
            evaluation_methods=standard.methods,
            signal_start=signal_start,
            signal_end=signal_end,
        )
        gate = evaluate_standard_gate(
            standard.name,
            result["metrics"],
            significance_level=significance_level,
            minimum_ic_mean=minimum_ic_mean,
            minimum_rolling_sharpe_60_median=minimum_rolling_sharpe_60_median,
            minimum_annualized_return=minimum_annualized_return,
        )
        standard_results[standard.name] = {
            "label": standard.label,
            "description": standard.description,
            "methods": list(standard.method_names),
            "gate": gate.as_dict(),
            "metrics": result["metrics"],
            "metrics_path": result["metrics_path"],
            "detail_paths": result["detail_paths"],
            "artifact_paths": result["artifact_paths"],
            "code_path": result["code_path"],
        }

    overall_passed = bool(standard_results) and all(
        item["gate"]["passed"] for item in standard_results.values()
    )
    payload = {
        "factor_name": factor_name,
        "expression": expression,
        "signal_start": str(signal_start),
        "signal_end": str(signal_end),
        "horizon": horizon,
        "n_quantiles": n_quantiles,
        "decay": decay,
        "return_definition": RETURN_DEFINITION,
        "thresholds": {
            "significance_level": significance_level,
            "minimum_ic_mean": minimum_ic_mean,
            "minimum_rolling_sharpe_60_median": minimum_rolling_sharpe_60_median,
            "minimum_annualized_return": minimum_annualized_return,
        },
        "overall_passed": overall_passed,
        "standards": standard_results,
    }
    summary_path = output_root / "standard_evaluation.json"
    _atomic_json(summary_path, payload)
    payload["summary_path"] = str(summary_path)
    return payload
