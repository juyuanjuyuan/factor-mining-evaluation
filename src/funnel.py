#!/usr/bin/env python3
"""Resumable staged evaluation funnel for the runnable Alpha101 registry."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from engine import (
    DEFAULT_FILES,
    evaluate_factor_expression,
    expression_data_symbols,
    load_market_data,
    parse_and_validate_expression,
)
from factor_registry import DEFAULT_FACTOR_BATCH, select_registered_factors
from evaluators import (
    EvaluationMethod,
    evaluation_method_names,
    evaluation_required_data_symbols,
    resolve_evaluation_methods,
)
from paths import DATA_DIR, FACTOR_OUTPUT_DIR
from returns import RETURN_DEFINITION


@dataclass(frozen=True)
class FunnelStage:
    """One independently persisted stage in the evaluation funnel."""

    name: str
    directory: str
    requested_methods: tuple[str, ...]
    required_details: tuple[str, ...]
    required_artifacts: tuple[str, ...] = ()
    gate_metric: str | None = None

    @property
    def methods(self) -> tuple[EvaluationMethod, ...]:
        return resolve_evaluation_methods(self.requested_methods)

    @property
    def method_names(self) -> tuple[str, ...]:
        return evaluation_method_names(self.methods)


FUNNEL_STAGES: tuple[FunnelStage, ...] = (
    FunnelStage(
        name="stage1_validity",
        directory="stage1_validity",
        requested_methods=("future_data_perturbation",),
        required_details=("future_data_perturbation",),
        gate_metric="future_perturbation_changed_values",
    ),
    FunnelStage(
        name="stage2_ic",
        directory="stage2_ic",
        requested_methods=(
            "rank_ic",
            "rank_icir",
            "newey_west_ic_significance",
        ),
        required_details=("ic", "newey_west_ic_autocovariances"),
        gate_metric="nw_ic_p_value",
    ),
    FunnelStage(
        name="stage2b_neutral",
        directory="stage2b_neutral",
        requested_methods=(
            "market_cap_neutralize",
            "rank_ic",
            "rank_icir",
            "newey_west_ic_significance",
        ),
        required_details=(
            "market_cap_neutralization",
            "ic",
            "newey_west_ic_autocovariances",
        ),
        gate_metric="nw_ic_p_value",
    ),
    FunnelStage(
        name="stage3_portfolio",
        directory="stage3_portfolio",
        requested_methods=(
            "tradability_filter",
            "quantile_returns",
            "quantile_cumulative",
            "quantile_plot",
            "fitness",
            "top_quantile_performance",
            "rolling_sharpe",
            "rolling_drawdown",
        ),
        required_details=(
            "tradability_filter",
            "group_returns",
            "cumulative_returns",
            "yearly_fitness",
            "top_quantile_performance",
            "rolling_sharpe_20",
            "rolling_sharpe_60",
            "rolling_sharpe_252",
            "rolling_drawdown_20",
            "rolling_drawdown_60",
            "rolling_drawdown_252",
        ),
        required_artifacts=("plot",),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--registry-file",
        type=Path,
        default=DEFAULT_FACTOR_BATCH,
        help="canonical factor-definition JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FACTOR_OUTPUT_DIR / "alpha101_funnel_v1",
    )
    parser.add_argument(
        "--select",
        default="all",
        help="all, one Alpha number/name, comma list, or range such as 1-20,101",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.05,
        help="two-sided Newey-West p-value gate for stages 2 and 2b",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate expressions/files and print the frozen funnel configuration",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="rerun matching complete stage results instead of reusing them",
    )
    parser.add_argument(
        "--no-cache-data",
        action="store_true",
        help="reload only the matrices needed by each factor/stage to reduce memory",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop the entire funnel after the first evaluation failure",
    )
    parser.add_argument("--close-file", default=DEFAULT_FILES["c"])
    parser.add_argument("--open-file", default=DEFAULT_FILES["o"])
    parser.add_argument("--high-file", default=DEFAULT_FILES["h"])
    parser.add_argument("--low-file", default=DEFAULT_FILES["l"])
    parser.add_argument("--volume-file", default=DEFAULT_FILES["vol"])
    parser.add_argument("--amount-file", default=DEFAULT_FILES["amt"])
    parser.add_argument("--vwap-file", default=DEFAULT_FILES["vwap"])
    parser.add_argument("--market-cap-file", default=DEFAULT_FILES["cap"])
    parser.add_argument("--limit-ratio-file", default=DEFAULT_FILES["limit"])
    parser.add_argument("--st-status-file", default=DEFAULT_FILES["st"])
    return parser


def _file_names(args: argparse.Namespace) -> dict[str, str]:
    return {
        "c": args.close_file,
        "o": args.open_file,
        "h": args.high_file,
        "l": args.low_file,
        "vol": args.volume_file,
        "amt": args.amount_file,
        "vwap": args.vwap_file,
        "cap": args.market_cap_file,
        "limit": args.limit_ratio_file,
        "st": args.st_status_file,
    }


def required_data_symbols(factors: tuple[Any, ...]) -> set[str]:
    symbols = {"c", "o"}
    for factor in factors:
        symbols.update(expression_data_symbols(factor.expression))
    for stage in FUNNEL_STAGES:
        symbols.update(evaluation_required_data_symbols(stage.methods))
    return symbols


def _validate_files(
    data_dir: Path,
    file_names: Mapping[str, str],
    symbols: set[str],
) -> None:
    missing = [
        str(data_dir / file_names[symbol])
        for symbol in sorted(symbols)
        if not (data_dir / file_names[symbol]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing market matrices: {missing}")


def _safe_json_mapping(value: Any) -> dict[str, str] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): str(path) for key, path in payload.items()}


def _nonempty_relative_file(root: Path, relative_path: str) -> bool:
    try:
        path = (root / relative_path).resolve()
        path.relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return path.is_file() and path.stat().st_size > 0


def result_artifacts_complete(
    stage_dir: Path,
    record: Mapping[str, Any],
    stage: FunnelStage,
) -> bool:
    """Require the stage's generated code and declared outputs to be nonempty."""

    artifact_name = str(record.get("artifact_name", "")).strip()
    code_path = stage_dir / "code" / f"{artifact_name}.py"
    if not artifact_name or not code_path.is_file() or code_path.stat().st_size == 0:
        return False

    details = _safe_json_mapping(record.get("evaluation_details"))
    artifacts = _safe_json_mapping(record.get("evaluation_artifacts"))
    if details is None or artifacts is None:
        return False
    if not set(stage.required_details) <= set(details):
        return False
    if not set(stage.required_artifacts) <= set(artifacts):
        return False
    required_paths = (
        *(details[name] for name in stage.required_details),
        *(artifacts[name] for name in stage.required_artifacts),
    )
    return all(_nonempty_relative_file(stage_dir, path) for path in required_paths)


def matching_stage_record(
    stage_dir: Path,
    factor: Any,
    stage: FunnelStage,
    *,
    horizon: int,
    n_quantiles: int,
) -> dict[str, Any] | None:
    """Return a reusable result only when metadata and artifacts all match."""

    metrics_path = stage_dir / "metrics.csv"
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        return None
    metrics = pd.read_csv(metrics_path)
    required_columns = {
        "factor_name",
        "artifact_name",
        "expression",
        "horizon",
        "n_quantiles",
        "return_definition",
        "evaluation_methods",
        "evaluation_details",
        "evaluation_artifacts",
    }
    if not required_columns <= set(metrics.columns):
        return None
    rows = metrics.loc[metrics["factor_name"].astype(str) == factor.name]
    if len(rows) != 1:
        return None
    record = rows.iloc[0].to_dict()
    try:
        matches = (
            str(record["expression"]) == factor.expression
            and int(record["horizon"]) == horizon
            and int(record["n_quantiles"]) == n_quantiles
            and str(record["return_definition"]) == RETURN_DEFINITION
            and str(record["evaluation_methods"]) == ",".join(stage.method_names)
        )
    except (TypeError, ValueError):
        return None
    if not matches or not result_artifacts_complete(stage_dir, record, stage):
        return None
    return record


def evaluate_gate(
    stage: FunnelStage,
    metrics: Mapping[str, Any],
    *,
    significance_level: float,
) -> tuple[str, Any, str]:
    """Return (outcome, gate value, explanation) for a completed stage."""

    if stage.gate_metric is None:
        return "completed", None, "portfolio review completed; no v1 numeric gate"
    value = metrics.get(stage.gate_metric)
    if stage.gate_metric == "future_perturbation_changed_values":
        try:
            changed_values = int(value)
        except (TypeError, ValueError):
            changed_values = -1
        passed = changed_values == 0
        return (
            "passed" if passed else "eliminated",
            value,
            "no checkpoint changed" if passed else "future dependency detected",
        )
    try:
        p_value = float(value)
    except (TypeError, ValueError):
        p_value = np.nan
    passed = bool(np.isfinite(p_value) and p_value < significance_level)
    return (
        "passed" if passed else "eliminated",
        p_value,
        (
            f"p_value < {significance_level:g}"
            if passed
            else f"p_value is not < {significance_level:g}"
        ),
    )


def _append_csv(path: Path, record: Mapping[str, Any]) -> None:
    pd.DataFrame([record]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def _upsert_summary(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    new_rows = pd.DataFrame(records)
    if path.is_file():
        existing = pd.read_csv(path)
        selected = set(new_rows["factor_name"].astype(str))
        existing = existing.loc[~existing["factor_name"].astype(str).isin(selected)]
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows
    combined = combined.sort_values("factor_number", kind="stable")
    temporary = path.with_suffix(".tmp")
    combined.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _summary_record(
    factor: Any,
    stage_results: Mapping[str, Mapping[str, Any]],
    stage_outcomes: Mapping[str, str],
    *,
    run_id: str,
) -> dict[str, Any]:
    terminal_stage = ""
    final_status = "not_run"
    for stage in FUNNEL_STAGES:
        outcome = stage_outcomes.get(stage.name)
        if outcome:
            terminal_stage = stage.name
            if outcome in {"eliminated", "failed", "completed"}:
                final_status = outcome
                break
            final_status = "incomplete"

    raw = stage_results.get("stage2_ic", {})
    neutral = stage_results.get("stage2b_neutral", {})
    portfolio = stage_results.get("stage3_portfolio", {})
    validity = stage_results.get("stage1_validity", {})
    return {
        "funnel_run_id": run_id,
        "factor_number": factor.number,
        "factor_name": factor.name,
        "implementation_set": factor.implementation_set,
        "uses_proxy": factor.uses_proxy,
        "final_status": final_status,
        "terminal_stage": terminal_stage,
        **{
            f"{stage.name}_outcome": stage_outcomes.get(stage.name, "")
            for stage in FUNNEL_STAGES
        },
        "future_perturbation_changed_values": validity.get(
            "future_perturbation_changed_values"
        ),
        "raw_ic_mean": raw.get("ic_mean"),
        "raw_ir": raw.get("ir"),
        "raw_nw_p_value": raw.get("nw_ic_p_value"),
        "neutral_ic_mean": neutral.get("ic_mean"),
        "neutral_ir": neutral.get("ir"),
        "neutral_nw_p_value": neutral.get("nw_ic_p_value"),
        "neutralization_mean_r2": neutral.get(
            "market_cap_neutralization_mean_r2"
        ),
        "top_group_above_every_lower_group": portfolio.get(
            "top_group_above_every_lower_group"
        ),
        "top_group_final_cumulative": portfolio.get(
            "top_group_final_cumulative"
        ),
    }


def _manifest(
    *,
    run_id: str,
    factors: tuple[Any, ...],
    data_dir: Path,
    output_dir: Path,
    horizon: int,
    n_quantiles: int,
    significance_level: float,
    required_symbols: set[str],
) -> dict[str, Any]:
    return {
        "funnel_run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "funnel_version": 1,
        "factor_count": len(factors),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "horizon": horizon,
        "n_quantiles": n_quantiles,
        "significance_level": significance_level,
        "return_definition": RETURN_DEFINITION,
        "required_symbols": sorted(required_symbols),
        "stages": [
            {
                "name": stage.name,
                "directory": stage.directory,
                "methods": list(stage.method_names),
                "gate_metric": stage.gate_metric,
            }
            for stage in FUNNEL_STAGES
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    if not 0 < args.significance_level < 1:
        raise ValueError("significance-level must be strictly between 0 and 1")
    if args.horizon < 1 or args.quantiles < 3:
        raise ValueError("horizon must be positive and quantiles must be at least 3")

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    registry_file = args.registry_file.expanduser().resolve()
    factors = select_registered_factors(
        args.select,
        implementation_set="runnable",
        path=registry_file,
    )
    if not factors:
        raise ValueError("No runnable Alpha101 factors were selected")
    for factor in factors:
        parse_and_validate_expression(factor.expression)
    file_names = _file_names(args)
    symbols = required_data_symbols(factors)
    _validate_files(data_dir, file_names, symbols)

    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    manifest = _manifest(
        run_id=run_id,
        factors=factors,
        data_dir=data_dir,
        output_dir=output_dir,
        horizon=args.horizon,
        n_quantiles=args.quantiles,
        significance_level=args.significance_level,
        required_symbols=symbols,
    )
    manifest["registry_file"] = str(registry_file)
    if args.dry_run:
        manifest["factor_names"] = [factor.name for factor in factors]
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _append_csv(output_dir / "funnel_runs.csv", manifest)
    status_path = output_dir / "funnel_status.csv"
    stage_results: dict[str, dict[str, dict[str, Any]]] = {
        factor.name: {} for factor in factors
    }
    stage_outcomes: dict[str, dict[str, str]] = {
        factor.name: {} for factor in factors
    }
    candidates = list(factors)
    failures: list[dict[str, str]] = []
    evaluated_count = 0
    reused_count = 0
    shared_data: dict[str, pd.DataFrame] | None = None
    aborted = False

    for stage in FUNNEL_STAGES:
        stage_dir = output_dir / stage.directory
        next_candidates: list[Any] = []
        for position, factor in enumerate(candidates, start=1):
            record = None if args.rerun else matching_stage_record(
                stage_dir,
                factor,
                stage,
                horizon=args.horizon,
                n_quantiles=args.quantiles,
            )
            execution_state = "reused"
            started = time.perf_counter()
            try:
                if record is None:
                    if shared_data is None and not args.no_cache_data:
                        print(
                            json.dumps(
                                {
                                    "state": "loading_data",
                                    "symbols": sorted(symbols),
                                    "data_dir": str(data_dir),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        shared_data = load_market_data(
                            data_dir,
                            " + ".join(sorted(symbols)),
                            file_names,
                        )
                    result = evaluate_factor_expression(
                        factor_name=factor.name,
                        expression=factor.expression,
                        data_dir=data_dir,
                        output_dir=stage_dir,
                        horizon=args.horizon,
                        n_quantiles=args.quantiles,
                        file_names=file_names,
                        preloaded_data=shared_data,
                        evaluation_methods=stage.methods,
                    )
                    record = result["metrics"]
                    execution_state = "evaluated"
                    evaluated_count += 1
                else:
                    reused_count += 1

                outcome, gate_value, reason = evaluate_gate(
                    stage,
                    record,
                    significance_level=args.significance_level,
                )
                elapsed = time.perf_counter() - started
                stage_results[factor.name][stage.name] = dict(record)
                stage_outcomes[factor.name][stage.name] = outcome
                if outcome == "passed":
                    next_candidates.append(factor)
                status = {
                    "funnel_run_id": run_id,
                    "factor_number": factor.number,
                    "factor_name": factor.name,
                    "stage": stage.name,
                    "execution_state": execution_state,
                    "outcome": outcome,
                    "gate_metric": stage.gate_metric or "",
                    "gate_value": gate_value,
                    "reason": reason,
                    "elapsed_seconds": elapsed,
                    "error": "",
                }
                _append_csv(status_path, status)
                print(
                    json.dumps(
                        {
                            "stage": stage.name,
                            "position": position,
                            "stage_total": len(candidates),
                            "factor_name": factor.name,
                            "execution_state": execution_state,
                            "outcome": outcome,
                            "gate_value": gate_value,
                            "elapsed_seconds": elapsed,
                        },
                        ensure_ascii=False,
                        allow_nan=True,
                    ),
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                error = f"{type(exc).__name__}: {exc}"
                failures.append(
                    {
                        "factor_name": factor.name,
                        "stage": stage.name,
                        "error": error,
                    }
                )
                stage_outcomes[factor.name][stage.name] = "failed"
                _append_csv(
                    status_path,
                    {
                        "funnel_run_id": run_id,
                        "factor_number": factor.number,
                        "factor_name": factor.name,
                        "stage": stage.name,
                        "execution_state": execution_state,
                        "outcome": "failed",
                        "gate_metric": stage.gate_metric or "",
                        "gate_value": "",
                        "reason": "evaluation failed",
                        "elapsed_seconds": elapsed,
                        "error": error,
                    },
                )
                print(
                    json.dumps(
                        {
                            "stage": stage.name,
                            "factor_name": factor.name,
                            "outcome": "failed",
                            "elapsed_seconds": elapsed,
                            "error": error,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if args.stop_on_error:
                    aborted = True
                    break
            finally:
                gc.collect()
        candidates = next_candidates
        if aborted or not candidates:
            break

    summaries = [
        _summary_record(
            factor,
            stage_results[factor.name],
            stage_outcomes[factor.name],
            run_id=run_id,
        )
        for factor in factors
    ]
    _upsert_summary(output_dir / "funnel_summary.csv", summaries)
    final_counts = (
        pd.Series([record["final_status"] for record in summaries])
        .value_counts()
        .to_dict()
    )
    summary = {
        "funnel_run_id": run_id,
        "selected": len(factors),
        "evaluated_stage_runs": evaluated_count,
        "reused_stage_runs": reused_count,
        "final_status_counts": final_counts,
        "failed": len(failures),
        "failures": failures,
        "summary": str(output_dir / "funnel_summary.csv"),
        "status": str(status_path),
        "stage_directories": {
            stage.name: str(output_dir / stage.directory)
            for stage in FUNNEL_STAGES
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if failures or aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
