#!/usr/bin/env python3
"""Batch-evaluate exact-input or VWAP-proxy Alpha101 expressions.

The runner uses the standard IC/IR and quantile-return evaluator, archives
each factor independently, skips matching successful results by default, and
records failures without discarding the rest of the batch.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from engine import (
    DEFAULT_FILES,
    evaluate_factor_expression,
    expression_data_symbols,
    load_market_data,
    parse_and_validate_expression,
)
from factor_registry import (
    DEFAULT_FACTOR_BATCH,
    load_registered_factors,
    select_registered_factors,
)
from returns import RETURN_DEFINITION
from evaluators import (
    available_evaluation_methods,
    evaluation_method_names,
    evaluation_required_data_symbols,
    resolve_evaluation_methods,
)
from paths import DATA_DIR, FACTOR_OUTPUT_DIR


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
        "--set",
        choices=("exact", "vwap_proxy", "market_cap", "runnable"),
        default="exact",
        help="factor set to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to outputs/factor_evaluation/alpha101_<set>",
    )
    parser.add_argument(
        "--select",
        default="all",
        help="all, one alpha number/name, comma list, or range such as 1-20,101",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="open-to-open holding horizon after entering at t+1 open",
    )
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument(
        "--methods",
        default="default",
        help=(
            "comma-separated evaluation methods; dependencies are added "
            f"automatically. Available: {', '.join(available_evaluation_methods())}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and list the batch without loading matrices or writing results",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="rerun factors even when matching metrics already exist",
    )
    parser.add_argument(
        "--no-cache-data",
        action="store_true",
        help="reload only the required matrices per factor to reduce peak memory",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop after the first failed factor instead of continuing",
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


def _validate_files(
    data_dir: Path,
    factors: tuple[Any, ...],
    file_names: dict[str, str],
    extra_symbols: set[str] | None = None,
) -> None:
    required = {"c", "o"}
    if extra_symbols:
        required.update(extra_symbols)
    for factor in factors:
        required.update(expression_data_symbols(factor.expression))
    missing = [
        str(data_dir / file_names[symbol])
        for symbol in sorted(required)
        if not (data_dir / file_names[symbol]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing market matrices: {missing}")


def _matching_completed(
    metrics_path: Path,
    factors: tuple[Any, ...],
    horizon: int,
    quantiles: int,
    method_names: tuple[str, ...],
) -> set[str]:
    if not metrics_path.is_file():
        return set()
    metrics = pd.read_csv(metrics_path)
    required_columns = {
        "factor_name",
        "expression",
        "horizon",
        "n_quantiles",
        "return_definition",
    }
    if not required_columns.issubset(metrics.columns):
        return set()
    expected = {factor.name: factor.expression for factor in factors}
    requested_methods = ",".join(method_names)
    default_methods = ",".join(
        evaluation_method_names(resolve_evaluation_methods("default"))
    )
    completed: set[str] = set()
    for row in metrics.itertuples(index=False):
        name = str(row.factor_name)
        raw_methods = getattr(row, "evaluation_methods", None)
        row_methods = (
            default_methods
            if raw_methods is None or pd.isna(raw_methods)
            else str(raw_methods)
        )
        if (
            name in expected
            and str(row.expression) == expected[name]
            and int(row.horizon) == horizon
            and int(row.n_quantiles) == quantiles
            and str(row.return_definition) == RETURN_DEFINITION
            and row_methods == requested_methods
        ):
            completed.add(name)
    return completed


def _append_batch_status(path: Path, record: dict[str, Any]) -> None:
    row = pd.DataFrame([record])
    row.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def _dry_run_payload(
    factors: tuple[Any, ...],
    extra_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    evaluation_symbols = {"c", "o", *(extra_symbols or set())}
    return [
        {
            "number": factor.number,
            "factor_name": factor.name,
            "factor_required_symbols": list(factor.required_symbols),
            "required_symbols": sorted(
                evaluation_symbols | set(factor.required_symbols)
            ),
            "expression": factor.expression,
            "paper_expression": factor.paper_expression,
        }
        for factor in factors
    ]


def main() -> int:
    args = build_parser().parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    registry_file = args.registry_file.expanduser().resolve()
    output_dir_arg = args.output_dir or (
        FACTOR_OUTPUT_DIR / f"alpha101_{args.set}"
    )
    output_dir = output_dir_arg.expanduser().resolve()
    factors = select_registered_factors(
        args.select,
        implementation_set=args.set,
        path=registry_file,
    )
    expected_set = tuple(
        factor
        for factor in load_registered_factors(registry_file)
        if args.set == "runnable"
        or factor.implementation_set == args.set
    )
    if not factors:
        raise ValueError("No executable Alpha101 factors were selected")
    for factor in factors:
        parse_and_validate_expression(factor.expression)
    evaluation_methods = resolve_evaluation_methods(args.methods)
    method_names = evaluation_method_names(evaluation_methods)
    method_data_symbols = evaluation_required_data_symbols(evaluation_methods)

    file_names = _file_names(args)
    _validate_files(
        data_dir,
        factors,
        file_names,
        extra_symbols=method_data_symbols,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "count": len(factors),
                    "set": args.set,
                    "set_size": len(expected_set),
                    "registry_file": str(registry_file),
                    "data_dir": str(data_dir),
                    "output_dir": str(output_dir),
                    "evaluation_methods": list(method_names),
                    "return_definition": RETURN_DEFINITION,
                    "factors": _dry_run_payload(
                        factors,
                        extra_symbols=method_data_symbols,
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = (
        set()
        if args.rerun
        else _matching_completed(
            output_dir / "metrics.csv",
            factors,
            args.horizon,
            args.quantiles,
            method_names,
        )
    )
    pending = tuple(
        factor for factor in factors if factor.name not in completed
    )

    shared_data = None
    if pending and not args.no_cache_data:
        required_symbols = {"c", "o", *method_data_symbols}
        for factor in pending:
            required_symbols.update(
                expression_data_symbols(factor.expression)
            )
        loading_expression = " + ".join(sorted(required_symbols))
        print(
            json.dumps(
                {
                    "state": "loading_data",
                    "symbols": sorted(required_symbols),
                    "data_dir": str(data_dir),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        shared_data = load_market_data(
            data_dir,
            loading_expression,
            file_names,
            extra_symbols=method_data_symbols,
        )

    batch_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    status_path = output_dir / "batch_status.csv"
    success_count = 0
    skipped_count = 0
    failures: list[dict[str, str]] = []

    for position, factor in enumerate(factors, start=1):
        if factor.name in completed:
            skipped_count += 1
            print(
                json.dumps(
                    {
                        "position": position,
                        "total": len(factors),
                        "factor_name": factor.name,
                        "state": "skipped_existing",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        started = time.perf_counter()
        try:
            result = evaluate_factor_expression(
                factor_name=factor.name,
                expression=factor.expression,
                data_dir=data_dir,
                output_dir=output_dir,
                horizon=args.horizon,
                n_quantiles=args.quantiles,
                file_names=file_names,
                preloaded_data=shared_data,
                evaluation_methods=evaluation_methods,
            )
            elapsed = time.perf_counter() - started
            success_count += 1
            metrics = result["metrics"]
            status_record = {
                "batch_id": batch_id,
                "factor_number": factor.number,
                "factor_name": factor.name,
                "state": "success",
                "elapsed_seconds": elapsed,
                "error": "",
                "expression": factor.expression,
            }
            _append_batch_status(status_path, status_record)
            print(
                json.dumps(
                    {
                        "position": position,
                        "total": len(factors),
                        "factor_name": factor.name,
                        "state": "success",
                        "elapsed_seconds": elapsed,
                        "ic_mean": metrics.get("ic_mean"),
                        "ir": metrics.get("ir"),
                        "gn_final_cumulative": metrics.get(
                            "gn_final_cumulative"
                        ),
                    },
                    ensure_ascii=False,
                    allow_nan=True,
                ),
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            failure = {
                "factor_name": factor.name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            _append_batch_status(
                status_path,
                {
                    "batch_id": batch_id,
                    "factor_number": factor.number,
                    "factor_name": factor.name,
                    "state": "failed",
                    "elapsed_seconds": elapsed,
                    "error": failure["error"],
                    "expression": factor.expression,
                },
            )
            print(
                json.dumps(
                    {
                        "position": position,
                        "total": len(factors),
                        "factor_name": factor.name,
                        "state": "failed",
                        "elapsed_seconds": elapsed,
                        "error": failure["error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.stop_on_error:
                break
        finally:
            gc.collect()

    summary = {
        "batch_id": batch_id,
        "selected": len(factors),
        "success": success_count,
        "skipped_existing": skipped_count,
        "failed": len(failures),
        "evaluation_methods": list(method_names),
        "return_definition": RETURN_DEFINITION,
        "failures": failures,
        "metrics": str(output_dir / "metrics.csv"),
        "batch_status": str(status_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
