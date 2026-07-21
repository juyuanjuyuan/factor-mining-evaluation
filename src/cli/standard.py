#!/usr/bin/env python3
"""CLI for the two code-owned Webapp evaluation standards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from engine import DEFAULT_FILES
from evaluation_standards import (
    REGISTERED_EVALUATION_STANDARDS,
    evaluate_factor_standards,
    evaluation_standard_names,
)
from genetic_mining.admission import admit_factor_to_library
from paths import DATA_DIR, FACTOR_OUTPUT_DIR, OUTPUT_DIR, PROJECT_ROOT


def _write_json(path: Path, payload: dict) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="print standards and their exact method order")
    run = subparsers.add_parser("run", help="evaluate one frozen expression on a test set")
    run.add_argument("--factor-name", required=True)
    run.add_argument("--expression", required=True)
    run.add_argument("--test-start", required=True)
    run.add_argument("--test-end", required=True)
    run.add_argument("--standards", default="all", help="all, ic_test, profitability_test")
    run.add_argument("--data-dir", type=Path, default=DATA_DIR)
    run.add_argument(
        "--output-dir",
        type=Path,
        default=FACTOR_OUTPUT_DIR / "standards",
    )
    run.add_argument("--horizon", type=int, default=1)
    run.add_argument("--quantiles", type=int, default=10)
    run.add_argument("--significance-level", type=float, default=0.05)
    run.add_argument("--minimum-ic-mean", type=float, default=0.0)
    run.add_argument("--minimum-sharpe-60-median", type=float, default=1.0)
    run.add_argument("--minimum-annualized-return", type=float, default=0.30)
    run.add_argument("--admit-if-passed", action="store_true")
    run.add_argument(
        "--library-file",
        type=Path,
        default=PROJECT_ROOT / "factor_registry" / "webapp_factor_library.json",
    )
    run.add_argument(
        "--correlation-state-dir",
        type=Path,
        default=OUTPUT_DIR / "webapp",
    )
    run.add_argument("--correlation-threshold", type=float, default=0.75)
    run.add_argument("--project", default="遗传规划")
    run.add_argument("--close-file", default=DEFAULT_FILES["c"])
    run.add_argument("--open-file", default=DEFAULT_FILES["o"])
    run.add_argument("--high-file", default=DEFAULT_FILES["h"])
    run.add_argument("--low-file", default=DEFAULT_FILES["l"])
    run.add_argument("--volume-file", default=DEFAULT_FILES["vol"])
    run.add_argument("--amount-file", default=DEFAULT_FILES["amt"])
    run.add_argument("--vwap-file", default=DEFAULT_FILES["vwap"])
    run.add_argument("--market-cap-file", default=DEFAULT_FILES["cap"])
    run.add_argument("--limit-ratio-file", default=DEFAULT_FILES["limit"])
    run.add_argument("--st-status-file", default=DEFAULT_FILES["st"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list":
        print(
            json.dumps(
                [
                    {
                        "name": name,
                        "label": REGISTERED_EVALUATION_STANDARDS[name].label,
                        "methods": list(
                            REGISTERED_EVALUATION_STANDARDS[name].method_names
                        ),
                        "description": REGISTERED_EVALUATION_STANDARDS[
                            name
                        ].description,
                    }
                    for name in evaluation_standard_names()
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    file_names = {
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
    result = evaluate_factor_standards(
        factor_name=args.factor_name,
        expression=args.expression,
        data_dir=args.data_dir,
        output_dir=args.output_dir / args.factor_name,
        signal_start=args.test_start,
        signal_end=args.test_end,
        standards=args.standards,
        horizon=args.horizon,
        n_quantiles=args.quantiles,
        file_names=file_names,
        significance_level=args.significance_level,
        minimum_ic_mean=args.minimum_ic_mean,
        minimum_rolling_sharpe_60_median=args.minimum_sharpe_60_median,
        minimum_annualized_return=args.minimum_annualized_return,
    )
    if args.admit_if_passed:
        if set(result["standards"]) != set(evaluation_standard_names()):
            raise ValueError("Admission requires both ic_test and profitability_test")
        if result["overall_passed"]:
            result["admission"] = admit_factor_to_library(
                factor_name=args.factor_name,
                expression=args.expression,
                library_file=args.library_file,
                data_dir=args.data_dir,
                state_dir=args.correlation_state_dir,
                project=args.project,
                source_batch_id="evaluation_standard_cli",
                correlation_threshold=args.correlation_threshold,
            ).as_dict()
        else:
            result["admission"] = {
                "admitted": False,
                "explanation": "IC/盈利能力评价标准未全部通过，未执行相关性准入",
            }
        admission_path = args.output_dir / args.factor_name / "admission_result.json"
        _write_json(admission_path, result["admission"])
        result["admission_result_path"] = str(admission_path.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))
    admission = result.get("admission")
    if args.admit_if_passed:
        return 0 if admission and (
            admission.get("admitted") or admission.get("already_present")
        ) else 2
    return 0 if result["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
