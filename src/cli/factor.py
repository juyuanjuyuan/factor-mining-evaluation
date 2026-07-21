#!/usr/bin/env python3
"""CLI entry point for one archived factor evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from engine import DEFAULT_FILES, evaluate_factor_expression, result_as_json
from evaluators import (
    available_evaluation_methods,
    resolve_evaluation_methods,
)
from paths import DATA_DIR, FACTOR_OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one wide-matrix factor with selected modular methods, then "
            "archive their metrics, details, artifacts, and editable test code."
        )
    )
    parser.add_argument("--factor-name", required=True)
    parser.add_argument("--expression", required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FACTOR_OUTPUT_DIR / "custom",
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


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_factor_expression(
        factor_name=args.factor_name,
        expression=args.expression,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        horizon=args.horizon,
        n_quantiles=args.quantiles,
        evaluation_methods=resolve_evaluation_methods(args.methods),
        file_names={
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
        },
    )
    print(result_as_json(result))


if __name__ == "__main__":
    main()
