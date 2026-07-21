#!/usr/bin/env python3
"""Normalize all wide market matrices to the close matrix's exact axes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from paths import DATA_DIR


DEFAULT_FILES = (
    "open_df.pq",
    "high_df.pq",
    "low_df.pq",
    "volume_df.pq",
    "amount_df.pq",
    "vwap_proxy_df.pq",
    "market_cap_df.pq",
    "limit_ratio_df.pq",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--close-file", default="close_df.pq")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.data_dir.expanduser().resolve()
    close = pd.read_parquet(root / args.close_file)
    if close.empty or close.index.has_duplicates or close.columns.has_duplicates:
        raise ValueError("close matrix must be nonempty with unique axes")

    expected_index = close.index
    expected_columns = close.columns
    for filename in args.files:
        path = root / filename
        frame = pd.read_parquet(path)
        if set(frame.index) != set(expected_index):
            missing = len(set(expected_index) - set(frame.index))
            extra = len(set(frame.index) - set(expected_index))
            raise ValueError(
                f"{filename}: date set differs from close "
                f"(missing={missing}, extra={extra})"
            )
        if set(frame.columns) != set(expected_columns):
            missing = len(set(expected_columns) - set(frame.columns))
            extra = len(set(frame.columns) - set(expected_columns))
            raise ValueError(
                f"{filename}: security set differs from close "
                f"(missing={missing}, extra={extra})"
            )

        normalized = frame.reindex(
            index=expected_index,
            columns=expected_columns,
        )
        if normalized.index.equals(frame.index) and normalized.columns.equals(
            frame.columns
        ):
            print(f"{filename}: already normalized")
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        normalized.to_parquet(temporary)
        temporary.replace(path)
        print(f"{filename}: normalized to {normalized.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
