#!/usr/bin/env python3
"""Build a wide volume matrix from the validated DataYes equity source.

The output is aligned exactly to a reference close matrix: same dates, same
security columns, and NaN where the DataYes source has no corresponding row.
No price-derived or amount-derived volume proxy is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-python-root", type=Path, required=True)
    parser.add_argument("--reference-close", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    return value


def main() -> int:
    args = build_parser().parse_args()
    source_python_root = args.source_python_root.expanduser().resolve()
    if not source_python_root.is_dir():
        raise FileNotFoundError(
            f"DataYes Python source root does not exist: {source_python_root}"
        )
    sys.path.insert(0, str(source_python_root))
    from turtle_system.data.datayes import DataYesEquityDataSource

    close_path = args.reference_close.expanduser().resolve()
    close = pd.read_parquet(close_path).sort_index()
    if close.empty or close.index.has_duplicates or close.columns.has_duplicates:
        raise ValueError("reference close matrix must be nonempty with unique axes")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close.columns = close.columns.astype(str).str.zfill(6)

    row_by_day = {timestamp.date(): row for row, timestamp in enumerate(close.index)}
    column_by_ticker = {
        ticker: column for column, ticker in enumerate(close.columns)
    }
    values = np.full(close.shape, np.nan, dtype=np.float64)

    source = DataYesEquityDataSource(
        args.source_root.expanduser().resolve(),
        conflict_policy="quarantine",
    )
    last_file: str | None = None
    matched_records = 0
    duplicate_cells = 0
    for record in source.iter_records(
        start_date=close.index.min().date(),
        end_date=close.index.max().date(),
        universe=tuple(close.columns),
    ):
        if record.source.file != last_file:
            last_file = record.source.file
            print(f"reading {last_file}", flush=True)
        row = row_by_day.get(record.trade_time.date())
        column = column_by_ticker.get(record.ticker)
        if row is None or column is None:
            continue
        existing = values[row, column]
        if np.isfinite(existing):
            duplicate_cells += 1
            if existing != record.volume:
                raise ValueError(
                    "conflicting volume for "
                    f"{record.trade_time.date()} {record.ticker}: "
                    f"{existing} != {record.volume}"
                )
            continue
        values[row, column] = float(record.volume)
        matched_records += 1

    volume = pd.DataFrame(values, index=close.index, columns=close.columns)
    volume.index.name = close.index.name or "TRADE_DATE"
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    volume.to_parquet(output_path)

    close_valid = close.notna().to_numpy()
    volume_valid = np.isfinite(values)
    overlap = close_valid & volume_valid
    source_report = source.quality_report.to_dict()
    manifest = {
        "source_root": str(source.root.resolve()),
        "source_field": "TURNOVER_VOL",
        "source_unit": (
            "native DataYes turnover-volume unit; no multiplier or price "
            "conversion applied"
        ),
        "reference_close": str(close_path),
        "output": str(output_path),
        "shape": list(volume.shape),
        "start_date": close.index.min(),
        "end_date": close.index.max(),
        "first_volume_date": (
            volume_valid.any(axis=1).nonzero()[0][0]
            if volume_valid.any()
            else None
        ),
        "last_volume_date": (
            close.index[volume_valid.any(axis=1).nonzero()[0][-1]]
            if volume_valid.any()
            else None
        ),
        "matched_records": matched_records,
        "duplicate_cells": duplicate_cells,
        "close_non_null_cells": int(close_valid.sum()),
        "volume_non_null_cells": int(volume_valid.sum()),
        "overlap_non_null_cells": int(overlap.sum()),
        "close_cells_missing_volume": int((close_valid & ~volume_valid).sum()),
        "overlap_ratio": (
            float(overlap.sum() / close_valid.sum()) if close_valid.any() else None
        ),
        "source_quality": source_report,
    }
    # Correct the first date after using its integer position above.
    if volume_valid.any():
        first_position = volume_valid.any(axis=1).nonzero()[0][0]
        manifest["first_volume_date"] = close.index[first_position]

    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_value,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "manifest": str(manifest_path),
                "shape": list(volume.shape),
                "matched_records": matched_records,
                "overlap_ratio": manifest["overlap_ratio"],
                "last_volume_date": _json_value(manifest["last_volume_date"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
