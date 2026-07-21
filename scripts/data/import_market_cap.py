#!/usr/bin/env python3
"""Convert the DataYes market-cap long table to the canonical wide matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from paths import DATA_DIR, DATA_MANIFEST_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--source-file", default="通联-股票市值.pq")
    parser.add_argument("--close-file", default="close_df.pq")
    parser.add_argument("--output-file", default="market_cap_df.pq")
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=DATA_MANIFEST_DIR / "market_cap_manifest.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.data_dir.expanduser().resolve()
    source_path = root / args.source_file
    close_path = root / args.close_file
    source = pd.read_parquet(source_path)
    required = {"day", "code", "values"}
    if not required.issubset(source.columns):
        raise ValueError(
            f"{source_path} must contain columns {sorted(required)}"
        )
    source = source.loc[:, ["day", "code", "values"]].copy()
    source["day"] = pd.to_datetime(source["day"])
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["values"] = pd.to_numeric(source["values"], errors="raise")
    if source.duplicated(["day", "code"]).any():
        raise ValueError("market-cap source has duplicate day/code rows")
    finite = source["values"].to_numpy(dtype=float, copy=False)
    if np.isinf(finite).any() or np.nanmin(finite) < 0:
        raise ValueError("market-cap values must be finite-or-null and nonnegative")

    close = pd.read_parquet(close_path)
    if close.empty or close.index.has_duplicates or close.columns.has_duplicates:
        raise ValueError("close matrix must be nonempty with unique axes")
    wide = source.pivot(index="day", columns="code", values="values")
    wide = wide.reindex(index=close.index, columns=close.columns)
    wide.index.name = close.index.name
    wide.columns.name = close.columns.name

    output_path = root / args.output_file
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    wide.to_parquet(temporary)
    temporary.replace(output_path)

    values = wide.to_numpy(dtype=float, copy=False)
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output": str(output_path),
        "source": str(source_path),
        "source_format": "long(day, code, values)",
        "meaning": "total market capitalization",
        "shape": list(wide.shape),
        "start_date": str(wide.index.min()),
        "end_date": str(wide.index.max()),
        "source_start_date": str(source["day"].min()),
        "source_end_date": str(source["day"].max()),
        "non_null_cells": int(np.isfinite(values).sum()),
        "total_cells": int(values.size),
        "coverage": float(np.isfinite(values).mean()),
    }
    manifest_path = args.manifest_file.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
