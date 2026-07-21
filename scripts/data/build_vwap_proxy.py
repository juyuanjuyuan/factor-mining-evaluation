#!/usr/bin/env python3
"""Build the explicit Alpha101 VWAP proxy: (adjusted high + adjusted low) / 2."""

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
    parser.add_argument("--high-file", default="high_df.pq")
    parser.add_argument("--low-file", default="low_df.pq")
    parser.add_argument("--output-file", default="vwap_proxy_df.pq")
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=DATA_MANIFEST_DIR / "vwap_proxy_manifest.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.data_dir.expanduser().resolve()
    high_path = root / args.high_file
    low_path = root / args.low_file
    high = pd.read_parquet(high_path)
    low = pd.read_parquet(low_path)
    if not high.index.equals(low.index) or not high.columns.equals(low.columns):
        raise ValueError("high and low matrices must have identical axes")
    if high.empty or high.index.has_duplicates or high.columns.has_duplicates:
        raise ValueError("high/low matrices must be nonempty with unique axes")

    proxy = (high + low) / 2.0
    proxy.index.name = high.index.name
    output = root / args.output_file
    temporary = output.with_suffix(output.suffix + ".tmp")
    proxy.to_parquet(temporary)
    temporary.replace(output)

    values = proxy.to_numpy(dtype=float, copy=False)
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output": str(output),
        "formula": "(high_df + low_df) / 2",
        "is_proxy": True,
        "source_high": str(high_path),
        "source_low": str(low_path),
        "shape": list(proxy.shape),
        "start_date": str(proxy.index.min()),
        "end_date": str(proxy.index.max()),
        "non_null_cells": int(np.isfinite(values).sum()),
        "total_cells": int(values.size),
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
