#!/usr/bin/env python3
"""Build the wide A-share price-limit-ratio matrix used by tradability checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transforms.price_limits import infer_price_limit_ratio_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close-file", default=ROOT / "data" / "close_df.pq")
    parser.add_argument(
        "--output-file",
        default=ROOT / "data" / "limit_ratio_df.pq",
    )
    parser.add_argument(
        "--no-limit-first-n",
        type=int,
        default=5,
        help="valid trading days to mark as no-limit for stocks first appearing after dataset start",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    close = pd.read_parquet(args.close_file).sort_index()
    limit_ratio = infer_price_limit_ratio_frame(
        close,
        no_limit_first_n=args.no_limit_first_n,
    )
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    limit_ratio.to_parquet(output)
    counts = limit_ratio.stack(future_stack=True).dropna().value_counts().sort_index()
    print(f"wrote {output}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
