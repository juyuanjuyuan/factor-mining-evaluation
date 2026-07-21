#!/usr/bin/env python3
"""Merge company daily-market increments into the project's market matrices.

The incoming files are long-form monthly Parquet extracts.  The latest extract
is authoritative when it conflicts with an existing value.  Prices are stored
on the incoming backward-adjusted basis: historical OHLC values for each
overlapping security are rebased so that the existing final close equals the
incoming first-day previous close.  This preserves price continuity while
letting the latest corporate-action adjustment win.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from paths import DATA_DIR, PROJECT_ROOT
from transforms.price_limits import infer_price_limit_ratio_frame


REQUIRED_COLUMNS = {
    "trade_date",
    "security_code",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "pre_close_raw",
    "volume",
    "amount",
    "total_market_cap",
    "float_market_cap",
    "backward_adj_factor",
    "limit_up_price",
    "limit_down_price",
    "trading_status",
    "is_suspended",
    "is_st",
}
PRICE_COLUMNS = {
    "open_df.pq": "open_raw",
    "high_df.pq": "high_raw",
    "low_df.pq": "low_raw",
    "close_df.pq": "close_raw",
}
DIRECT_COLUMNS = {
    "volume_df.pq": "volume",
    "amount_df.pq": "amount",
    "market_cap_df.pq": "total_market_cap",
}
EXTRA_WIDE_COLUMNS = {
    "float_market_cap_df.pq": "float_market_cap",
    "limit_up_price_df.pq": "limit_up_price",
    "limit_down_price_df.pq": "limit_down_price",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "tmp")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the merge results. Without this flag, validate and report only.",
    )
    return parser.parse_args()


def read_increment(source_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files found in {source_dir}")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"incoming data is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"]).dt.tz_localize(None).dt.normalize()
    frame["code"] = frame["security_code"].astype(str).str.zfill(6)
    if frame[["date", "code"]].isna().any().any():
        raise ValueError("incoming trade_date/security_code contains missing values")
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("incoming data has duplicate trade_date/security_code rows")
    return frame.sort_values(["date", "code"]).reset_index(drop=True), files


def wide_from_increment(
    incoming: pd.DataFrame,
    column: str,
    *,
    dates: pd.DatetimeIndex,
    codes: pd.Index,
    price_scaled: bool = False,
    scale: pd.Series | None = None,
    suspended_to_nan: bool = False,
) -> pd.DataFrame:
    values = incoming[column].astype(float).copy()
    if price_scaled:
        values = values * incoming["backward_adj_factor"].astype(float)
        if scale is not None:
            values = values * incoming["code"].map(scale).fillna(1.0)
    if suspended_to_nan:
        values = values.mask(incoming["is_suspended"].astype(bool))
    frame = pd.DataFrame({"date": incoming["date"], "code": incoming["code"], "value": values})
    return frame.pivot(index="date", columns="code", values="value").reindex(
        index=dates,
        columns=codes,
    )


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    incoming, source_files = read_increment(source_dir)

    existing_close = pd.read_parquet(data_dir / "close_df.pq")
    last_existing_date = pd.Timestamp(existing_close.index.max()).normalize()
    incoming_dates = pd.DatetimeIndex(sorted(incoming["date"].unique()))
    if incoming_dates.min() <= last_existing_date:
        raise ValueError(
            "incoming dates overlap existing close data; this one-way append script refuses "
            f"to overwrite them ({incoming_dates.min().date()} <= {last_existing_date.date()})"
        )

    incoming_codes = pd.Index(sorted(incoming["code"].unique()))
    output_codes = existing_close.columns.union(incoming_codes, sort=False)
    output_dates = existing_close.index.append(incoming_dates)
    if output_dates.has_duplicates:
        raise ValueError("merged date index would contain duplicates")

    first_day = incoming[incoming["date"] == incoming_dates.min()].set_index("code")
    common_codes = existing_close.columns.intersection(first_day.index)
    old_boundary = existing_close.loc[last_existing_date, common_codes].astype(float)
    new_previous_close = (
        first_day.loc[common_codes, "pre_close_raw"].astype(float)
        * first_day.loc[common_codes, "backward_adj_factor"].astype(float)
    )
    valid_scale = old_boundary.notna() & new_previous_close.notna() & (old_boundary > 0)
    rebase_scale = (new_previous_close[valid_scale] / old_boundary[valid_scale]).rename("scale")

    suspended_rows = int(incoming["is_suspended"].astype(bool).sum())
    summary = {
        "source_files": [str(path) for path in source_files],
        "incoming_rows": int(len(incoming)),
        "incoming_dates": [str(date.date()) for date in incoming_dates],
        "incoming_codes": int(len(incoming_codes)),
        "existing_last_date": str(last_existing_date.date()),
        "output_shape": [int(len(output_dates)), int(len(output_codes))],
        "new_security_codes": output_codes.difference(existing_close.columns).tolist(),
        "old_security_codes_missing_from_increment": int(
            len(existing_close.columns.difference(incoming_codes))
        ),
        "rebased_security_codes": int(len(rebase_scale)),
        "rebase_scale_min": float(rebase_scale.min()),
        "rebase_scale_max": float(rebase_scale.max()),
        "suspended_rows_normalized_to_nan": suspended_rows,
        "st_conflicts_resolved_by_latest_extract": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.apply:
        print("dry run only; pass --apply to write files")
        return 0

    outputs: dict[str, pd.DataFrame] = {}
    for filename, source_column in PRICE_COLUMNS.items():
        historical = pd.read_parquet(data_dir / filename).reindex(
            index=existing_close.index,
            columns=output_codes,
        )
        historical.loc[:, rebase_scale.index] = historical.loc[:, rebase_scale.index].mul(
            rebase_scale,
            axis="columns",
        )
        appended = wide_from_increment(
            incoming,
            source_column,
            dates=incoming_dates,
            codes=output_codes,
            price_scaled=True,
            scale=rebase_scale,
            suspended_to_nan=True,
        )
        outputs[filename] = pd.concat([historical, appended])

    for filename, source_column in DIRECT_COLUMNS.items():
        historical = pd.read_parquet(data_dir / filename).reindex(
            index=existing_close.index,
            columns=output_codes,
        )
        appended = wide_from_increment(
            incoming,
            source_column,
            dates=incoming_dates,
            codes=output_codes,
            suspended_to_nan=filename in {"volume_df.pq", "amount_df.pq"},
        )
        outputs[filename] = pd.concat([historical, appended])

    outputs["vwap_proxy_df.pq"] = (outputs["high_df.pq"] + outputs["low_df.pq"]) / 2
    outputs["limit_ratio_df.pq"] = infer_price_limit_ratio_frame(outputs["close_df.pq"])
    for filename, source_column in EXTRA_WIDE_COLUMNS.items():
        historical = pd.DataFrame(np.nan, index=existing_close.index, columns=output_codes)
        appended = wide_from_increment(
            incoming,
            source_column,
            dates=incoming_dates,
            codes=output_codes,
            price_scaled=filename.startswith("limit_"),
            scale=rebase_scale,
        )
        outputs[filename] = pd.concat([historical, appended])

    # The project consumes ST data as a long table.  The new extract is complete
    # for its dates, so remove old rows from the first incoming date onward and
    # replace them with the latest source, including explicit False values.
    st_path = data_dir / "st_status_df.pq"
    old_st = pd.read_parquet(st_path).copy()
    old_st["day"] = pd.to_datetime(old_st["day"]).dt.tz_localize(None).dt.normalize()
    old_st["code"] = old_st["code"].astype(str).str.zfill(6)
    old_st = old_st[old_st["day"] < incoming_dates.min()]
    new_st = pd.DataFrame(
        {
            "day": incoming["date"],
            "code": incoming["code"],
            "是否st": incoming["is_st"].astype(bool),
        }
    )
    outputs["st_status_df.pq"] = pd.concat([old_st, new_st], ignore_index=True).sort_values(
        ["day", "code"]
    )

    # Preserve every incoming source field, including raw prices, exchange,
    # adjustment factors, statuses and ST-effective metadata.
    archive_path = data_dir / "company_market_daily.parquet"
    write_parquet_atomic(incoming.drop(columns=["date", "code"]), archive_path)

    for filename, frame in outputs.items():
        write_parquet_atomic(frame, data_dir / filename)

    manifest_path = data_dir / "manifests" / "company_market_merge_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(outputs)} normalized data files and {archive_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
