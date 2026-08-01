#!/usr/bin/env python3
"""Repair the current ST tail from boundary state, limits, and new events."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import pandas as pd

import _bootstrap  # noqa: F401
from paths import DATA_DIR
from transforms.price_limits import reconcile_incremental_st_status


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the repaired st_status_df.pq; otherwise only report diagnostics.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    status_path = data_dir / "st_status_df.pq"
    market_path = data_dir / "company_market_daily.parquet"
    security_path = data_dir / "security_name_reference.parquet"

    historical = pd.read_parquet(status_path)
    market = pd.read_parquet(market_path).copy()
    market["date"] = pd.to_datetime(market["trade_date"])
    market["code"] = market["security_code"].astype("string").str.zfill(6)
    repaired, diagnostics = reconcile_incremental_st_status(historical, market)

    latest_day = pd.Timestamp(repaired["day"].max())
    latest = repaired[repaired["day"] == latest_day].copy()
    if security_path.is_file():
        security = pd.read_parquet(
            security_path,
            columns=["security_code", "security_name", "name_observed_at"],
        )
        security["security_code"] = (
            security["security_code"].astype("string").str.zfill(6)
        )
        names = latest.merge(
            security,
            left_on="code",
            right_on="security_code",
            how="left",
            validate="one_to_one",
        )
        name_is_st = names["security_name"].fillna("").str.match(
            r"^(?:\*?ST|S)",
            case=False,
        )
        false_negatives = names[name_is_st & ~names["是否st"]]
        if not false_negatives.empty:
            sample = false_negatives[["code", "security_name"]].head(20).to_dict("records")
            raise ValueError(
                "Repaired latest ST state contradicts security names: "
                f"{sample}"
            )
        diagnostics["latest_name_st_count"] = int(name_is_st.sum())
        diagnostics["latest_name_st_false_negatives"] = 0

    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if not args.apply:
        print("dry run only; pass --apply to write st_status_df.pq")
        return 0
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    backup_path = manifest_dir / "st_status_df_before_20260731_repair.pq"
    if not backup_path.exists():
        shutil.copy2(status_path, backup_path)
    _atomic_parquet(repaired, status_path)
    repair_manifest = {
        "repaired_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status_path": str(status_path),
        "backup_path": str(backup_path),
        "market_source": str(market_path),
        **diagnostics,
    }
    manifest_path = manifest_dir / "st_status_tail_repair_manifest.json"
    manifest_path.write_text(
        json.dumps(repair_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {status_path} ({len(repaired)} rows)")
    print(f"backup {backup_path}")
    print(f"manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
