#!/usr/bin/env python3
"""Validate and import audited company ST/delisting status into ``data/``.

The company download is event-derived. Its ST output already ends on formal
withdrawal events. Completed delisting-consolidation intervals are copied as
published; an explicit state-6 start without a state-7 end is an open active
interval and is extended only to the downloaded/close-axis cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd

import _bootstrap  # noqa: F401
from paths import DATA_DIR, PROJECT_ROOT


DEFAULT_SOURCE_DIR = Path(
    "/Users/huangjuyuan/Desktop/database_summerintern/outputs/a_share_special_status"
)
ST_FILE = "st_status_df.pq"
DELISTING_FILE = "delisting_period_status_df.pq"
RAW_FILE = "special_status_raw.parquet"
SOURCE_MANIFEST_FILE = "manifest.json"
IMPORT_MANIFEST_FILE = "company_special_status_import_manifest.json"
ST_BACKUP_FILE = "st_status_df_before_20260803_company_status_import.pq"
DELISTING_START_CODE = 6
DELISTING_END_CODE = 7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_true_status(
    frame: pd.DataFrame,
    *,
    value_column: str,
    close_index: pd.DatetimeIndex,
    close_codes: pd.Index,
) -> pd.DataFrame:
    required = {"day", "code", value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{value_column} input is missing columns: {sorted(missing)}")
    result = frame.loc[:, ["day", "code", value_column]].copy()
    result["day"] = pd.to_datetime(result["day"], errors="raise").dt.normalize()
    codes = result["code"].astype("string").str.strip()
    if codes.isna().any() or not codes.str.fullmatch(r"\d{1,6}").all():
        raise ValueError(f"{value_column} contains invalid security codes")
    result["code"] = codes.str.zfill(6)
    try:
        values = result[value_column].astype("boolean")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{value_column} canonical input must contain boolean true rows only"
        ) from exc
    if values.isna().any() or not bool(values.eq(True).all()):
        raise ValueError(f"{value_column} canonical input must contain true rows only")
    result[value_column] = values.astype(bool)
    if result.duplicated(["day", "code"]).any():
        raise ValueError(f"{value_column} contains duplicate (day, code) rows")
    invalid_days = result.loc[~result["day"].isin(close_index), "day"]
    invalid_codes = result.loc[~result["code"].isin(close_codes), "code"]
    if len(invalid_days) or len(invalid_codes):
        raise ValueError(
            f"{value_column} is outside close axes: "
            f"days={invalid_days.head().tolist()}, codes={invalid_codes.head().tolist()}"
        )
    return result.sort_values(["day", "code"], kind="stable").reset_index(drop=True)


def extend_open_delisting_intervals(
    delisting: pd.DataFrame,
    raw_events: pd.DataFrame,
    *,
    close_index: pd.DatetimeIndex,
    close_codes: pd.Index,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Extend explicit unmatched state-6 events through the data cutoff."""

    required = ("security_code", "raw_status_code", "effective_start")
    missing = set(required).difference(raw_events.columns)
    if missing:
        raise ValueError(f"Raw status events are missing columns: {sorted(missing)}")
    events = raw_events.loc[:, list(required)].copy()
    events["security_code"] = (
        events["security_code"].astype("string").str.strip().str.zfill(6)
    )
    events["raw_status_code"] = pd.to_numeric(
        events["raw_status_code"], errors="raise"
    ).astype(int)
    events["effective_start"] = pd.to_datetime(
        events["effective_start"], errors="raise"
    ).dt.normalize()
    cutoff = close_index.max()
    additions: list[pd.DataFrame] = []
    open_intervals: list[dict[str, object]] = []
    for code, group in events.groupby("security_code", sort=True):
        active_start: pd.Timestamp | None = None
        for row in group.sort_values(
            ["effective_start", "raw_status_code"], kind="stable"
        ).itertuples(index=False):
            state = int(row.raw_status_code)
            day = pd.Timestamp(row.effective_start)
            if state == DELISTING_START_CODE:
                if active_start is not None:
                    raise ValueError(f"{code} has overlapping open delisting starts")
                active_start = day
            elif state == DELISTING_END_CODE:
                if active_start is None:
                    raise ValueError(f"{code} has state 7 without an active state 6")
                active_start = None
            elif state == 5 and active_start is not None:
                raise ValueError(f"{code} delisted without a closing state-7 event")
        if active_start is None or active_start > cutoff:
            continue
        if code not in close_codes:
            raise ValueError(f"Open delisting code {code} is missing from close columns")
        days = close_index[(close_index >= active_start) & (close_index <= cutoff)]
        additions.append(
            pd.DataFrame(
                {"day": days, "code": code, "is_delisting_period": True}
            )
        )
        open_intervals.append(
            {
                "code": code,
                "effective_start": active_start.date().isoformat(),
                "extended_through": cutoff.date().isoformat(),
                "trading_day_count": int(len(days)),
                "basis": "explicit PARTY_STATE=6 with no PARTY_STATE=7 by data cutoff",
            }
        )
    if additions:
        delisting = pd.concat([delisting, *additions], ignore_index=True)
    if delisting.duplicated(["day", "code"]).any():
        sample = delisting.loc[
            delisting.duplicated(["day", "code"], keep=False)
        ].head()
        raise ValueError(
            f"Completed and open delisting periods overlap: {sample.to_dict('records')}"
        )
    return (
        delisting.sort_values(["day", "code"], kind="stable").reset_index(drop=True),
        open_intervals,
    )


def prepare_import(
    source_dir: Path,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source_paths = {
        "st": source_dir / ST_FILE,
        "delisting": source_dir / DELISTING_FILE,
        "raw": source_dir / RAW_FILE,
        "manifest": source_dir / SOURCE_MANIFEST_FILE,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing company status outputs: {missing}")
    source_manifest = json.loads(
        source_paths["manifest"].read_text(encoding="utf-8")
    )
    for key in ("st", "delisting", "raw"):
        expected = source_manifest.get("files", {}).get(key, {}).get("sha256")
        actual = _sha256(source_paths[key])
        if not expected or actual != expected:
            raise ValueError(
                f"Source {key} SHA256 mismatch: expected={expected}, actual={actual}"
            )

    close = pd.read_parquet(data_dir / "close_df.pq")
    close_index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    close_codes = pd.Index(close.columns.astype("string").str.strip()).str.zfill(6)
    if source_manifest.get("query_interval", {}).get("end") != cutoff_string(close_index):
        raise ValueError("Company status query end must match close_df.pq cutoff")

    raw = pd.read_parquet(source_paths["raw"])
    st = _normalize_true_status(
        pd.read_parquet(source_paths["st"]),
        value_column="是否st",
        close_index=close_index,
        close_codes=close_codes,
    )
    delisting = _normalize_true_status(
        pd.read_parquet(source_paths["delisting"]),
        value_column="is_delisting_period",
        close_index=close_index,
        close_codes=close_codes,
    )
    delisting, open_intervals = extend_open_delisting_intervals(
        delisting,
        raw,
        close_index=close_index,
        close_codes=close_codes,
    )

    diagnostics: dict[str, object] = {
        "source_dir": str(source_dir),
        "source_manifest_sha256": _sha256(source_paths["manifest"]),
        "source_files": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in source_paths.items()
        },
        "close_start": close_index.min().date().isoformat(),
        "close_end": close_index.max().date().isoformat(),
        "st_rows": int(len(st)),
        "st_securities": int(st["code"].nunique()),
        "delisting_rows": int(len(delisting)),
        "delisting_securities": int(delisting["code"].nunique()),
        "open_delisting_intervals": open_intervals,
        "status_semantics": {
            "st": "formal ST/*ST state is true from implementation through the day before the next state event",
            "delisting": "formal delisting-consolidation state 6 through state 7 inclusive; an open state 6 extends only through this data cutoff",
        },
    }
    return st, delisting, raw, diagnostics


def cutoff_string(index: pd.DatetimeIndex) -> str:
    return index.max().date().isoformat()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Update project data using an atomic replace for each file; "
            "otherwise only validate and report."
        ),
    )
    args = parser.parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    st, delisting, raw, diagnostics = prepare_import(source_dir, data_dir)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if not args.apply:
        print("dry run only; pass --apply to update project data")
        return 0

    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    st_path = data_dir / ST_FILE
    st_backup = manifest_dir / ST_BACKUP_FILE
    if st_path.is_file() and not st_backup.exists():
        shutil.copy2(st_path, st_backup)
    _atomic_parquet(st, st_path)
    _atomic_parquet(delisting, data_dir / DELISTING_FILE)
    _atomic_parquet(raw, data_dir / RAW_FILE)

    diagnostics.update(
        {
            "imported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "project_root": str(PROJECT_ROOT),
            "st_backup": str(st_backup) if st_backup.is_file() else None,
            "written_files": {
                ST_FILE: _sha256(st_path),
                DELISTING_FILE: _sha256(data_dir / DELISTING_FILE),
                RAW_FILE: _sha256(data_dir / RAW_FILE),
            },
        }
    )
    manifest_path = manifest_dir / IMPORT_MANIFEST_FILE
    temporary_manifest = manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(f"updated {st_path}")
    print(f"wrote {data_dir / DELISTING_FILE}")
    print(f"wrote {data_dir / RAW_FILE}")
    print(f"manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
