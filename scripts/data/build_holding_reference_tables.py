#!/usr/bin/env python3
"""Build compact company and industry name tables for holdings display.

The security source is a point-in-time master snapshot.  It covers currently
listed and delisted securities, but its short name is the current or last-known
name rather than a complete daily rename history.  That basis is persisted
explicitly so the webapp never presents it as a signal-day name.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECURITY_MASTER = Path(
    "/Users/huangjuyuan/Desktop/database_summerintern/"
    "outputs/data_update_retry/security_master.parquet"
)
SECURITY_OUTPUT = "security_name_reference.parquet"
INDUSTRY_OUTPUT = "industry_l1_name_reference.parquet"
MANIFEST_OUTPUT = "holding_reference_manifest.json"

INDUSTRY_L1_NAMES = {
    "01031701": "石油石化",
    "01031702": "煤炭",
    "01031703": "有色金属",
    "01031704": "电力及公用事业",
    "01031705": "钢铁",
    "01031706": "基础化工",
    "01031707": "建筑",
    "01031708": "建材",
    "01031709": "轻工制造",
    "01031710": "机械",
    "01031711": "电力设备及新能源",
    "01031712": "国防军工",
    "01031713": "汽车",
    "01031714": "商贸零售",
    "01031715": "消费者服务",
    "01031716": "家电",
    "01031717": "纺织服装",
    "01031718": "医药",
    "01031719": "食品饮料",
    "01031720": "农林牧渔",
    "01031721": "银行",
    "01031722": "非银行金融",
    "01031723": "房地产",
    "01031724": "交通运输",
    "01031725": "电子",
    "01031726": "通信",
    "01031727": "计算机",
    "01031728": "传媒",
    "01031729": "综合",
    "01031730": "综合金融",
}

# The local price history uses 300114 through 2025-02-12 and 302132 from
# 2025-02-17.  The current security master contains only the replacement code.
# Shenzhen Stock Exchange announcement 2025-025 documents the migration.
LEGACY_SECURITY_ALIASES = (
    {
        "current_code": "302132",
        "security_code": "300114",
        "security_name": "中航电测",
        "delisting_date": "2025-02-12",
        "listing_status": "CODE_CHANGED",
        "name_observed_at": "2025-02-12",
        "name_basis": "historical_code_alias",
        "source": "Shenzhen Stock Exchange announcement 2025-025",
    },
)


def _naive_dates(values: pd.Series) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce", utc=True)
    return converted.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()


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


def build_security_reference(
    master_path: Path,
    *,
    name_observed_at: pd.Timestamp,
    required_codes: pd.Index,
) -> pd.DataFrame:
    master = pd.read_parquet(master_path)
    required = {
        "security_id",
        "security_code",
        "exchange",
        "security_name",
        "listing_date",
        "delisting_date",
        "listing_status",
    }
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(f"Security master is missing columns: {sorted(missing)}")

    result = master.loc[:, sorted(required)].copy()
    result["security_code"] = (
        result["security_code"].astype("string").str.strip().str.zfill(6)
    )
    result["security_name"] = result["security_name"].astype("string").str.strip()
    result["listing_date"] = _naive_dates(result["listing_date"])
    result["delisting_date"] = _naive_dates(result["delisting_date"])
    result["name_observed_at"] = name_observed_at
    result["name_basis"] = "current_or_last_known_master_name"
    result["source"] = "datayes.md_security"

    for alias in LEGACY_SECURITY_ALIASES:
        if alias["security_code"] in set(result["security_code"]):
            continue
        current = result.loc[result["security_code"] == alias["current_code"]]
        if len(current) != 1:
            raise ValueError(
                f"Cannot build alias {alias['security_code']}: "
                f"current code {alias['current_code']} is missing or duplicated"
            )
        row = current.iloc[0].copy()
        for key, value in alias.items():
            if key != "current_code":
                row[key] = value
        row["delisting_date"] = pd.Timestamp(alias["delisting_date"])
        row["name_observed_at"] = pd.Timestamp(alias["name_observed_at"])
        result = pd.concat([result, row.to_frame().T], ignore_index=True)

    if result["security_code"].duplicated().any():
        duplicate = result.loc[
            result["security_code"].duplicated(keep=False), "security_code"
        ].tolist()
        raise ValueError(f"Security reference has duplicate codes: {duplicate[:10]}")
    if result[["security_code", "security_name"]].isna().any().any():
        raise ValueError("Security reference contains a missing code or name")

    normalized_required = pd.Index(
        required_codes.astype("string").str.strip().str.zfill(6)
    )
    uncovered = normalized_required.difference(result["security_code"])
    if len(uncovered):
        raise ValueError(
            "Security reference does not cover market-data columns: "
            f"{uncovered[:20].tolist()}"
        )
    return result.sort_values("security_code", kind="stable").reset_index(drop=True)


def build_industry_reference(industry_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(
        industry_path,
        columns=["trade_date", "industry_l1_code"],
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    panel["industry_l1_code"] = (
        panel["industry_l1_code"].astype("string").str.strip()
    )
    observed_codes = set(panel["industry_l1_code"].dropna().unique())
    unknown = observed_codes.difference(INDUSTRY_L1_NAMES)
    if unknown:
        raise ValueError(f"Industry panel contains unmapped codes: {sorted(unknown)}")

    coverage = (
        panel.groupby("industry_l1_code", observed=True)["trade_date"]
        .agg(first_observed_date="min", last_observed_date="max", observation_count="size")
        .reset_index()
    )
    reference = pd.DataFrame(
        {
            "industry_l1_code": list(INDUSTRY_L1_NAMES),
            "industry_l1_name": list(INDUSTRY_L1_NAMES.values()),
            "classification_system": "中信一级行业（ZX）",
            "source": "DataYes basic.ts_clf_ZX; TYPE_ID prefix 010317",
        }
    )
    return (
        reference.merge(coverage, on="industry_l1_code", how="left", validate="one_to_one")
        .sort_values("industry_l1_code")
        .reset_index(drop=True)
    )


def _source_as_of(master_path: Path, explicit: str | None) -> pd.Timestamp:
    if explicit:
        return pd.Timestamp(explicit).normalize()
    manifest_path = master_path.with_name("manifest.json")
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("market_data_end"):
            return pd.Timestamp(payload["market_data_end"]).normalize()
    raise ValueError(
        "Cannot infer security-name snapshot date; pass --name-observed-at YYYY-MM-DD"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--security-master",
        type=Path,
        default=DEFAULT_SECURITY_MASTER,
    )
    parser.add_argument("--industry-panel", type=Path)
    parser.add_argument("--name-observed-at")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    master_path = args.security_master.expanduser().resolve()
    industry_path = (
        args.industry_panel.expanduser().resolve()
        if args.industry_panel
        else data_dir / "行业数据.parquet"
    )
    close_path = data_dir / "close_df.pq"
    if not master_path.is_file():
        raise FileNotFoundError(master_path)
    if not industry_path.is_file():
        raise FileNotFoundError(industry_path)
    if not close_path.is_file():
        raise FileNotFoundError(close_path)

    close_codes = pd.read_parquet(close_path).columns
    observed_at = _source_as_of(master_path, args.name_observed_at)
    security = build_security_reference(
        master_path,
        name_observed_at=observed_at,
        required_codes=pd.Index(close_codes),
    )
    industry = build_industry_reference(industry_path)

    security_path = data_dir / SECURITY_OUTPUT
    industry_output_path = data_dir / INDUSTRY_OUTPUT
    _atomic_parquet(security, security_path)
    _atomic_parquet(industry, industry_output_path)

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "security_reference": {
            "path": str(security_path),
            "rows": int(len(security)),
            "market_code_coverage": int(len(close_codes)),
            "name_observed_at": observed_at.date().isoformat(),
            "name_basis": "current_or_last_known_master_name",
            "source": str(master_path),
            "historical_name_warning": (
                "Names are current or last-known master names, not a complete "
                "point-in-time rename history."
            ),
        },
        "industry_reference": {
            "path": str(industry_output_path),
            "rows": int(len(industry)),
            "classification_system": "中信一级行业（ZX）",
            "source": str(industry_path),
        },
    }
    manifest_path = data_dir / "manifests" / MANIFEST_OUTPUT
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(
        f"wrote {security_path} ({len(security)} rows), "
        f"{industry_output_path} ({len(industry)} rows)"
    )


if __name__ == "__main__":
    main()
