"""Lazy enrichment of holdings rows with human-readable reference names."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


SECURITY_REFERENCE_FILE = "security_name_reference.parquet"
INDUSTRY_REFERENCE_FILE = "industry_l1_name_reference.parquet"


@lru_cache(maxsize=8)
def _read_reference(path_text: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return pd.read_parquet(path_text)


class HoldingReferenceData:
    """Load compact name tables and enrich only the requested holdings page."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).expanduser().resolve()

    def _read(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.is_file():
            return pd.DataFrame()
        return _read_reference(str(path), path.stat().st_mtime_ns).copy()

    def _security(self) -> pd.DataFrame:
        frame = self._read(SECURITY_REFERENCE_FILE)
        if frame.empty:
            return frame
        required = {"security_code", "security_name", "name_basis"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{SECURITY_REFERENCE_FILE} is missing columns: {sorted(missing)}"
            )
        frame["security_code"] = (
            frame["security_code"].astype("string").str.strip().str.zfill(6)
        )
        if frame["security_code"].duplicated().any():
            raise ValueError(f"{SECURITY_REFERENCE_FILE} contains duplicate codes")
        return frame.set_index("security_code")

    def _industry(self) -> pd.DataFrame:
        frame = self._read(INDUSTRY_REFERENCE_FILE)
        if frame.empty:
            return frame
        required = {
            "industry_l1_code",
            "industry_l1_name",
            "classification_system",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{INDUSTRY_REFERENCE_FILE} is missing columns: {sorted(missing)}"
            )
        frame["industry_l1_code"] = (
            frame["industry_l1_code"].astype("string").str.strip()
        )
        if frame["industry_l1_code"].duplicated().any():
            raise ValueError(f"{INDUSTRY_REFERENCE_FILE} contains duplicate codes")
        return frame.set_index("industry_l1_code")

    def enrich(self, holdings: pd.DataFrame) -> pd.DataFrame:
        result = holdings.copy()
        security = self._security()
        codes = result["security_code"].astype("string").str.strip().str.zfill(6)
        if security.empty:
            result["security_name"] = pd.NA
            result["security_name_basis"] = pd.NA
        else:
            result["security_name"] = codes.map(security["security_name"])
            result["security_name_basis"] = codes.map(security["name_basis"])

        industry = self._industry()
        industry_codes = result["industry_l1_code"].astype("string").str.strip()
        if industry.empty:
            result["industry_l1_name"] = pd.NA
        else:
            result["industry_l1_name"] = industry_codes.map(
                industry["industry_l1_name"]
            )
        return result

    def metadata(self) -> dict[str, object]:
        security = self._security()
        industry = self._industry()
        observed_at = None
        if not security.empty and "name_observed_at" in security:
            dates = pd.to_datetime(security["name_observed_at"], errors="coerce")
            if dates.notna().any():
                observed_at = dates.max().date().isoformat()
        classification = None
        if not industry.empty:
            values = industry["classification_system"].dropna().astype(str).unique()
            if len(values):
                classification = values[0]
        return {
            "security_name_basis": "当前或退市前最后简称",
            "security_name_is_point_in_time": False,
            "security_name_observed_at": observed_at,
            "industry_classification": classification,
        }
