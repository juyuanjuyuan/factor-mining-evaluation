#!/usr/bin/env python3
"""Synthetic contracts for importing company special-status events."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "import_company_special_status.py"
)
SPEC = importlib.util.spec_from_file_location("import_company_special_status", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCRIPT)


def main() -> None:
    days = pd.DatetimeIndex(
        pd.to_datetime(
            ["2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26"]
        )
    )
    codes = pd.Index(["000004", "600421"])
    completed = pd.DataFrame(
        {
            "day": [days[0]],
            "code": ["600421"],
            "is_delisting_period": [True],
        }
    )
    raw = pd.DataFrame(
        {
            "security_code": ["000004", "600421", "600421"],
            "raw_status_code": [6, 6, 7],
            "effective_start": ["2026-06-23", "2026-06-01", "2026-06-22"],
        }
    )
    imported, open_intervals = SCRIPT.extend_open_delisting_intervals(
        completed,
        raw,
        close_index=days,
        close_codes=codes,
    )
    active = imported[imported["code"] == "000004"]
    assert active["day"].tolist() == list(days[1:])
    assert open_intervals == [
        {
            "code": "000004",
            "effective_start": "2026-06-23",
            "extended_through": "2026-06-26",
            "trading_day_count": 4,
            "basis": "explicit PARTY_STATE=6 with no PARTY_STATE=7 by data cutoff",
        }
    ]

    recovered_st = SCRIPT._normalize_true_status(
        pd.DataFrame(
            {
                "day": [days[0]],
                "code": ["4"],
                "是否st": [True],
            }
        ),
        value_column="是否st",
        close_index=days,
        close_codes=codes,
    )
    assert recovered_st.iloc[0]["code"] == "000004"
    # True-only rows stop at the formal withdrawal boundary; the restored
    # security therefore has no ST row from 2026-06-23 onward.
    assert not recovered_st["day"].isin(days[1:]).any()
    try:
        SCRIPT._normalize_true_status(
            pd.DataFrame(
                {"day": [days[0]], "code": ["000004"], "是否st": ["False"]}
            ),
            value_column="是否st",
            close_index=days,
            close_codes=codes,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("string False must not pass the true-only importer")
    print("special-status import checks passed")


if __name__ == "__main__":
    main()
