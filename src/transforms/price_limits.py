"""Price-limit ratio helpers for A-share tradability checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CHINEXT_PREFIXES = ("300", "301", "302")
STAR_MARKET_PREFIXES = ("688", "689")
BSE_PREFIXES = ("43", "83", "87", "88", "92")


def price_limit_ratio_for_code(code: Any) -> float:
    """Infer the regular daily price-limit ratio from a six-digit stock code."""

    normalized = str(code).strip().zfill(6)
    if normalized.startswith(CHINEXT_PREFIXES + STAR_MARKET_PREFIXES):
        return 0.20
    if normalized.startswith(BSE_PREFIXES):
        return 0.30
    return 0.10


def infer_price_limit_ratio_frame(
    close: pd.DataFrame,
    *,
    no_limit_first_n: int = 5,
) -> pd.DataFrame:
    """Build a wide price-limit-ratio matrix aligned to ``close``.

    The ratio is inferred from code prefixes. For stocks that first appear
    after the first row of the available dataset, the first ``no_limit_first_n``
    valid trading days are set to NaN to approximate no-limit IPO windows.
    Stocks already present on the first dataset day are treated as seasoned
    listings so the dataset start is not mistaken for a listing date.
    """

    if not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("close must be a nonempty DataFrame")
    if close.index.has_duplicates or close.columns.has_duplicates:
        raise ValueError("close must not have duplicate axes")

    column_ratios = np.array(
        [price_limit_ratio_for_code(code) for code in close.columns],
        dtype=float,
    )
    values = np.broadcast_to(column_ratios, close.shape).copy()

    if no_limit_first_n > 0:
        valid = close.notna().to_numpy()
        for column_position in range(valid.shape[1]):
            valid_positions = np.flatnonzero(valid[:, column_position])
            if len(valid_positions) == 0:
                continue
            first_position = int(valid_positions[0])
            if first_position == 0:
                continue
            no_limit_positions = valid_positions[:no_limit_first_n]
            values[no_limit_positions, column_position] = np.nan

    return pd.DataFrame(values, index=close.index, columns=close.columns)


def normalize_st_status_frame(
    frame: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize ST status data to a boolean wide matrix aligned to ``close``."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("ST status input must be a nonempty DataFrame")
    if {"day", "code", "是否st"} <= set(frame.columns):
        status = frame.loc[:, ["day", "code", "是否st"]].copy()
        status["day"] = pd.to_datetime(status["day"])
        status["code"] = status["code"].astype(str).str.zfill(6)
        status["is_st"] = status["是否st"].astype(bool)
        wide = status.pivot_table(
            index="day",
            columns="code",
            values="is_st",
            aggfunc="max",
            fill_value=False,
        )
    else:
        wide = frame.copy()
        if not isinstance(wide.index, pd.DatetimeIndex):
            wide.index = pd.to_datetime(wide.index)
        wide.columns = wide.columns.astype(str).str.zfill(6)

    aligned = wide.reindex(index=close.index, columns=close.columns).astype("boolean")
    return aligned.fillna(False).astype(bool)
