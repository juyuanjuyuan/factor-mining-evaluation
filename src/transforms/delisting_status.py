"""Normalize point-in-time delisting-consolidation status data."""

from __future__ import annotations

import pandas as pd


DELISTING_STATUS_LONG_COLUMNS = frozenset(
    {"day", "code", "is_delisting_period"}
)


def _strict_true_only(values: pd.Series, *, label: str) -> pd.Series:
    """Accept actual boolean/0-1 values and reject strings or false rows."""

    try:
        normalized = values.astype("boolean")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain boolean true values only") from exc
    if normalized.isna().any() or not bool(normalized.eq(True).all()):
        raise ValueError(f"{label} must contain true observations only")
    return normalized.astype(bool)


def normalize_delisting_status_frame(
    frame: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Return a boolean delisting-period matrix aligned to ``close``.

    The canonical long file stores true observations only. Missing day/code
    pairs therefore mean that the formal delisting-consolidation state is not
    active on that date. A wide matrix is accepted for preloaded test data.
    Status is matched on the same date and is never carried by this loader.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Delisting status input must be a nonempty DataFrame")
    if DELISTING_STATUS_LONG_COLUMNS <= set(frame.columns):
        status = frame.loc[:, ["day", "code", "is_delisting_period"]].copy()
        status["day"] = pd.to_datetime(status["day"], errors="raise").dt.normalize()
        codes = status["code"].astype("string").str.strip()
        if codes.isna().any() or not codes.str.fullmatch(r"\d{1,6}").all():
            raise ValueError(
                "Delisting status codes must contain one- to six-digit numbers"
            )
        status["code"] = codes.str.zfill(6)
        if status.duplicated(["day", "code"]).any():
            raise ValueError("Delisting status has duplicate (day, code) rows")
        status["is_delisting_period"] = _strict_true_only(
            status["is_delisting_period"],
            label="Canonical delisting status",
        )
        wide = status.pivot(
            index="day",
            columns="code",
            values="is_delisting_period",
        )
    else:
        wide = frame.copy()
        wide.index = pd.DatetimeIndex(
            pd.to_datetime(wide.index, errors="raise")
        ).normalize()
        codes = pd.Index(wide.columns.astype("string").str.strip())
        normalized_codes = pd.Series(codes, dtype="string")
        if not normalized_codes.str.fullmatch(r"\d{1,6}").all():
            raise ValueError(
                "Delisting matrix columns must contain one- to six-digit numbers"
            )
        wide.columns = codes.str.zfill(6)
        if wide.index.has_duplicates or wide.columns.has_duplicates:
            raise ValueError("Delisting matrix has duplicate axes")

    try:
        aligned = wide.reindex(index=close.index, columns=close.columns).astype(
            "boolean"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Delisting matrix must contain boolean/0-1 values") from exc
    return aligned.fillna(False).astype(bool)
