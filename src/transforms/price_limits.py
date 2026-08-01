"""Price-limit ratio helpers for A-share tradability checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CHINEXT_PREFIXES = ("300", "301", "302")
STAR_MARKET_PREFIXES = ("688", "689")
BSE_PREFIXES = ("43", "83", "87", "88", "92")
NON_MAIN_BOARD_PREFIXES = CHINEXT_PREFIXES + STAR_MARKET_PREFIXES + BSE_PREFIXES
MAIN_BOARD_ST_LIMIT_CHANGE_DAY = pd.Timestamp("2026-07-06")


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


def reconcile_incremental_st_status(
    historical_status: pd.DataFrame,
    incoming_market: pd.DataFrame,
    *,
    main_board_limit_change_day: pd.Timestamp = MAIN_BOARD_ST_LIMIT_CHANGE_DAY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild an incremental ST tail without trusting stale vendor events.

    ``equ_inst_sstate`` event histories can be incomplete for securities that
    entered ST years ago or later removed the warning.  Treating the latest
    event returned by that table as the current state produces both false
    negatives and false positives.  This reconciler instead:

    1. carries the last trustworthy daily status into the increment;
    2. before 2026-07-06, uses the exchange's exact 5%/10% main-board limits
       as a same-day ST/normal observation;
    3. applies only status events whose effective date is inside the increment;
    4. carries the resulting state after the rule change, when main-board ST
       and ordinary stocks both use 10% limits.

    The incoming frame must already contain normalized ``date`` and ``code``
    columns plus raw previous close, exact exchange limits, and event metadata.
    """

    required_history = {"day", "code", "是否st"}
    missing_history = required_history.difference(historical_status.columns)
    if missing_history:
        raise ValueError(
            f"historical ST data is missing columns: {sorted(missing_history)}"
        )
    required_incoming = {
        "date",
        "code",
        "pre_close_raw",
        "limit_up_price",
        "limit_down_price",
        "st_effective_date",
        "st_party_state",
    }
    missing_incoming = required_incoming.difference(incoming_market.columns)
    if missing_incoming:
        raise ValueError(
            f"incoming market data is missing ST reconciliation columns: "
            f"{sorted(missing_incoming)}"
        )
    if incoming_market.empty:
        raise ValueError("incoming market data must not be empty")

    history = historical_status.loc[:, ["day", "code", "是否st"]].copy()
    history["day"] = pd.to_datetime(history["day"]).dt.tz_localize(None).dt.normalize()
    history["code"] = history["code"].astype("string").str.strip().str.zfill(6)
    history["是否st"] = history["是否st"].astype(bool)

    incoming = incoming_market.copy()
    incoming["date"] = pd.to_datetime(incoming["date"]).dt.tz_localize(None).dt.normalize()
    incoming["code"] = incoming["code"].astype("string").str.strip().str.zfill(6)
    if incoming.duplicated(["date", "code"]).any():
        raise ValueError("incoming market data has duplicate date/code rows")
    incoming = incoming.sort_values(["date", "code"]).reset_index(drop=True)

    start_day = pd.Timestamp(incoming["date"].min())
    history = history[history["day"] < start_day].copy()
    state: dict[str, bool] = {}
    boundary_day: pd.Timestamp | None = None
    if not history.empty:
        boundary_day = pd.Timestamp(history["day"].max())
        boundary = history[history["day"] == boundary_day]
        state = dict(zip(boundary["code"].astype(str), boundary["是否st"].astype(bool)))

    rebuilt_parts: list[pd.DataFrame] = []
    limit_inferred_observations = 0
    explicit_event_observations = 0
    change_day = pd.Timestamp(main_board_limit_change_day).normalize()

    for day, daily in incoming.groupby("date", sort=True):
        codes = daily["code"].astype(str).to_numpy()
        values = np.array([state.get(code, False) for code in codes], dtype=bool)

        previous_close = daily["pre_close_raw"].to_numpy(dtype=float, copy=False)
        limit_up_price = daily["limit_up_price"].to_numpy(dtype=float, copy=False)
        limit_down_price = daily["limit_down_price"].to_numpy(dtype=float, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            up_ratio = limit_up_price / previous_close - 1
            down_ratio = 1 - limit_down_price / previous_close
        is_main_board = ~daily["code"].str.startswith(NON_MAIN_BOARD_PREFIXES).to_numpy()
        comparable = (
            is_main_board
            & np.isfinite(up_ratio)
            & np.isfinite(down_ratio)
            & (previous_close > 0)
        )
        if pd.Timestamp(day) < change_day:
            five_percent = (
                comparable
                & (up_ratio >= 0.035)
                & (up_ratio <= 0.065)
                & (down_ratio >= 0.035)
                & (down_ratio <= 0.065)
            )
            ten_percent = (
                comparable
                & (up_ratio >= 0.085)
                & (up_ratio <= 0.115)
                & (down_ratio >= 0.085)
                & (down_ratio <= 0.115)
            )
            values[five_percent] = True
            values[ten_percent] = False
            limit_inferred_observations += int((five_percent | ten_percent).sum())

        effective_dates = pd.to_datetime(
            daily["st_effective_date"], errors="coerce"
        )
        if isinstance(effective_dates.dtype, pd.DatetimeTZDtype):
            effective_dates = effective_dates.dt.tz_localize(None)
        effective_dates = effective_dates.dt.normalize()
        party_state = pd.to_numeric(daily["st_party_state"], errors="coerce")
        explicit_event = (effective_dates == pd.Timestamp(day)) & party_state.notna()
        if explicit_event.any():
            event_positions = explicit_event.to_numpy()
            values[event_positions] = (
                party_state.loc[explicit_event].to_numpy(dtype=float) != 1
            )
            explicit_event_observations += int(explicit_event.sum())

        state.update(zip(codes, values.astype(bool)))
        rebuilt_parts.append(
            pd.DataFrame(
                {
                    "day": pd.Timestamp(day),
                    "code": codes,
                    "是否st": values,
                }
            )
        )

    rebuilt = pd.concat(rebuilt_parts, ignore_index=True)
    combined = (
        pd.concat([history, rebuilt], ignore_index=True)
        .sort_values(["day", "code"])
        .reset_index(drop=True)
    )
    diagnostics = {
        "increment_start_day": start_day.date().isoformat(),
        "increment_end_day": pd.Timestamp(incoming["date"].max()).date().isoformat(),
        "boundary_day": boundary_day.date().isoformat() if boundary_day is not None else None,
        "limit_inferred_observations": int(limit_inferred_observations),
        "explicit_event_observations": int(explicit_event_observations),
        "rebuilt_rows": int(len(rebuilt)),
        "latest_st_count": int(
            rebuilt.loc[rebuilt["day"] == rebuilt["day"].max(), "是否st"].sum()
        ),
    }
    return combined, diagnostics
