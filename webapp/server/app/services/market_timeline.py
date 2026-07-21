"""Read the actual tradable timeline without loading the wide price matrix."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from engine import DEFAULT_FILES

from ..config import Settings


@lru_cache(maxsize=8)
def _read_trading_days(path_text: str, mtime_ns: int) -> tuple[str, ...]:
    """Load only the parquet index; the thousands of security columns stay on disk."""

    del mtime_ns
    frame = pd.read_parquet(Path(path_text), columns=[])
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("close_df.pq must use a DatetimeIndex for model time slicing")
    dates = pd.DatetimeIndex(frame.index).sort_values().unique()
    if not len(dates):
        raise ValueError("close_df.pq has no trading days")
    return tuple(pd.Timestamp(day).date().isoformat() for day in dates)


def trading_days(settings: Settings) -> tuple[str, ...]:
    path = settings.data_dir / DEFAULT_FILES["c"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing close input parquet: {path}")
    return _read_trading_days(str(path.resolve()), path.stat().st_mtime_ns)


def timeline_payload(settings: Settings) -> dict[str, object]:
    days = trading_days(settings)
    return {
        "trading_days": list(days),
        "start_day": days[0],
        "end_day": days[-1],
        "count": len(days),
    }


def validate_model_windows(
    settings: Settings,
    *,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizon: int,
) -> None:
    """Require model splits to be real trading-day slices with valid labels."""

    days = trading_days(settings)
    positions = {day: position for position, day in enumerate(days)}
    requested = {
        "训练开始": train_start,
        "训练结束": train_end,
        "测试开始": test_start,
        "测试结束": test_end,
    }
    unknown = [f"{label} {day}" for label, day in requested.items() if day not in positions]
    if unknown:
        raise ValueError(
            "日期必须在当前行情数据的交易日范围内：" + "；".join(unknown)
        )
    minimum_days = horizon + 2
    for label, start, end in (
        ("训练集", train_start, train_end),
        ("测试集", test_start, test_end),
    ):
        count = positions[end] - positions[start] + 1
        if count < minimum_days:
            raise ValueError(
                f"{label}至少需要 {minimum_days} 个交易日，才能容纳 H={horizon} 的开盘收益标签"
            )
