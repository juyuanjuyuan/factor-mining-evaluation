"""Fixed A-share broad-market cycle bands used only as chart context.

The boundaries use CSI All Share (000985) peak/trough segmentation with a
25% reversal threshold.  They are descriptive, retrospective annotations,
not a real-time regime classifier or an input to factor metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd


@dataclass(frozen=True)
class MarketCycle:
    """One broad-market rising or falling segment for chart backgrounds."""

    start_day: str
    end_day: str | None
    direction: str
    label: str
    provisional: bool = False


# The final upward phase remains open because the 25% reversal rule has not
# confirmed a subsequent bear leg.  Keeping the uncertainty in the data model
# prevents a chart from silently treating the latest period as settled history.
MARKET_CYCLES: tuple[MarketCycle, ...] = (
    MarketCycle("2013-06-25", "2015-06-12", "up", "上升段"),
    MarketCycle("2015-06-12", "2016-01-28", "down", "下跌段"),
    MarketCycle("2016-01-28", "2017-11-13", "up", "上升段"),
    MarketCycle("2017-11-13", "2018-10-18", "down", "下跌段"),
    MarketCycle("2018-10-18", "2021-12-13", "up", "上升段"),
    MarketCycle("2021-12-13", "2024-02-05", "down", "下跌段"),
    MarketCycle("2024-02-05", None, "up", "上升段（暂定）", provisional=True),
)


def market_cycle_backgrounds() -> list[dict[str, str | bool | None]]:
    """Return JSON-safe cycle metadata for an interactive chart client."""

    return [
        {
            "start_day": cycle.start_day,
            "end_day": cycle.end_day,
            "direction": cycle.direction,
            "label": cycle.label,
            "provisional": cycle.provisional,
        }
        for cycle in MARKET_CYCLES
    ]


def cycle_backgrounds_for_dates(index: pd.Index) -> tuple[MarketCycle, ...]:
    """Clip cycle bands to a plotted date index without expanding its x-axis."""

    dates = pd.DatetimeIndex(index).dropna().sort_values()
    if not len(dates):
        return ()
    first_day = dates[0].normalize()
    last_day = dates[-1].normalize()
    clipped: list[MarketCycle] = []
    for cycle in MARKET_CYCLES:
        start = max(pd.Timestamp(cycle.start_day), first_day)
        end = min(
            pd.Timestamp(cycle.end_day) if cycle.end_day is not None else last_day,
            last_day,
        )
        if start <= end:
            clipped.append(
                replace(
                    cycle,
                    start_day=start.date().isoformat(),
                    end_day=end.date().isoformat(),
                )
            )
    return tuple(clipped)
