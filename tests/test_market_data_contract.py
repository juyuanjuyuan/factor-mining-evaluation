#!/usr/bin/env python3
"""Validate the normalized local market-data file and axis contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from paths import DATA_DIR
from returns import calculate_forward_open_return


WIDE_FILES = {
    "open": "open_df.pq",
    "high": "high_df.pq",
    "low": "low_df.pq",
    "close": "close_df.pq",
    "volume": "volume_df.pq",
    "amount": "amount_df.pq",
    "vwap_proxy": "vwap_proxy_df.pq",
    "market_cap": "market_cap_df.pq",
    "limit_ratio": "limit_ratio_df.pq",
    "float_market_cap": "float_market_cap_df.pq",
    "limit_up_price": "limit_up_price_df.pq",
    "limit_down_price": "limit_down_price_df.pq",
}


def main() -> None:
    root = DATA_DIR
    frames = {
        name: pd.read_parquet(root / filename)
        for name, filename in WIDE_FILES.items()
    }
    close = frames["close"]
    assert isinstance(close.index, pd.DatetimeIndex)
    assert close.index.is_monotonic_increasing
    assert not close.index.has_duplicates
    assert not close.columns.has_duplicates

    for name, frame in frames.items():
        assert frame.shape == close.shape, (name, frame.shape, close.shape)
        assert frame.index.equals(close.index), name
        assert frame.columns.equals(close.columns), name

    for name in (
        "volume",
        "amount",
        "vwap_proxy",
        "market_cap",
        "limit_ratio",
        "float_market_cap",
        "limit_up_price",
        "limit_down_price",
    ):
        finite = frames[name].to_numpy(dtype=float, copy=False)
        assert not np.isinf(finite).any(), name
        assert np.nanmin(finite) >= 0, name
    limit_values = set(
        np.unique(frames["limit_ratio"].stack(future_stack=True).dropna())
    )
    assert limit_values <= {0.10, 0.20, 0.30}, limit_values

    open_sample = frames["open"].iloc[:10, :10]
    return_label = calculate_forward_open_return(open_sample, horizon=1)
    pd.testing.assert_series_equal(
        return_label.iloc[0],
        open_sample.iloc[2] / open_sample.iloc[1] - 1,
        check_names=False,
    )
    assert return_label.iloc[-2:].isna().all().all()

    st_status = pd.read_parquet(root / "st_status_df.pq")
    assert {"day", "code", "是否st"}.issubset(st_status.columns)
    print(
        {
            "shape": close.shape,
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "files": WIDE_FILES,
        }
    )
    print("market data contract checks passed")


if __name__ == "__main__":
    main()
