#!/usr/bin/env python3
"""Reusable evaluation engine for wide equity factor matrices."""

from __future__ import annotations

import ast
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from evaluators import (
    DEFAULT_EVALUATION_METHODS,
    EvaluationContext,
    EvaluationMethod,
    assign_quantile,
    calc_group_returns,
    calc_ic,
    evaluate_quantile_returns,
    evaluate_rank_ic,
    evaluation_method_names,
    evaluation_required_data_symbols,
    get_ic_info,
    plot_group_cumulative,
    run_evaluation_methods,
    spearman_rank_correlation,
)
from evaluators.base import latest_versioned_key
from returns import RETURN_DEFINITION, calculate_forward_open_return
from transforms.delisting_status import (
    DELISTING_STATUS_LONG_COLUMNS,
    normalize_delisting_status_frame,
)
from transforms.linear_decay import apply_linear_decay, validate_linear_decay_window
from transforms.price_limits import normalize_st_status_frame


DEFAULT_FILES = {
    "c": "close_df.pq",
    "o": "open_df.pq",
    "h": "high_df.pq",
    "l": "low_df.pq",
    "vol": "volume_df.pq",
    "amt": "amount_df.pq",
    "vwap": "vwap_proxy_df.pq",
    "cap": "market_cap_df.pq",
    "limit": "limit_ratio_df.pq",
    "st": "st_status_df.pq",
    "delisting": "delisting_period_status_df.pq",
    # Classification is evaluator-only: factor expressions must never use
    # categorical labels as an implicit data input.
    "industry": "行业数据.parquet",
}

# These point-in-time evaluator inputs are loaded only by methods that declare
# them. They are deliberately excluded from factor expressions.
EVALUATOR_ONLY_DATA_SYMBOLS = frozenset({"industry", "delisting"})
EXPRESSION_DATA_SYMBOLS = frozenset(DEFAULT_FILES) - EVALUATOR_ONLY_DATA_SYMBOLS

NUMPY_ATTRIBUTES = {
    "abs",
    "clip",
    "exp",
    "inf",
    "isfinite",
    "isnan",
    "log",
    "log1p",
    "maximum",
    "minimum",
    "nan",
    "sign",
    "sqrt",
    "where",
}

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.keyword,
    ast.Tuple,
    ast.List,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
)


def roll(x: pd.DataFrame, window: int) -> Any:
    """Return a time-axis rolling object with the notebook's half-window minimum."""
    window = _window_int(window, "window")
    return x.rolling(window=window, min_periods=max(1, window // 2))


# Rolling reductions that pandas can only express through a per-window Python
# callback (`rolling().apply`) cost minutes on a full market matrix. The helpers
# below evaluate the same windows as one strided NumPy block per column chunk,
# preserving every pandas semantic the callback form has: windows run oldest to
# newest, the first `window - 1` rows see a shorter window, and a row whose
# window holds fewer than `max(1, window // 2)` observations stays NaN.

# Peak block size per column chunk; NumPy kernels allocate a few temporaries of
# the same shape on top of it.
_ROLLING_BLOCK_BYTES = 64 * 1024 * 1024


def _window_column_chunk(row_count: int, window: int) -> int:
    """Columns per sliding-window block, bounded by a fixed memory budget."""

    per_column_bytes = max(1, row_count * window * 8)
    return int(min(4096, max(1, _ROLLING_BLOCK_BYTES // per_column_bytes)))


def _leading_pad_counts(row_count: int, window: int) -> np.ndarray:
    """Padded slots in each row's trailing window, newest row first at index 0."""

    return np.clip(window - 1 - np.arange(row_count), 0, None).astype(float)


def _rolling_blocks(
    values: np.ndarray,
    window: int,
    *,
    pad_value: float,
) -> Any:
    """Yield ``(start, block)`` where block is ``(rows, columns, window)``."""

    row_count, column_count = values.shape
    step = _window_column_chunk(row_count, window)
    for start in range(0, column_count, step):
        chunk = values[:, start : start + step]
        padded = np.full(
            (row_count + window - 1, chunk.shape[1]),
            pad_value,
            dtype=float,
        )
        padded[window - 1 :] = chunk
        yield start, sliding_window_view(padded, window, axis=0)


def _rolling_window_reduce(
    x: pd.DataFrame,
    window: int,
    kernel: Any,
    *,
    pad_value: float = np.nan,
) -> pd.DataFrame:
    """Apply a vectorized window kernel with pandas ``roll`` semantics.

    ``kernel(block, pad_counts)`` receives one ``(rows, columns, window)``
    block and the per-row count of padded slots, and returns ``(rows,
    columns)``. Kernels never apply ``min_periods`` themselves; that mask is
    enforced here from the observation count so every operator agrees.

    ``pandas.rolling`` demotes an infinity to a missing observation before it
    reduces a window, so the block is sanitized the same way and every kernel
    can treat NaN as the single missing marker.
    """

    window = _window_int(window, "window")
    raw = x.to_numpy(dtype=float, copy=False)
    values = np.where(np.isfinite(raw), raw, np.nan)
    result = np.empty(values.shape, dtype=float)
    if not values.size:
        return pd.DataFrame(result, index=x.index, columns=x.columns)
    min_periods = max(1, window // 2)
    pad_counts = _leading_pad_counts(values.shape[0], window)
    for start, block in _rolling_blocks(values, window, pad_value=pad_value):
        observed = np.count_nonzero(~np.isnan(block), axis=2).astype(float)
        if not np.isnan(pad_value):
            # A non-NaN pad is a placeholder, not an observation.
            observed -= pad_counts[:, None]
        reduced = kernel(block, pad_counts)
        reduced[observed < min_periods] = np.nan
        result[:, start : start + block.shape[1]] = reduced
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll(x, window).mean()


def ts_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll(x, window).std(ddof=0)


def ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll(x, window).sum()


def ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll(x, window).min()


def ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll(x, window).max()


def ts_median(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll(x, window).median()


def _product_block(block: np.ndarray, pad_counts: np.ndarray) -> np.ndarray:
    # ``np.prod`` propagates NaN, so a window with any missing observation
    # stays missing. Padded slots carry the multiplicative identity.
    return block.prod(axis=2)


def ts_product(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_window_reduce(x, window, _product_block, pad_value=1.0)


def ts_count(condition: pd.DataFrame, window: int) -> pd.DataFrame:
    """Count true observations in each trailing window.

    This is the panel equivalent of the ``COUNT(condition, window)`` operator
    used by the GTJA191 formulas.  A missing condition remains missing rather
    than being silently treated as a true observation.
    """

    if not isinstance(condition, pd.DataFrame):
        raise TypeError("ts_count condition must be a pandas DataFrame")
    values = condition.where(condition.notna()).astype(float)
    return roll(values, window).sum()


def ts_corr(
    x: pd.DataFrame, y: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Rolling time-series Pearson correlation."""
    window = _window_int(window, "window")
    return x.rolling(
        window=window, min_periods=max(2, window // 2)
    ).corr(y)


def ts_cov(
    x: pd.DataFrame, y: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Rolling time-series population covariance."""
    window = _window_int(window, "window")
    return x.rolling(
        window=window, min_periods=max(2, window // 2)
    ).cov(y, ddof=0)


def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Percentile rank of the newest row in each trailing window."""
    return roll(x, window).rank(method="average", pct=True)


def _extreme_position_block(
    block: np.ndarray,
    pad_counts: np.ndarray,
    *,
    find_maximum: bool,
) -> np.ndarray:
    """One-based position of the window extreme, ignoring missing values.

    Ties resolve to the oldest occurrence, matching ``nanargmax``.
    """

    missing = np.isnan(block)
    filled = np.where(missing, -np.inf if find_maximum else np.inf, block)
    position = (
        filled.argmax(axis=2) if find_maximum else filled.argmin(axis=2)
    ).astype(float)
    # Alpha101 uses one-based positions within the oldest-to-newest window,
    # counted from the first real observation rather than the padding.
    position += 1.0 - pad_counts[:, None]
    return np.where(np.isfinite(block).any(axis=2), position, np.nan)


def _argmax_block(block: np.ndarray, pad_counts: np.ndarray) -> np.ndarray:
    return _extreme_position_block(block, pad_counts, find_maximum=True)


def _argmin_block(block: np.ndarray, pad_counts: np.ndarray) -> np.ndarray:
    return _extreme_position_block(block, pad_counts, find_maximum=False)


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_window_reduce(x, window, _argmax_block)


def ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_window_reduce(x, window, _argmin_block)


def delay(x: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return x.shift(_window_int(periods, "periods"))


def delta(x: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return x - delay(x, periods)


def pct(x: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    periods = _window_int(periods, "periods")
    return x / x.shift(periods) - 1


def rank_cs(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, method="average", pct=True)


def winsorize_cs(
    x: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Clip each day's cross-section to its own lower/upper quantiles."""

    lower = float(lower_quantile)
    upper = float(upper_quantile)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower < upper <= 1):
        raise ValueError("winsorize_cs quantiles must satisfy 0 <= lower < upper <= 1")
    quantiles = x.quantile([lower, upper], axis=1)
    lower_values = quantiles.loc[lower].reindex(x.index)
    upper_values = quantiles.loc[upper].reindex(x.index)
    return x.clip(lower=lower_values, upper=upper_values, axis=0)


def zscore_cs(x: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional z-score using population standard deviation."""

    mean = x.mean(axis=1)
    standard_deviation = x.std(axis=1, ddof=0).replace(0, np.nan)
    return x.sub(mean, axis=0).div(standard_deviation, axis=0)


def signed_power(x: pd.DataFrame, exponent: Any) -> pd.DataFrame:
    return np.sign(x) * np.power(np.abs(x), exponent)


def _decay_linear_block(block: np.ndarray, pad_counts: np.ndarray) -> np.ndarray:
    valid = np.isfinite(block)
    # While history ramps up pandas passes a shorter window, so the newest row
    # weighs the number of real slots rather than the nominal window length.
    weights = np.arange(1, block.shape[2] + 1, dtype=float) - pad_counts[:, None]
    weighted = np.where(valid, weights[:, None, :], 0.0)
    numerator = (np.where(valid, block, 0.0) * weighted).sum(axis=2)
    denominator = weighted.sum(axis=2)
    usable = denominator > 0
    return np.where(usable, numerator / np.where(usable, denominator, 1.0), np.nan)


def decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Linearly weighted mean with the largest weight on the newest row."""

    return _rolling_window_reduce(x, window, _decay_linear_block)


def ts_sma(
    x: pd.DataFrame, window: int, weight: float = 1.0
) -> pd.DataFrame:
    """Chinese technical-analysis SMA: ``(m*x + (n-m)*prev) / n``.

    This intentionally differs from :func:`ts_mean`: it is the recursive
    smoothing convention used in the GTJA191 source, where ``SMA(x, n, m)``
    has smoothing factor ``m / n``.
    """

    window = _window_int(window, "window")
    weight = float(weight)
    if not np.isfinite(weight) or weight <= 0 or weight > window:
        raise ValueError(
            "ts_sma weight must be finite, positive, and no larger than "
            f"window; received {weight!r} for window {window}"
        )
    return x.ewm(alpha=weight / window, adjust=False).mean()


def _linear_reg_slope_block(
    block: np.ndarray,
    pad_counts: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(block)
    count = valid.sum(axis=2)
    usable = count >= 2
    divisor = np.where(usable, count, 1.0)
    # Both axes are centred inside the window, so the ramp-up padding offset
    # cancels and the nominal time sequence can be used unshifted.
    time = np.where(
        valid,
        np.arange(1, block.shape[2] + 1, dtype=float),
        0.0,
    )
    observed = np.where(valid, block, 0.0)
    centered_time = np.where(
        valid, time - (time.sum(axis=2) / divisor)[:, :, None], 0.0
    )
    centered_observed = np.where(
        valid, observed - (observed.sum(axis=2) / divisor)[:, :, None], 0.0
    )
    denominator = np.square(centered_time).sum(axis=2)
    numerator = (centered_time * centered_observed).sum(axis=2)
    usable &= denominator > 0
    return np.where(usable, numerator / np.where(usable, denominator, 1.0), np.nan)


def ts_linear_reg_slope(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing least-squares slope against the time sequence ``1..window``."""

    return _rolling_window_reduce(x, window, _linear_reg_slope_block)


def ts_cum_sum(x: pd.DataFrame) -> pd.DataFrame:
    """Per-security cumulative sum along the trading-day axis."""

    return x.cumsum()


def ts_accumulate_positive_return(x: pd.DataFrame) -> pd.DataFrame:
    """Replicate GTJA191 Alpha143's stateful ``SELF`` recurrence.

    Given gross-return-like input ``x``, the recurrence starts from one and
    multiplies the prior state by ``x - 1`` only when ``x > 1``; otherwise it
    carries the prior state forward.  The unusual definition is preserved from
    the supplied DolphinDB implementation rather than reinterpreted as an
    ordinary compounded return.
    """

    if not isinstance(x, pd.DataFrame):
        raise TypeError("ts_accumulate_positive_return input must be a pandas DataFrame")
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    state = np.ones(len(x.columns), dtype=float)
    for row_position, values in enumerate(x.to_numpy(dtype=float)):
        valid = np.isfinite(values)
        grow = valid & (values > 1)
        state[grow] *= values[grow] - 1
        state[~valid] = np.nan
        result.iloc[row_position] = state
    return result


def scale_cs(x: pd.DataFrame, target: float = 1.0) -> pd.DataFrame:
    """Cross-sectionally scale each row so sum(abs(x)) equals ``target``."""
    denominator = x.abs().sum(axis=1).replace(0, np.nan)
    return x.div(denominator, axis=0) * float(target)


def where(
    condition: pd.DataFrame, if_true: Any, if_false: Any
) -> pd.DataFrame:
    """Elementwise conditional that preserves the wide DataFrame axes."""
    values = np.where(condition, if_true, if_false)
    return pd.DataFrame(values, index=condition.index, columns=condition.columns)


def elementwise_min(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.minimum(x, y), index=x.index, columns=x.columns
    )


def elementwise_max(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.maximum(x, y), index=x.index, columns=x.columns
    )


def adv(amount: pd.DataFrame, window: int) -> pd.DataFrame:
    """Average daily traded amount over the trailing window."""
    return ts_mean(amount, window)


def _window_int(value: Any, label: str) -> int:
    """Floor positive Alpha101 time parameters to whole trading days."""
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 1:
        raise ValueError(f"{label} must be at least 1; received {value!r}")
    return int(np.floor(numeric))


def _positive_int(value: Any, label: str) -> int:
    result = int(value)
    if result < 1 or result != value:
        raise ValueError(f"{label} must be a positive integer; received {value!r}")
    return result


def _coerce_signal_boundary(
    value: str | pd.Timestamp | None,
    index: pd.Index,
    label: str,
) -> pd.Timestamp | None:
    """Normalize one optional user-facing signal-date boundary.

    The market-data contract uses a ``DatetimeIndex``.  Keeping the small
    timezone adjustment here lets the web UI submit ordinary ``YYYY-MM-DD``
    dates whether a future dataset stores naive or timezone-aware dates.
    """

    if value is None:
        return None
    try:
        boundary = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date or datetime") from exc
    if pd.isna(boundary):
        raise ValueError(f"{label} must be an ISO date or datetime")
    if isinstance(index, pd.DatetimeIndex):
        if index.tz is not None and boundary.tzinfo is None:
            boundary = boundary.tz_localize(index.tz)
        elif index.tz is None and boundary.tzinfo is not None:
            boundary = boundary.tz_convert(None)
    return boundary


def restrict_evaluation_window(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    forward_return: pd.DataFrame,
    *,
    signal_start: str | pd.Timestamp | None,
    signal_end: str | pd.Timestamp | None,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Return a leakage-safe signal-date subset for one evaluation period.

    A signal formed after close on ``t`` enters at ``open[t+1]`` and exits at
    ``open[t+1+horizon]``.  When a bounded sample is requested, the last
    ``horizon + 1`` signal dates are excluded so every labelled return stays
    inside that sample.  Factor expressions are still evaluated on the full
    matrix before this function is called, preserving valid rolling history.
    """

    if signal_start is None and signal_end is None:
        return factor, close, forward_return, {}
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("Evaluation date windows require a DatetimeIndex")
    start = _coerce_signal_boundary(signal_start, close.index, "signal_start")
    end = _coerce_signal_boundary(signal_end, close.index, "signal_end")
    if start is not None and end is not None and start > end:
        raise ValueError("signal_start must not be after signal_end")

    mask = pd.Series(True, index=close.index)
    if start is not None:
        mask &= close.index >= start
    if end is not None:
        mask &= close.index <= end
    positions = np.flatnonzero(mask.to_numpy())
    if not len(positions):
        raise ValueError("The selected signal-date window contains no trading days")

    # The return label needs open[t+1+horizon], hence a bounded sample needs
    # one entry day plus ``horizon`` holding days after the signal date.
    final_position = int(positions[-1]) - (horizon + 1)
    first_position = int(positions[0])
    if final_position < first_position:
        raise ValueError(
            "The selected signal-date window is too short for the holding period; "
            f"it needs at least {horizon + 2} trading days"
        )
    selected_index = close.index[first_position : final_position + 1]
    metadata = {
        "sample_start_day": pd.Timestamp(selected_index[0]).date().isoformat(),
        "sample_end_day": pd.Timestamp(selected_index[-1]).date().isoformat(),
    }
    if start is not None:
        metadata["signal_start"] = start.date().isoformat()
    if end is not None:
        metadata["signal_end"] = end.date().isoformat()
    return (
        factor.reindex(index=selected_index),
        close.reindex(index=selected_index),
        forward_return.reindex(index=selected_index),
        metadata,
    )


def expression_namespace(data: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "np": np,
        "roll": roll,
        "ts_mean": ts_mean,
        "ts_std": ts_std,
        "ts_sum": ts_sum,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "ts_median": ts_median,
        "ts_product": ts_product,
        "ts_count": ts_count,
        "ts_corr": ts_corr,
        "ts_cov": ts_cov,
        "ts_rank": ts_rank,
        "ts_argmax": ts_argmax,
        "ts_argmin": ts_argmin,
        "delay": delay,
        "delta": delta,
        "pct": pct,
        "rank_cs": rank_cs,
        "winsorize_cs": winsorize_cs,
        "zscore_cs": zscore_cs,
        "signed_power": signed_power,
        "decay_linear": decay_linear,
        "ts_sma": ts_sma,
        "ts_linear_reg_slope": ts_linear_reg_slope,
        "ts_cum_sum": ts_cum_sum,
        "ts_accumulate_positive_return": ts_accumulate_positive_return,
        "scale_cs": scale_cs,
        "where": where,
        "elementwise_min": elementwise_min,
        "elementwise_max": elementwise_max,
        "adv": adv,
        "abs": np.abs,
        "log": np.log,
        "log1p": np.log1p,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "sign": np.sign,
    }
    namespace.update(data)
    return namespace


def parse_and_validate_expression(
    expression: str, allowed_names: set[str] | None = None
) -> ast.Expression:
    """Parse a factor expression and reject arbitrary Python execution."""
    if not expression or not expression.strip():
        raise ValueError("Factor expression cannot be empty")
    tree = ast.parse(expression, mode="eval")
    names = allowed_names or set(expression_namespace({}))
    names = names | set(EXPRESSION_DATA_SYMBOLS)

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(f"Unknown expression name: {node.id}")
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "np":
                raise ValueError("Only approved np.<function> attributes are allowed")
            if node.attr not in NUMPY_ATTRIBUTES or node.attr.startswith("_"):
                raise ValueError(f"Unsupported NumPy attribute: np.{node.attr}")
        if isinstance(node, ast.Call):
            if any(keyword.arg is None for keyword in node.keywords):
                raise ValueError("Dictionary expansion is not allowed in expressions")
    return tree


def expression_data_symbols(expression: str) -> set[str]:
    tree = parse_and_validate_expression(expression)
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} & set(
        EXPRESSION_DATA_SYMBOLS
    )


def expression_operator_names(expression: str) -> set[str]:
    tree = parse_and_validate_expression(expression)
    namespace_names = (
        set(expression_namespace({})) - set(EXPRESSION_DATA_SYMBOLS) - {"np"}
    )
    operators: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id in namespace_names:
            operators.add(function.id)
        elif (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "np"
        ):
            operators.add(f"np.{function.attr}")
    return operators


def evaluate_expression(
    expression: str, data: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    namespace = expression_namespace(data)
    tree = parse_and_validate_expression(expression, set(namespace))
    result = eval(compile(tree, "<factor-expression>", "eval"), {"__builtins__": {}}, namespace)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"Factor expression must return a pandas DataFrame, received {type(result).__name__}"
        )
    return result


def normalize_market_data_frame(
    symbol: str,
    frame: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize one market-data frame to the close matrix axes."""

    if symbol == "st":
        return normalize_st_status_frame(frame, close)
    if symbol == "delisting":
        return normalize_delisting_status_frame(frame, close)
    if symbol == "industry":
        return normalize_industry_classification_frame(frame, close)
    return (
        frame
        if frame.index.equals(close.index)
        and frame.columns.equals(close.columns)
        else frame.reindex(index=close.index, columns=close.columns)
    )


_INDUSTRY_LONG_COLUMNS = frozenset(
    {"trade_date", "security_code", "industry_l1_code"}
)


def normalize_industry_classification_frame(
    frame: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize point-in-time industry classifications to the close axes.

    The canonical on-disk input is the compact long table with
    ``trade_date``, zero-padded ``security_code``, and ``industry_l1_code``.
    A date-by-security wide matrix is also accepted for preloaded fixtures.
    In both forms the function only matches classifications on the same date;
    it never fills a classification forward or backward.
    """

    if _INDUSTRY_LONG_COLUMNS <= set(frame.columns):
        long = frame.loc[:, list(_INDUSTRY_LONG_COLUMNS)].copy()
        long["trade_date"] = pd.to_datetime(
            long["trade_date"], errors="raise"
        ).dt.normalize()
        codes = long["security_code"].astype("string").str.strip()
        if codes.isna().any() or not codes.str.fullmatch(r"\d{1,6}").all():
            raise ValueError(
                "Industry long table security_code must contain one- to six-digit codes"
            )
        long["security_code"] = codes.str.zfill(6)
        long["industry_l1_code"] = (
            long["industry_l1_code"].astype("string").str.strip().replace("", pd.NA)
        )
        if long.duplicated(["trade_date", "security_code"]).any():
            raise ValueError(
                "Industry long table has duplicate (trade_date, security_code) rows"
            )
        normalized = long.pivot(
            index="trade_date",
            columns="security_code",
            values="industry_l1_code",
        ).sort_index()
    else:
        normalized = frame.copy()
        index = pd.DatetimeIndex(
            pd.to_datetime(normalized.index, errors="raise")
        ).normalize()
        codes = pd.Index(normalized.columns.astype("string").str.strip())
        string_codes = pd.Series(codes, dtype="string")
        if not string_codes.str.fullmatch(r"\d{1,6}").all():
            raise ValueError("Industry matrix columns must contain one- to six-digit codes")
        normalized.index = index
        normalized.columns = codes.str.zfill(6)
        if normalized.index.has_duplicates or normalized.columns.has_duplicates:
            raise ValueError("Industry matrix has duplicate date or security-code labels")
        normalized = normalized.sort_index()

    return normalized.reindex(index=close.index, columns=close.columns)


def load_market_data(
    data_dir: str | Path,
    expression: str,
    file_names: Mapping[str, str] | None = None,
    extra_symbols: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir).expanduser().resolve()
    names = dict(DEFAULT_FILES)
    if file_names:
        unknown = set(file_names) - set(DEFAULT_FILES)
        if unknown:
            raise ValueError(f"Unknown data file symbols: {sorted(unknown)}")
        names.update(file_names)

    # Close remains the canonical alignment axis. Open is always required for
    # the next-open-to-open evaluation label.
    required = expression_data_symbols(expression) | {"c", "o"}
    if extra_symbols:
        unknown = set(extra_symbols) - set(DEFAULT_FILES)
        if unknown:
            raise ValueError(f"Unknown extra data symbols: {sorted(unknown)}")
        required.update(extra_symbols)
    if not (data_dir / names["c"]).is_file() and (
        data_dir / "data" / names["c"]
    ).is_file():
        data_dir = data_dir / "data"
    data: dict[str, pd.DataFrame] = {}
    for symbol in sorted(required):
        path = data_dir / names[symbol]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {symbol!r} input parquet: {path}")
        frame = pd.read_parquet(path)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"Input {path} is not a nonempty pandas DataFrame")
        is_st_long = symbol == "st" and {"day", "code", "是否st"} <= set(frame.columns)
        is_industry_long = symbol == "industry" and _INDUSTRY_LONG_COLUMNS <= set(
            frame.columns
        )
        is_delisting_long = (
            symbol == "delisting"
            and DELISTING_STATUS_LONG_COLUMNS <= set(frame.columns)
        )
        if not is_st_long and not is_industry_long and not is_delisting_long and (
            frame.index.has_duplicates or frame.columns.has_duplicates
        ):
            raise ValueError(f"Input {path} has duplicate index or column labels")
        data[symbol] = frame.sort_index()

    close = data["c"]
    for symbol, frame in list(data.items()):
        if symbol != "c":
            data[symbol] = normalize_market_data_frame(symbol, frame, close)
    return data


def prepare_market_data(
    expression: str,
    data: Mapping[str, pd.DataFrame],
    extra_symbols: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Select and align preloaded matrices required by one expression."""
    required = expression_data_symbols(expression) | {"c", "o"}
    if extra_symbols:
        required.update(extra_symbols)
    missing = required - set(data)
    if missing:
        raise KeyError(f"Preloaded market data is missing symbols: {sorted(missing)}")
    close = data["c"]
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("Preloaded close matrix must be a nonempty DataFrame")
    selected = {"c": close}
    for symbol in sorted(required - {"c"}):
        frame = data[symbol]
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(
                f"Preloaded {symbol!r} matrix must be a nonempty DataFrame"
            )
        selected[symbol] = normalize_market_data_frame(symbol, frame, close)
    return selected


def get_melt_df(now_pivot: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper for the original notebook's wide-to-long conversion."""
    frame = now_pivot.copy()
    frame.index.name = "day"
    result = frame.reset_index().melt(
        id_vars="day", var_name="code", value_name="values"
    )
    return result


def _spearman_arrays(x: np.ndarray, y: np.ndarray) -> float:
    """Backward-compatible alias for the modular Rank IC implementation."""

    return spearman_rank_correlation(x, y)


def evaluate_wide(
    factor: pd.DataFrame,
    open_prices: pd.DataFrame,
    horizon: int = 1,
    n_quantiles: int = 10,
    decay: int = 1,
) -> tuple[pd.Series, pd.DataFrame, int]:
    """Evaluate a factor against the canonical next-open-to-open label."""

    horizon = _positive_int(horizon, "horizon")
    n_quantiles = _positive_int(n_quantiles, "n_quantiles")
    decay = validate_linear_decay_window(decay)
    factor = factor.reindex(
        index=open_prices.index,
        columns=open_prices.columns,
    )
    factor = apply_linear_decay(factor, decay)
    future_return = calculate_forward_open_return(open_prices, horizon)
    context = EvaluationContext(
        factor_name="compatibility_evaluation",
        artifact_name="compatibility_evaluation",
        factor=factor,
        close=open_prices,
        forward_return=future_return,
        horizon=horizon,
        n_quantiles=n_quantiles,
        output_dir=Path.cwd(),
        decay=decay,
    )
    state = run_evaluation_methods(
        context,
        (evaluate_rank_ic, evaluate_quantile_returns),
    )
    ic_series = state.details["ic"]
    group_returns = state.details["group_returns"]
    if not isinstance(ic_series, pd.Series) or not isinstance(
        group_returns, pd.DataFrame
    ):
        raise TypeError("Modular evaluator returned unexpected detail types")
    return ic_series, group_returns, int(state.metrics["pair_count"])


def _artifact_name(factor_name: str) -> str:
    if not factor_name or not factor_name.strip():
        raise ValueError("factor_name cannot be empty")
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", factor_name.strip())
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        raise ValueError("factor_name contains no filesystem-safe characters")
    return cleaned


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _upsert_latest(path: Path, record: Mapping[str, Any]) -> None:
    new_row = pd.DataFrame([{key: _csv_value(value) for key, value in record.items()}])
    if path.exists():
        existing = pd.read_csv(path)
        if "factor_name" in existing:
            existing = existing[existing["factor_name"] != record["factor_name"]]
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row
    combined = combined.sort_values("factor_name", kind="stable")
    temporary = path.with_suffix(".tmp")
    combined.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _append_history(path: Path, record: Mapping[str, Any]) -> None:
    row = pd.DataFrame([{key: _csv_value(value) for key, value in record.items()}])
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, row], ignore_index=True, sort=False)
    else:
        combined = row
    temporary = path.with_suffix(".tmp")
    combined.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def render_test_code(
    *,
    factor_name: str,
    expression: str,
    data_dir: Path,
    output_dir: Path,
    horizon: int,
    n_quantiles: int,
    decay: int,
    file_names: Mapping[str, str],
    evaluation_methods: tuple[str, ...],
    signal_start: str | None,
    signal_end: str | None,
) -> str:
    project_src = Path(__file__).resolve().parents[1]
    from evaluators.tradability import TRADABILITY_DEFINITION

    expected_tradability_definition = (
        TRADABILITY_DEFINITION
        if "tradability_filter" in evaluation_methods
        else None
    )
    return f'''"""Editable, rerunnable factor test generated by evaluate-stock-factors."""

from pathlib import Path
import sys

def find_project_src(start: Path) -> Path:
    for parent in (start.parent, *start.parents):
        candidate = parent / "src"
        if (candidate / "engine.py").is_file():
            return candidate
    fallback = Path({str(project_src)!r})
    if (fallback / "engine.py").is_file():
        return fallback
    raise FileNotFoundError("Cannot locate project src directory")

PROJECT_SRC = find_project_src(Path(__file__).resolve())
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from engine import evaluate_factor_expression
from evaluators.base import REGISTERED_EVALUATION_METHODS
from evaluators.tradability import TRADABILITY_DEFINITION
from returns import RETURN_DEFINITION

FACTOR_NAME = {factor_name!r}
EXPRESSION = {expression!r}
DATA_DIR = Path({str(data_dir)!r})
OUTPUT_DIR = Path({str(output_dir)!r})
HORIZON = {horizon!r}
N_QUANTILES = {n_quantiles!r}
DECAY = {decay!r}
FILE_NAMES = {dict(file_names)!r}
EVALUATION_METHODS = {evaluation_methods!r}
SIGNAL_START = {signal_start!r}
SIGNAL_END = {signal_end!r}
EXPECTED_RETURN_DEFINITION = {RETURN_DEFINITION!r}
EXPECTED_TRADABILITY_DEFINITION = {expected_tradability_definition!r}

if RETURN_DEFINITION != EXPECTED_RETURN_DEFINITION:
    raise RuntimeError(
        "Return definition changed since this test was generated: "
        f"{{RETURN_DEFINITION}} != {{EXPECTED_RETURN_DEFINITION}}"
    )
if (
    EXPECTED_TRADABILITY_DEFINITION is not None
    and TRADABILITY_DEFINITION != EXPECTED_TRADABILITY_DEFINITION
):
    raise RuntimeError(
        "Tradability definition changed since this test was generated: "
        f"{{TRADABILITY_DEFINITION}} != {{EXPECTED_TRADABILITY_DEFINITION}}"
    )

result = evaluate_factor_expression(
    factor_name=FACTOR_NAME,
    expression=EXPRESSION,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    horizon=HORIZON,
    n_quantiles=N_QUANTILES,
    decay=DECAY,
    file_names=FILE_NAMES,
    # Execute the archived pipeline exactly as ordered; it may repeat methods
    # (e.g. rank_ic before and after market_cap_neutralize).
    evaluation_methods=[REGISTERED_EVALUATION_METHODS[name] for name in EVALUATION_METHODS],
    signal_start=SIGNAL_START,
    signal_end=SIGNAL_END,
)

print(result["metrics"])
print("plot:", result.get("plot_path"))
print("test code:", result["code_path"])
'''


def _write_test_code(
    *,
    output_dir: Path,
    artifact_name: str,
    run_id: str,
    source: str,
) -> tuple[Path, Path]:
    code_dir = output_dir / "code"
    history_dir = code_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    current_path = code_dir / f"{artifact_name}.py"
    history_path = history_dir / f"{artifact_name}__{run_id}.py"
    current_path.write_text(source, encoding="utf-8")
    shutil.copy2(current_path, history_path)
    return current_path, history_path


def evaluate_factor_expression(
    *,
    factor_name: str,
    expression: str,
    data_dir: str | Path,
    output_dir: str | Path,
    horizon: int = 1,
    n_quantiles: int = 10,
    decay: int = 1,
    file_names: Mapping[str, str] | None = None,
    preloaded_data: Mapping[str, pd.DataFrame] | None = None,
    evaluation_methods: Iterable[EvaluationMethod] | None = None,
    signal_start: str | pd.Timestamp | None = None,
    signal_end: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Run selected evaluation functions and persist their combined outputs."""

    horizon = _positive_int(horizon, "horizon")
    n_quantiles = _positive_int(n_quantiles, "n_quantiles")
    decay = validate_linear_decay_window(decay)
    selected_methods = tuple(
        DEFAULT_EVALUATION_METHODS
        if evaluation_methods is None
        else evaluation_methods
    )
    selected_method_names = evaluation_method_names(selected_methods)
    required_data_symbols = evaluation_required_data_symbols(selected_methods)
    artifact_name = _artifact_name(factor_name)
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("plots", "details", "code"):
        (output_dir / directory).mkdir(exist_ok=True)

    resolved_files = dict(DEFAULT_FILES)
    if file_names:
        resolved_files.update(file_names)
    now = datetime.now().astimezone()
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%f')}_{uuid4().hex[:8]}"
    source = render_test_code(
        factor_name=factor_name,
        expression=expression,
        data_dir=data_dir,
        output_dir=output_dir,
        horizon=horizon,
        n_quantiles=n_quantiles,
        decay=decay,
        file_names=resolved_files,
        evaluation_methods=selected_method_names,
        signal_start=str(signal_start) if signal_start is not None else None,
        signal_end=str(signal_end) if signal_end is not None else None,
    )
    code_path, code_history_path = _write_test_code(
        output_dir=output_dir,
        artifact_name=artifact_name,
        run_id=run_id,
        source=source,
    )

    # Archive the exact attempted test before validation or data loading so failed
    # expressions remain available for diagnosis and editing.
    parse_and_validate_expression(expression)
    data = (
        load_market_data(
            data_dir,
            expression,
            resolved_files,
            extra_symbols=required_data_symbols,
        )
        if preloaded_data is None
        else prepare_market_data(
            expression,
            preloaded_data,
            extra_symbols=required_data_symbols,
        )
    )
    source_factor = evaluate_expression(expression, data).replace([np.inf, -np.inf], np.nan)
    source_factor = source_factor.reindex(
        index=data["c"].index,
        columns=data["c"].columns,
    )
    source_factor = apply_linear_decay(source_factor, decay)
    forward_return = calculate_forward_open_return(data["o"], horizon)
    factor, evaluation_close, forward_return, sample_metadata = restrict_evaluation_window(
        source_factor,
        data["c"],
        forward_return,
        signal_start=signal_start,
        signal_end=signal_end,
        horizon=horizon,
    )
    evaluation_context = EvaluationContext(
        factor_name=factor_name,
        artifact_name=artifact_name,
        factor=factor,
        close=evaluation_close,
        forward_return=forward_return,
        horizon=horizon,
        n_quantiles=n_quantiles,
        output_dir=output_dir,
        decay=decay,
        expression=expression,
        market_data=data,
        source_factor=source_factor,
    )
    evaluation_state = run_evaluation_methods(
        evaluation_context,
        selected_methods,
    )

    detail_paths: dict[str, Path] = {}
    for detail_name, detail in evaluation_state.details.items():
        detail_artifact = _artifact_name(detail_name)
        detail_path = (
            output_dir
            / "details"
            / f"{artifact_name}__{detail_artifact}.csv"
        )
        detail.to_csv(detail_path, encoding="utf-8-sig")
        detail_paths[detail_name] = detail_path

    record: dict[str, Any] = {
        "run_id": run_id,
        "evaluated_at": now.isoformat(timespec="seconds"),
        "factor_name": factor_name,
        "artifact_name": artifact_name,
        "expression": expression,
        "horizon": horizon,
        "return_definition": RETURN_DEFINITION,
        "n_quantiles": n_quantiles,
        "decay": decay,
        "evaluation_methods": ",".join(selected_method_names),
        "evaluation_details": json.dumps(
            {
                name: path.relative_to(output_dir).as_posix()
                for name, path in detail_paths.items()
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "evaluation_artifacts": json.dumps(
            {
                name: path.relative_to(output_dir).as_posix()
                for name, path in evaluation_state.artifacts.items()
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        **sample_metadata,
    }
    reserved_metrics = set(record) & set(evaluation_state.metrics)
    if reserved_metrics:
        raise ValueError(
            "Evaluation methods cannot overwrite run metadata: "
            f"{sorted(reserved_metrics)}"
        )
    record.update(evaluation_state.metrics)
    metrics_path = output_dir / "metrics.csv"
    history_path = output_dir / "metrics_history.csv"
    _upsert_latest(metrics_path, record)
    _append_history(history_path, record)

    result = {
        "metrics": record,
        "metrics_path": str(metrics_path),
        "metrics_history_path": str(history_path),
        "code_path": str(code_path),
        "code_history_path": str(code_history_path),
        "evaluation_methods": list(selected_method_names),
        "detail_paths": {
            name: str(path) for name, path in detail_paths.items()
        },
        "artifact_paths": {
            name: str(path)
            for name, path in evaluation_state.artifacts.items()
        },
    }
    detail_aliases = {
        "ic": "ic_path",
        "group_returns": "group_returns_path",
        "cumulative_returns": "cumulative_returns_path",
    }
    for detail_name, result_key in detail_aliases.items():
        detail_key = latest_versioned_key(detail_name, detail_paths)
        if detail_key is not None:
            result[result_key] = str(detail_paths[detail_key])
    if "plot" in evaluation_state.artifacts:
        result["plot_path"] = str(evaluation_state.artifacts["plot"])
    return result


def result_as_json(result: Mapping[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True)
