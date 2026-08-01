"""Detect possible look-ahead by recomputing factors on data prefixes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method
from .future_perturbation import _checkpoint_positions


FactorBuilder = Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]


def truncate_data_through(
    data: Mapping[str, pd.DataFrame],
    checkpoint_position: int,
) -> dict[str, pd.DataFrame]:
    """Return each input matrix through the checkpoint row inclusive."""

    return {
        symbol: frame.iloc[: checkpoint_position + 1].copy()
        for symbol, frame in data.items()
    }


def prefix_truncation_consistency_test(
    factor: pd.DataFrame,
    data: Mapping[str, pd.DataFrame],
    factor_builder: FactorBuilder,
    *,
    n_checkpoints: int = 4,
    checkpoint_positions: Iterable[int] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute checkpoint rows using only data available through that day."""

    positions = _checkpoint_positions(
        factor,
        n_checkpoints,
        checkpoint_positions,
    )
    rows: list[dict[str, Any]] = []
    total_compared = 0
    total_changed = 0
    max_abs_difference = 0.0

    for position in positions:
        prefix_data = truncate_data_through(data, position)
        recalculated = factor_builder(prefix_data).reindex(
            index=factor.index[: position + 1],
            columns=factor.columns,
        )
        original_row = factor.iloc[position].to_numpy(dtype=float, copy=False)
        recalculated_row = recalculated.iloc[-1].to_numpy(
            dtype=float,
            copy=False,
        )
        both_nan = np.isnan(original_row) & np.isnan(recalculated_row)
        compared = ~both_nan
        equal = np.isclose(
            original_row,
            recalculated_row,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        changed = compared & ~equal
        finite_pairs = np.isfinite(original_row) & np.isfinite(recalculated_row)
        row_max_difference = (
            float(
                np.max(
                    np.abs(
                        original_row[finite_pairs]
                        - recalculated_row[finite_pairs]
                    )
                )
            )
            if finite_pairs.any()
            else 0.0
        )
        compared_count = int(compared.sum())
        changed_count = int(changed.sum())
        total_compared += compared_count
        total_changed += changed_count
        max_abs_difference = max(max_abs_difference, row_max_difference)
        rows.append(
            {
                "day": factor.index[position],
                "position": position,
                "compared_values": compared_count,
                "changed_values": changed_count,
                "max_abs_difference": row_max_difference,
                "passed": changed_count == 0,
            }
        )

    detail = pd.DataFrame(rows).set_index("day")
    summary = {
        "prefix_truncation_passed": total_changed == 0,
        "prefix_truncation_checkpoints": len(positions),
        "prefix_truncation_compared_values": total_compared,
        "prefix_truncation_changed_values": total_changed,
        "prefix_truncation_changed_checkpoints": int((~detail["passed"]).sum()),
        "prefix_truncation_max_abs_difference": max_abs_difference,
    }
    return detail, summary


@evaluation_method("prefix_truncation_consistency")
def evaluate_prefix_truncation_consistency(state: EvaluationState) -> None:
    """Registered wrapper using the current expression and loaded market data."""

    if state.context.expression is None or state.context.market_data is None:
        raise ValueError(
            "Prefix-truncation consistency requires expression and market_data in context"
        )
    from engine import evaluate_expression
    from transforms.linear_decay import apply_linear_decay

    expression = state.context.expression
    source_factor = (
        state.context.source_factor
        if state.context.source_factor is not None
        else state.context.factor
    )
    source_positions = source_factor.index.get_indexer(state.context.factor.index)
    if (source_positions < 0).any():
        raise ValueError("Causality source factor does not contain the evaluation window")
    local_positions = _checkpoint_positions(
        state.context.factor,
        n_checkpoints=4,
        positions=None,
    )
    checkpoint_positions = tuple(
        int(source_positions[position]) for position in local_positions
    )

    def rebuild_factor(prefix: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        raw_factor = evaluate_expression(expression, prefix).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        return apply_linear_decay(raw_factor, state.context.decay)

    detail, summary = prefix_truncation_consistency_test(
        source_factor,
        state.context.market_data,
        rebuild_factor,
        checkpoint_positions=checkpoint_positions,
    )
    state.add_detail("prefix_truncation_consistency", detail)
    state.add_metrics(summary)
