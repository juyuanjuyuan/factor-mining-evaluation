"""Detect possible look-ahead by perturbing data strictly after checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


FactorBuilder = Callable[[Mapping[str, pd.DataFrame]], pd.DataFrame]


def _checkpoint_positions(
    factor: pd.DataFrame,
    n_checkpoints: int,
    positions: Iterable[int] | None,
) -> tuple[int, ...]:
    last_position = len(factor.index) - 1
    if last_position < 1:
        raise ValueError("At least two dates are required for perturbation testing")
    if positions is not None:
        selected = sorted({int(position) for position in positions})
        if not selected or selected[0] < 0 or selected[-1] >= last_position:
            raise ValueError("Checkpoint positions must be between 0 and T-2")
        return tuple(selected)
    if n_checkpoints < 1:
        raise ValueError("n_checkpoints must be positive")
    eligible = np.flatnonzero(
        np.isfinite(factor.to_numpy(dtype=float, copy=False)).any(axis=1)
    )
    eligible = eligible[eligible < last_position]
    if not len(eligible):
        raise ValueError("No finite factor row is available before the final date")
    offsets = np.linspace(
        0,
        len(eligible) - 1,
        num=min(n_checkpoints, len(eligible)),
        dtype=int,
    )
    return tuple(int(position) for position in np.unique(eligible[offsets]))


def perturb_data_after(
    data: Mapping[str, pd.DataFrame],
    checkpoint_position: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Randomly change finite values after a row while preserving axes and NaNs."""

    perturbed: dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        values = frame.to_numpy(dtype=float, copy=True)
        future = values[checkpoint_position + 1 :]
        finite = np.isfinite(future)
        if finite.any():
            multipliers = rng.lognormal(mean=0.0, sigma=0.45, size=future.shape)
            future[finite] *= multipliers[finite]
            zero_mask = finite & (future == 0)
            if zero_mask.any():
                nonzero_scale = np.nanmedian(np.abs(values[np.isfinite(values)]))
                scale = (
                    float(nonzero_scale)
                    if np.isfinite(nonzero_scale) and nonzero_scale > 0
                    else 1.0
                )
                replacements = rng.uniform(0.01, 0.2, size=future.shape) * scale
                future[zero_mask] = replacements[zero_mask]
        perturbed[symbol] = pd.DataFrame(
            values,
            index=frame.index,
            columns=frame.columns,
        )
    return perturbed


def future_data_perturbation_test(
    factor: pd.DataFrame,
    data: Mapping[str, pd.DataFrame],
    factor_builder: FactorBuilder,
    *,
    n_checkpoints: int = 4,
    checkpoint_positions: Iterable[int] | None = None,
    seed: int = 20260702,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute checkpoint rows after random future-only input perturbations."""

    positions = _checkpoint_positions(
        factor,
        n_checkpoints,
        checkpoint_positions,
    )
    rows: list[dict[str, Any]] = []
    total_compared = 0
    total_changed = 0
    max_abs_difference = 0.0

    for test_number, position in enumerate(positions):
        rng = np.random.default_rng(seed + test_number)
        perturbed_data = perturb_data_after(data, position, rng)
        recalculated = factor_builder(perturbed_data).reindex(
            index=factor.index,
            columns=factor.columns,
        )
        original_row = factor.iloc[position].to_numpy(dtype=float, copy=False)
        recalculated_row = recalculated.iloc[position].to_numpy(
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
        "future_perturbation_checkpoints": len(positions),
        "future_perturbation_compared_values": total_compared,
        "future_perturbation_changed_values": total_changed,
        "future_perturbation_changed_checkpoints": int(
            (~detail["passed"]).sum()
        ),
        "future_perturbation_max_abs_difference": max_abs_difference,
        "future_perturbation_seed": seed,
    }
    return detail, summary


@evaluation_method("future_data_perturbation")
def evaluate_future_data_perturbation(state: EvaluationState) -> None:
    """Registered wrapper using the current expression and loaded market data."""

    if state.context.expression is None or state.context.market_data is None:
        raise ValueError(
            "Future-data perturbation requires expression and market_data in context"
        )
    from engine import evaluate_expression

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

    def rebuild_factor(perturbed: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        # The engine sanitizes the original factor before creating the
        # EvaluationContext. Apply the identical normalization here; otherwise
        # a stable +/-inf is compared with the original NaN and is falsely
        # reported as a future-data change.
        return evaluate_expression(expression, perturbed).replace(
            [np.inf, -np.inf],
            np.nan,
        )

    detail, summary = future_data_perturbation_test(
        source_factor,
        state.context.market_data,
        rebuild_factor,
        checkpoint_positions=checkpoint_positions,
    )
    state.add_detail("future_data_perturbation", detail)
    state.add_metrics(summary)
