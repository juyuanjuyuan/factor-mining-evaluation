"""Persist the highest-quantile historical holdings for on-demand review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method, versioned_key
from .quantile_groups import quantile_membership


HOLDING_AUDIT_ARTIFACT = "holding_audit"
HOLDING_AUDIT_SUMMARY = "holding_audit_summary"
HOLDING_AUDIT_COLUMNS = (
    "signal_day",
    "entry_day",
    "exit_day",
    "security_code",
    "group",
    "factor_rank",
    "rank_in_top_group",
    "target_weight",
    "factor_value",
    "forward_open_return",
    "entry_open",
    "exit_open",
    "market_cap_yi",
    "industry_l1_code",
    "entry_is_st",
)


def _next_artifact_path(state: EvaluationState) -> Path:
    """Choose the filename matching EvaluationState's versioned artifact key."""

    version = 1
    while versioned_key(HOLDING_AUDIT_ARTIFACT, version) in state.artifacts:
        version += 1
    suffix = "" if version == 1 else f"__{version}"
    return (
        state.context.output_dir
        / "details"
        / f"{state.context.artifact_name}__{HOLDING_AUDIT_ARTIFACT}{suffix}.parquet"
    )


def _require_dataframe(state: EvaluationState, name: str) -> pd.DataFrame:
    detail = state.require_detail(name)
    if not isinstance(detail, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return detail


def _industry_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


@evaluation_method(
    HOLDING_AUDIT_ARTIFACT,
    requires=(
        "industry_market_cap_neutralize",
        "tradability_filter",
        "quantile_net_returns",
    ),
    required_data_symbols=("cap", "industry", "o", "st"),
)
def evaluate_holding_audit(state: EvaluationState) -> None:
    """Write the top quantile's daily holdings to a compact Parquet artifact.

    The method consumes ``state.factor`` after every preceding transform.  Its
    dependency order deliberately forces the profitability-template sequence
    through joint industry/size neutralization, entry-day tradability filtering,
    and net group-return construction before positions are materialized.
    """

    market_data = state.context.market_data
    if market_data is None or not {"cap", "industry", "o", "st"} <= set(market_data):
        raise ValueError(
            "holding_audit requires cap/industry/o/st market-data matrices"
        )

    group_returns = _require_dataframe(state, "group_returns")
    transaction_costs = _require_dataframe(state, "quantile_transaction_cost")
    if not group_returns.index.equals(transaction_costs.index):
        raise ValueError(
            "holding_audit requires group returns and transaction costs "
            "on identical signal dates"
        )
    if not group_returns.index.isin(state.factor.index).all():
        raise ValueError(
            "holding_audit portfolio signal dates must exist in the working factor"
        )

    # Net-return construction already excludes dates whose factor/forward
    # return universe cannot form a portfolio.  In a full-history run this
    # necessarily removes the final horizon+1 rows, whose future exit opens
    # do not exist yet.  Audit exactly those emitted portfolio dates instead
    # of rejecting the complete working-factor timeline at its natural tail.
    factor = state.factor.reindex(index=group_returns.index)
    forward_return = state.context.forward_return.reindex(
        index=factor.index,
        columns=factor.columns,
    )
    if not forward_return.index.equals(factor.index) or not forward_return.columns.equals(
        factor.columns
    ):
        raise ValueError("holding_audit requires forward returns aligned to factor")

    open_prices = market_data["o"].reindex(columns=factor.columns)
    cap = market_data["cap"].reindex(index=factor.index, columns=factor.columns)
    industry = market_data["industry"].reindex(
        index=factor.index, columns=factor.columns
    )
    st_status = market_data["st"].reindex(
        index=open_prices.index, columns=factor.columns
    ).fillna(False).astype(bool)

    signal_positions = open_prices.index.get_indexer(factor.index)
    if (signal_positions < 0).any():
        raise ValueError("holding_audit signal dates must exist in open-price data")
    entry_positions = signal_positions + 1
    exit_positions = entry_positions + state.context.horizon
    if len(open_prices.index) == 0 or int(exit_positions.max()) >= len(open_prices.index):
        raise ValueError("holding_audit requires complete entry and exit open prices")

    top_group = f"G{state.context.n_quantiles}"
    if top_group not in group_returns.columns or top_group not in transaction_costs.columns:
        raise ValueError(
            f"holding_audit requires {top_group} in net returns and transaction costs"
        )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - project runtime provides pyarrow
        raise RuntimeError("holding_audit requires pyarrow for Parquet output") from exc

    output_path = _next_artifact_path(state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.parquet")
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    position_count = 0
    summary_rows: list[dict[str, object]] = []

    factor_values = factor.to_numpy(dtype=float, copy=False)
    return_values = forward_return.to_numpy(dtype=float, copy=False)
    cap_values = cap.to_numpy(dtype=float, copy=False)
    industry_values = industry.to_numpy(copy=False)
    open_values = open_prices.to_numpy(dtype=float, copy=False)
    st_values = st_status.to_numpy(dtype=bool, copy=False)
    codes = factor.columns.astype(str).str.zfill(6).to_numpy()

    try:
        for row_number, signal_day in enumerate(factor.index):
            membership = quantile_membership(
                factor_values[row_number],
                return_values[row_number],
                n_quantiles=state.context.n_quantiles,
            )
            if membership is None:
                continue
            valid_positions, ranks, groups = membership
            top_mask = groups == state.context.n_quantiles - 1
            top_positions = valid_positions[top_mask]
            top_ranks = ranks[top_mask]
            if not len(top_positions):
                continue

            order = np.argsort(-top_ranks, kind="stable")
            top_positions = top_positions[order]
            top_ranks = top_ranks[order]
            top_count = len(top_positions)
            entry_position = int(entry_positions[row_number])
            exit_position = int(exit_positions[row_number])
            signal_label = pd.Timestamp(signal_day).date().isoformat()
            entry_label = pd.Timestamp(open_prices.index[entry_position]).date().isoformat()
            exit_label = pd.Timestamp(open_prices.index[exit_position]).date().isoformat()
            gross_return = float(return_values[row_number, top_positions].mean())
            net_return = float(group_returns.at[signal_day, top_group])
            transaction_cost = float(transaction_costs.at[signal_day, top_group])
            if not np.isclose(gross_return - transaction_cost, net_return, atol=1e-12):
                raise AssertionError(
                    "holding_audit top-group return does not reconcile with quantile_net_returns"
                )

            chunk = pd.DataFrame(
                {
                    "signal_day": signal_label,
                    "entry_day": entry_label,
                    "exit_day": exit_label,
                    "security_code": codes[top_positions],
                    "group": top_group,
                    "factor_rank": top_ranks.astype(int),
                    "rank_in_top_group": np.arange(1, top_count + 1, dtype=int),
                    "target_weight": np.full(top_count, 1.0 / top_count),
                    "factor_value": factor_values[row_number, top_positions],
                    "forward_open_return": return_values[row_number, top_positions],
                    "entry_open": open_values[entry_position, top_positions],
                    "exit_open": open_values[exit_position, top_positions],
                    "market_cap_yi": cap_values[row_number, top_positions] / 1e8,
                    "industry_l1_code": [
                        _industry_code(industry_values[row_number, position])
                        for position in top_positions
                    ],
                    "entry_is_st": st_values[entry_position, top_positions],
                },
                columns=HOLDING_AUDIT_COLUMNS,
            )
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            position_count += top_count
            summary_rows.append(
                {
                    "signal_day": pd.Timestamp(signal_day),
                    "top_group_position_count": top_count,
                    "top_group_weight": float(1.0 / top_count),
                    "top_group_gross_return": gross_return,
                    "top_group_transaction_cost": transaction_cost,
                    "top_group_net_return": net_return,
                }
            )
    finally:
        if writer is not None:
            writer.close()

    if writer is None or not summary_rows:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("holding_audit found no valid top-quantile positions")
    temporary_path.replace(output_path)
    summary = pd.DataFrame(summary_rows).set_index("signal_day")
    summary.index.name = "day"
    state.add_detail(HOLDING_AUDIT_SUMMARY, summary)
    state.add_artifact(HOLDING_AUDIT_ARTIFACT, output_path)
    state.add_metrics(
        {
            "holding_audit_signal_day_count": int(len(summary)),
            "holding_audit_position_count": int(position_count),
            "holding_audit_top_group_number": int(state.context.n_quantiles),
        }
    )
