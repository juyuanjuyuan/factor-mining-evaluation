"""Pipeline step applying daily OLS market-cap neutralization."""

from __future__ import annotations

import numpy as np

from transforms import neutralize_factor_by_market_cap
from .base import EvaluationState, evaluation_method


@evaluation_method(
    "market_cap_neutralize",
    required_data_symbols=("cap",),
)
def evaluate_market_cap_neutralization(state: EvaluationState) -> None:
    if state.context.market_data is None or "cap" not in state.context.market_data:
        raise ValueError("Market-cap neutralization requires the cap matrix")
    residuals, diagnostics = neutralize_factor_by_market_cap(
        state.factor,
        state.context.market_data["cap"],
    )
    valid_diagnostics = diagnostics.dropna(subset=["r_squared"])
    state.replace_factor(residuals)
    state.add_detail("market_cap_neutralization", diagnostics)
    state.add_metrics(
        {
            "market_cap_neutralized_days": int(
                residuals.notna().any(axis=1).sum()
            ),
            "market_cap_neutralization_mean_r2": (
                float(valid_diagnostics["r_squared"].mean())
                if len(valid_diagnostics)
                else np.nan
            ),
        }
    )
