"""Pipeline step applying joint industry and market-cap neutralization."""

from __future__ import annotations

import numpy as np

from transforms import neutralize_factor_by_industry_and_market_cap
from .base import EvaluationState, evaluation_method


@evaluation_method(
    "industry_market_cap_neutralize",
    required_data_symbols=("cap", "industry"),
)
def evaluate_industry_market_cap_neutralization(state: EvaluationState) -> None:
    """Replace the working factor with joint industry/size OLS residuals."""

    market_data = state.context.market_data
    if market_data is None or "cap" not in market_data or "industry" not in market_data:
        raise ValueError(
            "Joint industry/market-cap neutralization requires cap and "
            "point-in-time industry matrices"
        )
    residuals, diagnostics = neutralize_factor_by_industry_and_market_cap(
        state.factor,
        market_data["industry"],
        market_data["cap"],
    )
    valid_diagnostics = diagnostics.dropna(subset=["r_squared"])
    factor_observations = int(diagnostics["n_factor_obs"].sum())
    state.replace_factor(residuals)
    state.add_detail("industry_market_cap_neutralization", diagnostics)
    state.add_metrics(
        {
            "industry_market_cap_neutralized_days": int(
                (diagnostics["n_used_obs"] > 0).sum()
            ),
            "industry_market_cap_neutralization_mean_r2": (
                float(valid_diagnostics["r_squared"].mean())
                if len(valid_diagnostics)
                else np.nan
            ),
            "industry_market_cap_neutralization_covered_share": (
                float(diagnostics["n_used_obs"].sum() / factor_observations)
                if factor_observations
                else np.nan
            ),
            "industry_market_cap_neutralization_unclassified_obs": int(
                diagnostics["n_unclassified_obs"].sum()
            ),
            "industry_market_cap_neutralization_invalid_cap_obs": int(
                diagnostics["n_invalid_cap_obs"].sum()
            ),
            "industry_market_cap_neutralization_small_industry_obs": int(
                diagnostics["n_small_industry_obs"].sum()
            ),
        }
    )
