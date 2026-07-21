"""Pipeline step applying daily industry fixed-effect neutralization."""

from __future__ import annotations

import numpy as np

from transforms import neutralize_factor_by_industry
from .base import EvaluationState, evaluation_method


@evaluation_method(
    "industry_neutralize",
    required_data_symbols=("industry",),
)
def evaluate_industry_neutralization(state: EvaluationState) -> None:
    """Replace the working factor with daily within-industry residuals."""

    if state.context.market_data is None or "industry" not in state.context.market_data:
        raise ValueError(
            "Industry neutralization requires the point-in-time industry matrix"
        )
    residuals, diagnostics = neutralize_factor_by_industry(
        state.factor,
        state.context.market_data["industry"],
    )
    valid_diagnostics = diagnostics.dropna(subset=["r_squared"])
    factor_observations = int(diagnostics["n_factor_obs"].sum())
    state.replace_factor(residuals)
    state.add_detail("industry_neutralization", diagnostics)
    state.add_metrics(
        {
            "industry_neutralized_days": int((diagnostics["n_used_obs"] > 0).sum()),
            "industry_neutralization_mean_r2": (
                float(valid_diagnostics["r_squared"].mean())
                if len(valid_diagnostics)
                else np.nan
            ),
            "industry_neutralization_covered_share": (
                float(diagnostics["n_used_obs"].sum() / factor_observations)
                if factor_observations
                else np.nan
            ),
            "industry_neutralization_unclassified_obs": int(
                diagnostics["n_unclassified_obs"].sum()
            ),
            "industry_neutralization_small_industry_obs": int(
                diagnostics["n_small_industry_obs"].sum()
            ),
        }
    )
