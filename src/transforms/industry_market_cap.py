"""Daily joint industry fixed-effect and market-cap neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_factor_by_industry_and_market_cap(
    factor: pd.DataFrame,
    industry: pd.DataFrame,
    market_cap: pd.DataFrame,
    *,
    min_industry_observations: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily OLS residuals from industry fixed effects plus log market cap.

    The fitted cross-section is ``factor ~ intercept + log(cap) + industry``.
    A reference-industry coding with an intercept is used internally; fitted
    values and residuals are consequently invariant to the chosen reference.
    Industries with fewer than ``min_industry_observations`` eligible stocks are
    excluded before fitting, matching :func:`neutralize_factor_by_industry`.
    """

    if min_industry_observations < 3:
        raise ValueError("min_industry_observations must be at least three")

    industry = industry.reindex(index=factor.index, columns=factor.columns)
    market_cap = market_cap.reindex(index=factor.index, columns=factor.columns)
    factor_values = factor.to_numpy(dtype=float, copy=False)
    industry_values = industry.to_numpy(dtype=object, copy=False)
    cap_values = market_cap.to_numpy(dtype=float, copy=False)
    residual_values = np.full(factor.shape, np.nan, dtype=float)
    diagnostic_rows: list[dict[str, float | int | object]] = []

    for row_number, day in enumerate(factor.index):
        factor_row = factor_values[row_number]
        cap_row = cap_values[row_number]
        industry_row = industry_values[row_number]
        finite_factor = np.isfinite(factor_row)
        positive_cap = np.isfinite(cap_row) & (cap_row > 0)
        industry_labels = pd.Series(industry_row, dtype="string").str.strip()
        classified = industry_labels.notna().to_numpy() & industry_labels.ne("").to_numpy()
        eligible = finite_factor & positive_cap & classified
        positions = np.flatnonzero(eligible)

        industry_count = 0
        eligible_industry_count = 0
        used_positions = np.empty(0, dtype=int)
        residual = np.empty(0, dtype=float)
        intercept = np.nan
        log_cap_beta = np.nan
        r_squared = np.nan
        residual_std = np.nan
        model_rank = 0
        n_model_parameters = 0

        if len(positions):
            labels = industry_labels.iloc[positions].to_numpy(dtype=str)
            _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
            industry_count = int(len(counts))
            valid_groups = counts >= min_industry_observations
            use_row = valid_groups[inverse]
            used_positions = positions[use_row]
            eligible_industry_count = int(valid_groups.sum())

            if len(used_positions):
                y = factor_row[used_positions]
                log_cap = np.log(cap_row[used_positions])
                _, group_codes = np.unique(inverse[use_row], return_inverse=True)
                n_groups = eligible_industry_count
                # With an intercept, omit the final industry dummy.  The
                # omitted group remains represented by the intercept.
                industry_dummies = np.zeros((len(y), max(n_groups - 1, 0)))
                if n_groups > 1:
                    dummy_rows = np.flatnonzero(group_codes < n_groups - 1)
                    industry_dummies[dummy_rows, group_codes[dummy_rows]] = 1.0
                design = np.column_stack((np.ones(len(y)), log_cap, industry_dummies))
                n_model_parameters = int(design.shape[1])
                coefficients, _, model_rank, _ = np.linalg.lstsq(design, y, rcond=None)
                model_rank = int(model_rank)
                if len(y) > model_rank:
                    fitted = design @ coefficients
                    residual = y - fitted
                    residual_values[row_number, used_positions] = residual
                    centered = y - y.mean()
                    total_sum_squares = float(np.dot(centered, centered))
                    residual_sum_squares = float(np.dot(residual, residual))
                    r_squared = (
                        1.0 - residual_sum_squares / total_sum_squares
                        if total_sum_squares > 0
                        else np.nan
                    )
                    residual_std = float(np.std(residual, ddof=1))
                    intercept = float(coefficients[0])
                    # If log-cap is a linear combination of industry dummies,
                    # its coefficient is not identified, but the projection is.
                    if model_rank == n_model_parameters:
                        log_cap_beta = float(coefficients[1])

        diagnostic_rows.append(
            {
                "day": day,
                "n_factor_obs": int(finite_factor.sum()),
                "n_classified_obs": int((finite_factor & classified).sum()),
                "n_positive_cap_obs": int((finite_factor & positive_cap).sum()),
                "n_eligible_obs": int(eligible.sum()),
                "n_used_obs": int(len(used_positions)),
                "n_unclassified_obs": int((finite_factor & ~classified).sum()),
                "n_invalid_cap_obs": int((finite_factor & classified & ~positive_cap).sum()),
                "n_small_industry_obs": int(len(positions) - len(used_positions)),
                "industry_count": industry_count,
                "eligible_industry_count": eligible_industry_count,
                "n_model_parameters": n_model_parameters,
                "model_rank": model_rank,
                "residual_degrees_of_freedom": int(len(used_positions) - model_rank),
                "intercept": intercept,
                "log_cap_beta": log_cap_beta,
                "r_squared": r_squared,
                "residual_std": residual_std,
            }
        )

    residuals = pd.DataFrame(
        residual_values,
        index=factor.index,
        columns=factor.columns,
    )
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("day")
    return residuals, diagnostics
