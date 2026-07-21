"""Daily point-in-time industry neutralization for wide factor matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_factor_by_industry(
    factor: pd.DataFrame,
    industry: pd.DataFrame,
    *,
    min_industry_observations: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demean a factor within each daily industry cross-section.

    Only finite factor values with a same-day industry classification are
    eligible. Industries smaller than ``min_industry_observations`` are
    excluded instead of producing unstable residuals. This is equivalent to
    residualizing a factor on a daily intercept plus industry fixed effects.
    """

    if min_industry_observations < 3:
        raise ValueError("min_industry_observations must be at least three")

    industry = industry.reindex(index=factor.index, columns=factor.columns)
    factor_values = factor.to_numpy(dtype=float, copy=False)
    industry_values = industry.to_numpy(dtype=object, copy=False)
    residual_values = np.full(factor.shape, np.nan, dtype=float)
    diagnostic_rows: list[dict[str, float | int | object]] = []

    for row_number, day in enumerate(factor.index):
        factor_row = factor_values[row_number]
        industry_row = industry_values[row_number]
        finite_factor = np.isfinite(factor_row)
        industry_labels = pd.Series(industry_row, dtype="string").str.strip()
        classified = industry_labels.notna().to_numpy() & industry_labels.ne("").to_numpy()
        eligible = finite_factor & classified
        positions = np.flatnonzero(eligible)

        industry_count = 0
        eligible_industry_count = 0
        used_positions = np.empty(0, dtype=int)
        residual = np.empty(0, dtype=float)
        if len(positions):
            labels = industry_labels.iloc[positions].to_numpy(dtype=str)
            _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
            industry_count = int(len(counts))
            valid_groups = counts >= min_industry_observations
            use_row = valid_groups[inverse]
            used_positions = positions[use_row]
            eligible_industry_count = int(valid_groups.sum())
            if len(used_positions):
                used_values = factor_row[used_positions]
                used_inverse = inverse[use_row]
                group_sums = np.bincount(used_inverse, weights=used_values)
                group_sizes = np.bincount(used_inverse)
                residual = used_values - group_sums[used_inverse] / group_sizes[used_inverse]
                residual_values[row_number, used_positions] = residual

        used_values = factor_row[used_positions]
        centered = used_values - used_values.mean() if len(used_values) else used_values
        total_sum_squares = float(np.dot(centered, centered))
        residual_sum_squares = float(np.dot(residual, residual))
        diagnostic_rows.append(
            {
                "day": day,
                "n_factor_obs": int(finite_factor.sum()),
                "n_classified_obs": int(eligible.sum()),
                "n_used_obs": int(len(used_positions)),
                "n_unclassified_obs": int((finite_factor & ~classified).sum()),
                "n_small_industry_obs": int(len(positions) - len(used_positions)),
                "industry_count": industry_count,
                "eligible_industry_count": eligible_industry_count,
                "r_squared": (
                    1.0 - residual_sum_squares / total_sum_squares
                    if total_sum_squares > 0
                    else np.nan
                ),
                "residual_std": (
                    float(np.std(residual, ddof=1)) if len(residual) > 1 else np.nan
                ),
            }
        )

    residuals = pd.DataFrame(
        residual_values,
        index=factor.index,
        columns=factor.columns,
    )
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("day")
    return residuals, diagnostics
