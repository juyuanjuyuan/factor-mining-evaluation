"""Daily cross-sectional OLS market-cap neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_factor_by_market_cap(
    factor: pd.DataFrame,
    market_cap: pd.DataFrame,
    *,
    min_observations: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return residuals from factor ~ intercept + log(total market cap)."""

    if min_observations < 3:
        raise ValueError("min_observations must be at least three")
    market_cap = market_cap.reindex(index=factor.index, columns=factor.columns)
    factor_values = factor.to_numpy(dtype=float, copy=False)
    cap_values = market_cap.to_numpy(dtype=float, copy=False)
    residual_values = np.full(factor.shape, np.nan, dtype=float)
    diagnostic_rows: list[dict[str, float | int | object]] = []

    for row_number, day in enumerate(factor.index):
        y_all = factor_values[row_number]
        cap_all = cap_values[row_number]
        valid = np.isfinite(y_all) & np.isfinite(cap_all) & (cap_all > 0)
        positions = np.flatnonzero(valid)
        if len(positions) < min_observations:
            diagnostic_rows.append(
                {
                    "day": day,
                    "n_obs": len(positions),
                    "intercept": np.nan,
                    "log_cap_beta": np.nan,
                    "r_squared": np.nan,
                    "residual_std": np.nan,
                }
            )
            continue
        y = y_all[valid]
        log_cap = np.log(cap_all[valid])
        if np.ptp(log_cap) <= np.finfo(float).eps:
            fitted = np.full_like(y, y.mean())
            intercept = float(y.mean())
            beta = np.nan
        else:
            design = np.column_stack((np.ones(len(y)), log_cap))
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ coefficients
            intercept = float(coefficients[0])
            beta = float(coefficients[1])
        residual = y - fitted
        residual_values[row_number, positions] = residual
        centered = y - y.mean()
        total_sum_squares = float(np.dot(centered, centered))
        residual_sum_squares = float(np.dot(residual, residual))
        r_squared = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else np.nan
        )
        diagnostic_rows.append(
            {
                "day": day,
                "n_obs": len(positions),
                "intercept": intercept,
                "log_cap_beta": beta,
                "r_squared": r_squared,
                "residual_std": float(np.std(residual, ddof=1)),
            }
        )

    residuals = pd.DataFrame(
        residual_values,
        index=factor.index,
        columns=factor.columns,
    )
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("day")
    return residuals, diagnostics
