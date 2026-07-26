"""Leakage-safe training fitness for genetic-programming factor search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from engine import evaluate_expression, pct, restrict_evaluation_window, ts_mean, ts_std
from evaluators.tradability import mask_untradeable_entries
from returns import RETURN_DEFINITION, calculate_forward_open_return
from transforms import neutralize_factor_by_industry_and_market_cap

from .tree import ExpressionTree, canonicalize_tree


PAPER_LOCAL_STYLE_NAMES = (
    "log_total_market_cap",
    "close_return_20",
    "log_average_traded_amount_20",
    "close_return_volatility_20",
)
MARKET_CAP_INDUSTRY_MODE = "market_cap_industry"
MIN_INDUSTRY_OBSERVATIONS = 3
SUPPORTED_PREPROCESS_MODES = frozenset(
    {"paper_local", "market_cap", MARKET_CAP_INDUSTRY_MODE, "none"}
)


@dataclass(frozen=True)
class FitnessContext:
    """Training-only matrices shared by every candidate fitness evaluation."""

    market_data: Mapping[str, pd.DataFrame]
    forward_return: pd.DataFrame
    style_exposures: Mapping[str, pd.DataFrame]
    signal_index: pd.Index
    train_start: str
    train_end: str
    horizon: int
    preprocess_mode: str
    minimum_ic_days: int
    sample_metadata: Mapping[str, str]


@dataclass(frozen=True)
class FitnessResult:
    """Training fitness plus the expression direction frozen from that training data.

    ``expression`` is the unmodified GP-tree expression and remains the cache key.
    ``frozen_expression`` is the actual expression sent to the independent test
    stages.  Keeping them separate prevents a direction normalization from
    invalidating the resumable tree-level fitness cache.
    """

    expression: str
    raw_fitness: float | None
    adjusted_fitness: float | None
    ic_mean: float | None
    ic_std: float | None
    ir: float | None
    ic_count: int
    pair_count: int
    node_count: int
    depth: int
    error: str = ""
    frozen_expression: str = ""
    direction_multiplier: int = 1
    aligned_ic_mean: float | None = None
    frozen_node_count: int | None = None

    @property
    def selection_score(self) -> float:
        value = self.adjusted_fitness
        return float(value) if value is not None and np.isfinite(value) else -np.inf

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FitnessResult":
        normalized = dict(payload)
        expression = str(normalized["expression"])
        ic_mean = normalized.get("ic_mean")
        try:
            finite_ic_mean = float(ic_mean)
        except (TypeError, ValueError):
            finite_ic_mean = np.nan
        normalized.setdefault("frozen_expression", expression)
        normalized.setdefault("direction_multiplier", 1)
        normalized.setdefault(
            "aligned_ic_mean",
            abs(finite_ic_mean) if np.isfinite(finite_ic_mean) else None,
        )
        normalized.setdefault("frozen_node_count", normalized.get("node_count"))
        return cls(**normalized)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def apply_direction_multiplier(
    tree: ExpressionTree,
    direction_multiplier: int,
) -> ExpressionTree:
    """Return the canonical GP tree after applying a frozen sign direction."""

    if direction_multiplier == 1:
        return canonicalize_tree(tree)
    if direction_multiplier == -1:
        if tree.kind == "function" and tree.value == "neg":
            return canonicalize_tree(tree.children[0])
        return canonicalize_tree(ExpressionTree("function", "neg", (tree,)))
    raise ValueError("direction_multiplier must be either -1 or 1")


def freeze_expression_direction(
    tree: ExpressionTree,
    ic_mean: float,
) -> tuple[str, int, int]:
    """Orient a GP expression from its training-only signed Rank IC.

    The returned expression is also the canonical parent tree for this
    generation's selection and genetic operators.  A leading GP ``neg`` is
    simplified when applying a second negative sign, so the persisted factor is
    the shortest equivalent expression.
    """

    if not np.isfinite(ic_mean):
        raise ValueError("cannot freeze an expression direction from a non-finite IC")
    direction_multiplier = -1 if ic_mean < 0 else 1
    oriented_tree = apply_direction_multiplier(tree, direction_multiplier)
    return oriented_tree.to_expression(), direction_multiplier, oriented_tree.node_count


def build_fitness_result(
    tree: ExpressionTree,
    *,
    ic_mean: float,
    ic_std: float,
    ic_count: int,
    pair_count: int,
    parsimony_coefficient: float,
) -> FitnessResult:
    """Build signed diagnostics and an aligned selection score from training data."""

    frozen_expression, direction_multiplier, frozen_node_count = (
        freeze_expression_direction(tree, ic_mean)
    )
    aligned_ic_mean = abs(float(ic_mean))
    ir = ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan
    adjusted = aligned_ic_mean - float(parsimony_coefficient) * frozen_node_count
    return FitnessResult(
        expression=tree.to_expression(),
        # Retain the signed legacy field for diagnostics.  Direction-normalized
        # selection is represented explicitly by aligned_ic_mean/adjusted_fitness.
        raw_fitness=_finite_or_none(ic_mean),
        adjusted_fitness=_finite_or_none(adjusted),
        ic_mean=_finite_or_none(ic_mean),
        ic_std=_finite_or_none(ic_std),
        ir=_finite_or_none(ir),
        ic_count=int(ic_count),
        pair_count=int(pair_count),
        node_count=tree.node_count,
        depth=tree.depth,
        frozen_expression=frozen_expression,
        direction_multiplier=direction_multiplier,
        aligned_ic_mean=_finite_or_none(aligned_ic_mean),
        frozen_node_count=frozen_node_count,
    )


def prepare_fitness_context(
    market_data: Mapping[str, pd.DataFrame],
    *,
    train_start: str,
    train_end: str,
    horizon: int = 1,
    preprocess_mode: str = "paper_local",
    minimum_ic_days: int = 60,
) -> FitnessContext:
    """Precompute fixed training labels/styles without reading the test window."""

    if preprocess_mode not in SUPPORTED_PREPROCESS_MODES:
        raise ValueError(
            "preprocess_mode must be one of paper_local, market_cap, "
            "market_cap_industry, or none"
        )
    if minimum_ic_days < 2:
        raise ValueError("minimum_ic_days must be at least two")
    required = {"c", "o"}
    if preprocess_mode in {"paper_local", "market_cap", MARKET_CAP_INDUSTRY_MODE}:
        required.add("cap")
    if preprocess_mode == MARKET_CAP_INDUSTRY_MODE:
        required.add("industry")
    if preprocess_mode == "paper_local":
        required.update({"amt", "limit", "st"})
    missing = required - set(market_data)
    if missing:
        raise KeyError(f"Fitness preprocessing is missing market data: {sorted(missing)}")

    close = market_data["c"]
    forward_return = calculate_forward_open_return(market_data["o"], horizon)
    _, _, bounded_return, sample_metadata = restrict_evaluation_window(
        close,
        close,
        forward_return,
        signal_start=train_start,
        signal_end=train_end,
        horizon=horizon,
    )
    signal_index = bounded_return.index
    styles: dict[str, pd.DataFrame] = {}
    if preprocess_mode in {"paper_local", "market_cap", MARKET_CAP_INDUSTRY_MODE}:
        cap = market_data["cap"].reindex(index=close.index, columns=close.columns)
        styles["log_total_market_cap"] = np.log(cap.where(cap > 0)).reindex(
            index=signal_index
        )
    if preprocess_mode == "paper_local":
        amount = market_data["amt"].reindex(index=close.index, columns=close.columns)
        average_amount = ts_mean(amount, 20)
        styles.update(
            {
                "close_return_20": pct(close, 20).reindex(index=signal_index),
                # The repository has traded amount but not free-float turnover.
                # Keep this proxy explicitly named so vol/amt are never conflated.
                "log_average_traded_amount_20": np.log(
                    average_amount.where(average_amount > 0)
                ).reindex(index=signal_index),
                "close_return_volatility_20": ts_std(pct(close, 1), 20).reindex(
                    index=signal_index
                ),
            }
        )
    return FitnessContext(
        market_data=market_data,
        forward_return=bounded_return,
        style_exposures=styles,
        signal_index=signal_index,
        train_start=str(train_start),
        train_end=str(train_end),
        horizon=int(horizon),
        preprocess_mode=preprocess_mode,
        minimum_ic_days=int(minimum_ic_days),
        sample_metadata=sample_metadata,
    )


def median_mad_winsorize(
    factor: pd.DataFrame,
    *,
    mad_multiplier: float = 5.0,
) -> pd.DataFrame:
    """Paper definition: daily median +/- multiplier times unscaled MAD."""

    if not np.isfinite(mad_multiplier) or mad_multiplier <= 0:
        raise ValueError("mad_multiplier must be finite and positive")
    median = factor.median(axis=1, skipna=True)
    mad = factor.sub(median, axis=0).abs().median(axis=1, skipna=True)
    lower = median - mad_multiplier * mad
    upper = median + mad_multiplier * mad
    return factor.clip(lower=lower, upper=upper, axis=0)


def joint_cross_sectional_residualize(
    factor: pd.DataFrame,
    exposures: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Residualize each date jointly against all available style exposures."""

    if not exposures:
        return factor.copy()
    aligned = [
        frame.reindex(index=factor.index, columns=factor.columns)
        for frame in exposures.values()
    ]
    factor_values = factor.to_numpy(dtype=float, copy=False)
    exposure_values = [frame.to_numpy(dtype=float, copy=False) for frame in aligned]
    residuals = np.full(factor.shape, np.nan, dtype=float)

    for row in range(len(factor.index)):
        y_all = factor_values[row]
        x_all = np.column_stack([values[row] for values in exposure_values])
        valid = np.isfinite(y_all) & np.isfinite(x_all).all(axis=1)
        positions = np.flatnonzero(valid)
        if len(positions) < x_all.shape[1] + 3:
            continue
        y = y_all[valid]
        x = x_all[valid]
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0, ddof=0)
        informative = x_std > np.finfo(float).eps
        if informative.any():
            normalized = (x[:, informative] - x_mean[informative]) / x_std[informative]
            design = np.column_stack((np.ones(len(y)), normalized))
        else:
            design = np.ones((len(y), 1), dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals[row, positions] = y - design @ coefficients
    return pd.DataFrame(residuals, index=factor.index, columns=factor.columns)


def cross_sectional_zscore(factor: pd.DataFrame) -> pd.DataFrame:
    mean = factor.mean(axis=1, skipna=True)
    std = factor.std(axis=1, ddof=0, skipna=True).replace(0, np.nan)
    return factor.sub(mean, axis=0).div(std, axis=0)


def preprocess_training_factor(
    factor: pd.DataFrame,
    context: FitnessContext,
) -> pd.DataFrame:
    factor = factor.reindex(
        index=context.signal_index,
        columns=context.forward_return.columns,
    ).replace([np.inf, -np.inf], np.nan)
    if context.preprocess_mode == "none":
        return factor
    if context.preprocess_mode == "paper_local":
        factor, _, _ = mask_untradeable_entries(
            factor,
            context.market_data["o"],
            context.market_data["c"],
            context.market_data["limit"],
            context.market_data["st"],
        )
    winsorized = median_mad_winsorize(factor)
    if context.preprocess_mode == MARKET_CAP_INDUSTRY_MODE:
        residual, _ = neutralize_factor_by_industry_and_market_cap(
            winsorized,
            context.market_data["industry"],
            context.market_data["cap"],
            min_industry_observations=MIN_INDUSTRY_OBSERVATIONS,
        )
    else:
        residual = joint_cross_sectional_residualize(
            winsorized,
            context.style_exposures,
        )
    return cross_sectional_zscore(residual)


def daily_rank_ic(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
) -> tuple[pd.Series, int]:
    """Vectorized daily Spearman correlation with pairwise missingness."""

    forward_return = forward_return.reindex(index=factor.index, columns=factor.columns)
    valid = factor.notna() & forward_return.notna()
    counts = valid.sum(axis=1)
    factor_rank = factor.where(valid).rank(axis=1, method="average")
    return_rank = forward_return.where(valid).rank(axis=1, method="average")
    factor_centered = factor_rank.sub(factor_rank.mean(axis=1), axis=0)
    return_centered = return_rank.sub(return_rank.mean(axis=1), axis=0)
    numerator = (factor_centered * return_centered).sum(axis=1, min_count=1)
    denominator = np.sqrt(
        factor_centered.pow(2).sum(axis=1, min_count=1)
        * return_centered.pow(2).sum(axis=1, min_count=1)
    )
    ic = numerator.div(denominator).where(counts >= 3).replace(
        [np.inf, -np.inf], np.nan
    )
    return ic.dropna().astype(float), int(counts.where(counts >= 3, 0).sum())


def evaluate_processed_expression(
    expression: str,
    context: FitnessContext,
) -> pd.DataFrame:
    raw = evaluate_expression(expression, context.market_data).replace(
        [np.inf, -np.inf], np.nan
    )
    raw = raw.reindex(
        index=context.market_data["c"].index,
        columns=context.market_data["c"].columns,
    )
    return preprocess_training_factor(raw, context)


def evaluate_program_fitness(
    tree: ExpressionTree,
    context: FitnessContext,
    *,
    parsimony_coefficient: float,
) -> FitnessResult:
    expression = tree.to_expression()
    try:
        factor = evaluate_processed_expression(expression, context)
        ic, pair_count = daily_rank_ic(factor, context.forward_return)
        if len(ic) < context.minimum_ic_days:
            raise ValueError(
                f"only {len(ic)} finite IC days; requires {context.minimum_ic_days}"
            )
        ic_mean = float(ic.mean())
        ic_std = float(ic.std(ddof=1))
        return build_fitness_result(
            tree,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_count=len(ic),
            pair_count=pair_count,
            parsimony_coefficient=parsimony_coefficient,
        )
    except Exception as exc:
        return FitnessResult(
            expression=expression,
            raw_fitness=None,
            adjusted_fitness=None,
            ic_mean=None,
            ic_std=None,
            ir=None,
            ic_count=0,
            pair_count=0,
            node_count=tree.node_count,
            depth=tree.depth,
            error=f"{type(exc).__name__}: {exc}",
        )


def fitness_contract(context: FitnessContext) -> dict[str, Any]:
    contract = {
        "train_start": context.train_start,
        "train_end": context.train_end,
        "horizon": context.horizon,
        "return_definition": RETURN_DEFINITION,
        "preprocess_mode": context.preprocess_mode,
        "style_exposures": list(context.style_exposures),
        "minimum_ic_days": context.minimum_ic_days,
        "training_ic_direction_normalization": (
            "signed_rank_ic_to_positive_generation_genotype"
        ),
        **context.sample_metadata,
    }
    if context.preprocess_mode == MARKET_CAP_INDUSTRY_MODE:
        contract.update(
            {
                "neutralization_model": (
                    "factor ~ intercept + log(total_market_cap) + "
                    "industry_l1_fixed_effects"
                ),
                "minimum_industry_observations": MIN_INDUSTRY_OBSERVATIONS,
            }
        )
    return contract
