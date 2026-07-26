"""Built-in evaluation-method registry and dependency resolver."""

from __future__ import annotations

from collections.abc import Iterable

from .base import (
    REGISTERED_EVALUATION_METHODS,
    EvaluationMethod,
    method_metadata,
)
from .cycle_context import apply_cycle_context
from .fitness import evaluate_fitness
from .future_perturbation import evaluate_future_data_perturbation
from .ic_horizon_decay import evaluate_ic_horizon_decay
from .ic_peak_decay import evaluate_ic_peak_decay
from .ic_trend_filter import evaluate_ic_trend_filter
from .industry_neutralization import evaluate_industry_neutralization
from .industry_market_cap_neutralization import (
    evaluate_industry_market_cap_neutralization,
)
from .market_cap_neutralization import evaluate_market_cap_neutralization
from .newey_west import evaluate_newey_west_ic_significance
from .prefix_truncation import evaluate_prefix_truncation_consistency
from .quantile_net_returns import evaluate_quantile_net_returns
from .quantile_plot import plot_quantile_cumulative
from .quantile_returns import (
    evaluate_quantile_cumulative,
    evaluate_quantile_returns,
)
from .rank_ic import evaluate_rank_ic, evaluate_rank_icir
from .rolling_drawdown import evaluate_rolling_drawdown
from .rolling_sharpe import evaluate_rolling_sharpe
from .top_quantile import evaluate_top_quantile_performance
from .tradability import evaluate_tradability_filter


# This tuple is the default evaluation pipeline. Adding or removing a function
# here changes which methods the main factor-evaluation flow runs by default.
DEFAULT_EVALUATION_METHODS: tuple[EvaluationMethod, ...] = (
    evaluate_rank_ic,
    evaluate_rank_icir,
    evaluate_quantile_returns,
    evaluate_quantile_cumulative,
    plot_quantile_cumulative,
)


def resolve_evaluation_methods(
    names: str | Iterable[str] | None,
) -> tuple[EvaluationMethod, ...]:
    """Resolve method names and automatically place dependencies first."""

    if names is None:
        return DEFAULT_EVALUATION_METHODS
    requested = (
        [part.strip() for part in names.split(",")]
        if isinstance(names, str)
        else [str(part).strip() for part in names]
    )
    requested = [name for name in requested if name]
    if not requested or requested == ["default"]:
        return DEFAULT_EVALUATION_METHODS

    resolved: list[EvaluationMethod] = []
    visited: set[str] = set()
    visiting: set[str] = set()
    provided: set[str] = set()

    def visit(name: str, *, explicit: bool = False) -> None:
        if name in visited:
            return
        if name in provided and not explicit:
            return
        if name in visiting:
            raise ValueError(f"Circular evaluation-method dependency at {name!r}")
        if name not in REGISTERED_EVALUATION_METHODS:
            available = ", ".join(sorted(REGISTERED_EVALUATION_METHODS))
            raise ValueError(
                f"Unknown evaluation method {name!r}; available methods: {available}"
            )
        visiting.add(name)
        method = REGISTERED_EVALUATION_METHODS[name]
        for dependency in method_metadata(method).requires:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        resolved.append(method)
        provided.update(method_metadata(method).provides)

    for requested_name in requested:
        visit(requested_name, explicit=True)
    return tuple(resolved)


def available_evaluation_methods() -> tuple[str, ...]:
    return tuple(sorted(REGISTERED_EVALUATION_METHODS))
