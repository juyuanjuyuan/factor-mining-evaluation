"""Public API for composable factor evaluation methods."""

from .base import (
    EvaluationContext,
    EvaluationMethod,
    EvaluationState,
    evaluation_method,
    evaluation_method_names,
    evaluation_required_data_symbols,
    run_evaluation_methods,
)
from .cycle_context import apply_cycle_context
from .future_perturbation import (
    evaluate_future_data_perturbation,
    future_data_perturbation_test,
    perturb_data_after,
)
from .ic_horizon_decay import (
    IC_HORIZON_DECAY_MAX_HORIZON,
    IC_HORIZON_DECAY_MIN_HORIZON,
    evaluate_ic_horizon_decay,
    ic_horizon_decay,
)
from .ic_peak_decay import evaluate_ic_peak_decay, ic_peak_decay
from .ic_trend_filter import (
    compare_ic_trend_filters,
    evaluate_ic_trend_filter,
    fourier_low_pass_ic,
    kalman_ic_trend,
    overlapping_filtered_ic_mean,
    second_order_low_pass_ic,
)
from .industry_neutralization import evaluate_industry_neutralization
from .industry_market_cap_neutralization import (
    evaluate_industry_market_cap_neutralization,
)
from .market_cap_neutralization import evaluate_market_cap_neutralization
from .newey_west import (
    evaluate_newey_west_ic_significance,
    newey_west_mean_test,
    select_newey_west_lag,
)
from .prefix_truncation import (
    evaluate_prefix_truncation_consistency,
    prefix_truncation_consistency_test,
    truncate_data_through,
)
from .quantile_net_returns import (
    calculate_quantile_net_returns,
    evaluate_quantile_net_returns,
)
from .quantile_plot import plot_group_cumulative, plot_quantile_cumulative
from .quantile_returns import (
    assign_quantile,
    calc_group_returns,
    evaluate_quantile_cumulative,
    evaluate_quantile_returns,
)
from .rank_ic import (
    calc_ic,
    evaluate_rank_ic,
    evaluate_rank_icir,
    get_ic_info,
    spearman_rank_correlation,
)
from .rolling_drawdown import evaluate_rolling_drawdown, rolling_max_drawdown
from .rolling_sharpe import evaluate_rolling_sharpe, rolling_sharpe
from .top_quantile import (
    evaluate_top_quantile_performance,
    non_overlapping_top_quantile_leadership,
    top_quantile_performance,
)
from .tradability import (
    evaluate_tradability_filter,
    mask_untradeable_entries,
    open_limit_entry_masks,
)
from .registry import (
    DEFAULT_EVALUATION_METHODS,
    available_evaluation_methods,
    resolve_evaluation_methods,
)

__all__ = [
    "DEFAULT_EVALUATION_METHODS",
    "EvaluationContext",
    "EvaluationMethod",
    "EvaluationState",
    "assign_quantile",
    "available_evaluation_methods",
    "calc_group_returns",
    "calc_ic",
    "calculate_quantile_net_returns",
    "compare_ic_trend_filters",
    "apply_cycle_context",
    "evaluate_quantile_cumulative",
    "evaluate_quantile_net_returns",
    "evaluate_quantile_returns",
    "evaluate_future_data_perturbation",
    "evaluate_ic_horizon_decay",
    "evaluate_ic_peak_decay",
    "evaluate_ic_trend_filter",
    "evaluate_industry_market_cap_neutralization",
    "evaluate_industry_neutralization",
    "evaluate_market_cap_neutralization",
    "evaluate_newey_west_ic_significance",
    "evaluate_prefix_truncation_consistency",
    "evaluate_rank_ic",
    "evaluate_rank_icir",
    "evaluate_rolling_drawdown",
    "evaluate_rolling_sharpe",
    "evaluate_top_quantile_performance",
    "evaluate_tradability_filter",
    "evaluation_method",
    "evaluation_method_names",
    "evaluation_required_data_symbols",
    "future_data_perturbation_test",
    "fourier_low_pass_ic",
    "get_ic_info",
    "IC_HORIZON_DECAY_MAX_HORIZON",
    "IC_HORIZON_DECAY_MIN_HORIZON",
    "ic_horizon_decay",
    "ic_peak_decay",
    "kalman_ic_trend",
    "plot_group_cumulative",
    "plot_quantile_cumulative",
    "prefix_truncation_consistency_test",
    "perturb_data_after",
    "mask_untradeable_entries",
    "newey_west_mean_test",
    "open_limit_entry_masks",
    "overlapping_filtered_ic_mean",
    "rolling_max_drawdown",
    "rolling_sharpe",
    "non_overlapping_top_quantile_leadership",
    "resolve_evaluation_methods",
    "run_evaluation_methods",
    "select_newey_west_lag",
    "second_order_low_pass_ic",
    "spearman_rank_correlation",
    "top_quantile_performance",
    "truncate_data_through",
]
