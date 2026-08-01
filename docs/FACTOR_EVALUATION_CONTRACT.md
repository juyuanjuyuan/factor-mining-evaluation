# Factor Evaluation Contract

## Contents

- Input data
- Expression namespace
- Evaluation definitions
- Output layout
- Metrics schema
- Review checklist

## Input Data

Expect wide pandas DataFrames with trading days on the index and security codes on the columns.
The default parquet mapping is:

| Symbol | Meaning | Default file |
|---|---|---|
| `c` | close | `close_df.pq` |
| `o` | open | `open_df.pq` |
| `h` | high | `high_df.pq` |
| `l` | low | `low_df.pq` |
| `vol` | exchange-reported turnover volume | `volume_df.pq` |
| `amt` | traded amount | `amount_df.pq` |
| `vwap` | proxy `(high + low) / 2`, not exchange VWAP | `vwap_proxy_df.pq` |
| `cap` | total market capitalization | `market_cap_df.pq` |
| `limit` | daily price-limit ratio proxy | `limit_ratio_df.pq` |
| `st` | ST/*ST status, normalized to boolean wide matrix | `st_status_df.pq` |
| `industry` | evaluator-only point-in-time一级行业分类 | `行业数据.parquet` |

The close matrix is always loaded as the canonical alignment axis. The open matrix is always loaded
because it defines the return label. Expression matrices are loaded only when their symbols occur
in the expression; evaluator-only inputs are loaded only when the selected method declares them.
Every input is then aligned to close by index and columns.

`industry` is not an expression symbol. Its canonical input is a long table with
`trade_date`, six-digit `security_code`, and `industry_l1_code`. Each classification is matched to
the factor exposure date `t` only; missing dates remain missing and are never forward- or
back-filled.

The factor at day `t` is calculated after that day's close, so it cannot trade at day `t`.
The position enters at `open[t + 1]`. For holding horizon `H`, the label is:

```text
r_H(t) = open[t + 1 + H] / open[t + 1] - 1
```

Thus the default `H=1` label is exactly `open[t + 2] / open[t + 1] - 1`.
Factor expressions may only use information available at or before day `t`.

### Run-level factor decay

Before any evaluator consumes a score (including IC, neutralization,
tradability filtering, and quantile portfolios), an optional integer `decay`
is applied to the completed factor matrix. The default is `1`, exactly
preserving all previous evaluations. For `n > 1`, each security's score on
day `t` is:

```text
decay_n(x_t) = (n*x_t + (n-1)*x_(t-1) + ... + 1*x_(t-n+1)) / (n + ... + 1)
```

At the beginning of available history, the average uses only the observations
already available. Missing factor values do not contribute weight. Input must
be a positive integer: `1` means no extra smoothing, and zero, negative, or
fractional values are rejected. Decay uses only dates at or before `t`, so it
does not alter the next-open return-label convention.

This is a run-level **factor preprocessor**, not an evaluator in the ordered
pipeline: `src/transforms/linear_decay.py` owns its validation and numerical
implementation, while the engine invokes its stable `apply_linear_decay`
boundary exactly once before constructing `EvaluationContext`. Causality
diagnostics reuse that same transform module when rebuilding a factor, so no
diagnostic carries a private copy of the formula.

### Evaluation date window

`evaluate_factor_expression` accepts optional `signal_start` and `signal_end` boundaries. The CLI
exposes them as the paired `--signal-start YYYY-MM-DD --signal-end YYYY-MM-DD` flags. Both dates are
inclusive requested sample boundaries and refer to factor signal dates, not entry or exit dates.

The engine evaluates the expression against the full available history first, then slices the
factor, close, and forward-return matrices. This preserves legitimate rolling lookback before
`signal_start`. For a bounded window it excludes the final `horizon + 1` signal rows, ensuring that
every entry at `open[t+1]` and exit at `open[t+1+horizon]` remain inside the requested period.

Persisted metadata distinguishes the request from the usable sample:

- `signal_start`, `signal_end`: requested inclusive boundaries;
- `sample_start_day`, `sample_end_day`: first and last signal days actually evaluated after trading
  calendar alignment and end-of-window label containment.

Use non-overlapping training and testing periods for sample-split research. Candidate generation,
selection, and tuning belong only to training. Freeze the expression and evaluation configuration
before the testing run.

## Expression Namespace

Supported data symbols: `c`, `o`, `h`, `l`, `vol`, `amt`, `vwap`, `cap`,
`limit`, `st`.

`industry` is deliberately evaluator-only and cannot appear in a factor expression.

Supported operators:

| Operator | Definition |
|---|---|
| `ts_mean(x, w)` | rolling mean |
| `ts_std(x, w)` | rolling population standard deviation (`ddof=0`) |
| `ts_sum(x, w)` | rolling sum |
| `ts_min(x, w)` | rolling minimum |
| `ts_max(x, w)` | rolling maximum |
| `ts_median(x, w)` | rolling median |
| `ts_product(x, w)` | rolling product |
| `ts_count(cond, w)` | rolling count of true condition values |
| `ts_corr(x, y, w)` | rolling Pearson correlation |
| `ts_cov(x, y, w)` | rolling population covariance |
| `ts_rank(x, w)` | percentile rank of the latest value in its rolling window |
| `ts_argmax(x, w)`, `ts_argmin(x, w)` | one-based location of the extreme in the trailing window |
| `delay(x, p)` | lag by `p` rows |
| `delta(x, p)` | `x - delay(x, p)` |
| `pct(x, p)` | percentage change over `p` rows |
| `rank_cs(x)` | daily cross-sectional percentile rank |
| `winsorize_cs(x, lo, hi)` | daily cross-sectional clipping at lower/upper quantiles |
| `zscore_cs(x)` | daily cross-sectional z-score using population standard deviation |
| `signed_power(x, a)` | `sign(x) * abs(x) ** a` |
| `decay_linear(x, w)` | linearly weighted mean, newest row weighted most |
| `ts_sma(x, n, m)` | recursive Chinese technical-analysis SMA: `(m*x + (n-m)*previous) / n` |
| `ts_linear_reg_slope(x, w)` | trailing least-squares slope against the time sequence |
| `ts_cum_sum(x)` | cumulative time-axis sum for each security |
| `ts_accumulate_positive_return(x)` | GTJA191 Alpha143's source-defined stateful recurrence |
| `scale_cs(x, a)` | row scaling such that `sum(abs(x)) == a` |
| `where(cond, x, y)` | axis-preserving elementwise conditional |
| `elementwise_min(x, y)`, `elementwise_max(x, y)` | pairwise extrema |
| `adv(amt, w)` | trailing average daily traded amount |
| `abs`, `log`, `log1p`, `sqrt`, `exp`, `sign` | elementwise NumPy functions |

Alpha101 non-integer time parameters are floored to whole trading days, as
specified by the paper. Evaluation horizons and quantile counts remain strict
positive integers.

Selected safe NumPy calls such as `np.abs`, `np.log`, `np.where`, `np.maximum`, and `np.minimum`
are allowed. Arbitrary Python, imports, dunder attributes, DataFrame methods, and filesystem access
are rejected.

Add a genuinely reusable missing operator to `src/engine.py`, its namespace, and
its tests before using it in generated factors.

## Evaluation Definitions

Evaluation definitions are implemented as registered functions under
`src/evaluators/`.
The main flow executes an ordered function list and merges each function's metrics, detail tables,
and artifacts. The standard default remains the complete Rank IC/IR plus quantile-return workflow.

### Method 1: IC and IR

For each date, compute Spearman rank correlation across securities between the factor and forward
return. Require at least three valid pairs and nonconstant factor and return ranks.

- `ic_mean`: arithmetic mean of valid daily IC values.
- `ic_std`: sample standard deviation of daily IC values (`ddof=1`).
- `ir`: signed `ic_mean / ic_std`.
- `ic_positive_ratio`: fraction of valid daily IC values above zero.

### Method 2: Quantile Returns

Within each date, rank valid securities by factor value and split them as evenly as possible into
`N` ascending groups. Compute each group's equal-weight mean forward return. `G1` is the lowest
factor group, `GN` is the highest, and `Long-Short = GN - G1`.

Compound each daily series as `(1 + return).cumprod() - 1`, matching the original notebook.

## Output Layout

For output root `outputs/factor_evaluation/custom` and factor `factor_test1`:

```text
outputs/factor_evaluation/custom/
├── metrics.csv
├── metrics_history.csv
├── plots/
│   └── factor_test1.png
├── code/
│   ├── factor_test1.py
│   └── history/
│       └── factor_test1__<run_id>.py
└── details/
    ├── factor_test1__ic.csv
    ├── factor_test1__group_returns.csv
    └── factor_test1__cumulative_returns.csv
```

`metrics.csv` is upserted by factor name. `metrics_history.csv` appends every successful run.
Current plot and detail filenames are stable, so downstream comparisons do not need timestamp
discovery. Test history remains timestamped to preserve every evaluated expression.

When a non-default method subset is selected, only the details and plots produced by those methods
are required. The generated test records the exact method names so the same pipeline can be rerun.

## Metrics Schema

Core columns:

| Column | Meaning |
|---|---|
| `run_id` | local timestamp plus unique suffix |
| `evaluated_at` | ISO local timestamp |
| `factor_name` | requested factor name |
| `artifact_name` | filesystem-safe factor name |
| `expression` | exact evaluated expression |
| `horizon` | number of open-to-open holding periods after entering at `t+1` open |
| `return_definition` | canonical formula: `open[t+1+horizon]/open[t+1]-1` |
| `n_quantiles` | number of groups |
| `decay` | post-expression linear-decay window; `1` is the historical no-decay default |
| `evaluation_methods` | ordered comma-separated method names used by this run |
| `evaluation_details` | JSON map of method detail names to relative CSV paths |
| `evaluation_artifacts` | JSON map of method artifact names to relative paths |
| `signal_start`, `signal_end` | requested bounded signal-date period; absent for a full-history run |
| `sample_start_day`, `sample_end_day` | actual evaluated signal-date range after calendar alignment and label-safe tail trimming |
| `ic_mean`, `ic_std`, `ir` | IC summary |
| `ic_positive_ratio`, `ic_count` | IC breadth |
| `pair_count` | total valid factor/return observations |
| `start_day`, `end_day` | valid IC date range |
| `g1_final_cumulative` | final cumulative return of lowest group |
| `gn_final_cumulative` | final cumulative return of highest group |
| `long_short_final_cumulative` | final cumulative highest-minus-lowest return |

## Review Checklist

- Confirm the expression contains no future shift or forward return.
- Confirm price/amount units and corporate-action treatment are suitable.
- Confirm IC sign agrees with the intended long direction.
- Inspect missing days, `ic_count`, and `pair_count`.
- Inspect group ordering rather than only endpoints.
- Compare `metrics.csv` across factors only when `return_definition`, horizon, universe, and date
  range all match.
- Treat extreme values as a prompt for winsorization/neutralization analysis, not automatic alpha.
