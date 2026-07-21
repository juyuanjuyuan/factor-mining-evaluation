---
name: evaluate-stock-factors
description: "Evaluate, compare, and rerun equity factor expressions in this repository, or add and modify its registered factor-evaluation methods. Use for IC/IR and quantile backtests, archived evaluation artifacts, ordered pipelines and neutralization state, evaluator metrics/details/artifacts, dependencies, required market-data symbols, webpage mathematical definitions, frontend labels, and webapp compatibility."
---

# Evaluate Stock Factors

Use the project runner and registered evaluation pipeline to keep factor results comparable,
reproducible, and consumable by the webapp.

Read `AGENTS.md` first. Read
[the evaluation contract](../../docs/FACTOR_EVALUATION_CONTRACT.md) when data names, return labels,
metrics, operators, or output formats matter. When adding or changing evaluation code, also read
[the evaluator extension reference](references/evaluator-extension.md) before editing.

## Runtime

Use the existing `rdagent` environment without installing or changing dependencies:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python <script>
```

Prefix plotting batch runs with `MPLCONFIGDIR=/private/tmp/matplotlib`.

## Evaluate a Factor

1. Confirm the expression uses only information available by day `t` close.
2. Preserve the canonical label:
   `open[t+1+horizon] / open[t+1] - 1`.
3. Keep `vol` (volume) and `amt` (traded amount) distinct.
4. Give the factor a stable, unique name without path separators.
5. Run:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_evaluation.py \
  --factor-name factor_test1 \
  --expression 'ts_mean(abs((h-l)/(h+l+1e-6)), 20)'
```

6. Use `--methods`, `--horizon`, `--quantiles`, file overrides, and `--output-dir` only when the
   requested evaluation differs from project defaults.
7. Treat missing or empty metrics, details, code snapshots, or required plots as a failed run.

Report at least the expression, factor name, signed IC mean, IC standard deviation, signed and
absolute IR, IC count, and final long-short cumulative return. Flag non-finite values.

## Understand the Ordered Pipeline

The default method order is:

```text
rank_ic → rank_icir → quantile_returns → quantile_cumulative → quantile_plot
```

Dependencies are inserted for ordinary method subsets. The webapp may also store and execute an
explicit pipeline exactly as ordered, including repeated methods.

`EvaluationContext.factor` is the immutable original factor.
`EvaluationState.factor` is the current working factor. A transform such as neutralization replaces
the working factor, so later methods that directly read `state.factor` automatically use the
transformed values.

A transform does not recompute earlier details. For example:

```text
rank_ic → rank_icir → market_cap_neutralize → rank_ic → rank_icir
```

The second IC and ICIR use the neutralized factor. But this pipeline:

```text
rank_ic → market_cap_neutralize → rank_icir
```

still summarizes the earlier raw IC. Rerun `rank_ic` or `quantile_returns` after a transform before
requesting transformed summaries or backtests.

Repeated outputs are preserved with `__2`, `__3`, and later suffixes. Consumers using
`require_detail()` or `require_cache()` read the newest version.

## Verify Outputs

Inspect:

- `metrics.csv` and append-only `metrics_history.csv`;
- `details/` daily IC, group-return, and cumulative-return CSVs;
- `plots/<factor>.png` when plotting is selected;
- editable `code/<factor>.py` and immutable `code/history/` snapshots.

Compare factors only when their return definition, horizon, universe, date range, data basis, and
evaluation method order match. Do not select factors from absolute IR alone; check signed IC,
stability, group ordering, long-short direction, and economic rationale.

Preserve failed generated code for diagnosis. Reuse a factor name only for a correction; use a new
name for a distinct expression.

## Add or Modify an Evaluation Method

Follow [references/evaluator-extension.md](references/evaluator-extension.md). In particular:

- implement through `EvaluationState`, not hard-coded engine schemas;
- preserve current-factor state and repeated-output versioning;
- update evaluator registration, tests, the webpage method definition, frontend labels, and webapp
  contracts together;
- after changing an evaluator's behavior or meaning, audit the corresponding webapp user-facing
  text in `/methods`, metric cards, tooltips, detail titles, compare tables, and any fallback labels
  so the UI describes the implemented logic rather than stale internal field names;
- treat the webpage definition as part of the evaluator contract: every registered method must
  document its Chinese name, mathematical formula, implementation definition, interpretation,
  dependencies, and required data in the `/methods` evaluation-module library before the change is
  complete;
- write formulas as valid LaTeX and render them through the shared KaTeX component; Unicode or
  plain-text formula approximations do not satisfy the webpage-definition requirement;
- keep `/pipelines` focused on pipeline-template composition; do not put the full mathematical
  method-definition catalog back on that page;
- add a method to the default tuple only if every normal evaluation should run it;
- run all required library and webapp checks before reporting completion.

## Required Verification

Use the exact checklist in `AGENTS.md`. The minimum evaluation-code checks are:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python tests/test_evaluation_methods.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python tests/test_factor_evaluation.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python tests/test_extended_evaluators.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python webapp/server/tests/test_api.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python webapp/server/tests/test_worker_lifecycle.py
```

If persisted metrics, details, or artifacts change, also rebuild the frontend when applicable and
complete one real webapp evaluation. State explicitly when any required check cannot run.
