# AGENTS.md

This file provides guidance to AGENTS when working with code in this repository.

## Project Overview

Modular cross-sectional equity factor evaluation (因子挖掘/评价) for Chinese A-share wide-matrix data. Factors are string expressions (e.g. `ts_mean(abs((h-l)/(h+l+1e-6)), 20)`) evaluated with Rank IC/IR and quantile-portfolio returns. Docs and README are in Chinese.

**Core invariant (return label):** a factor is computed after day `t` close and trades at the next open, so the label is `open[t+1+horizon]/open[t+1] - 1` (default `H=1` → `open[t+2]/open[t+1]-1`). This is defined once in `src/returns.py` (`RETURN_DEFINITION`) and recorded in every result row. Expressions must never use information after day `t`. Archived results without a matching `return_definition` are treated as stale and rerun.

## Environment

Prefer the existing conda interpreter (has numpy/pandas/matplotlib/pyarrow); do not install or change dependencies unless asked:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python <script>
```

Batch runs that plot need `MPLCONFIGDIR=/private/tmp/matplotlib` prefixed. Scripts and tests bootstrap `src/` onto `sys.path` themselves (`tests/_bootstrap.py`, wrappers in `scripts/`), so no install is required to run anything.

## Common Commands

```bash
# Evaluate one factor (outputs to outputs/factor_evaluation/custom/ by default)
python scripts/run_factor_evaluation.py \
  --factor-name factor_test1 \
  --expression 'ts_mean(abs((h-l)/(h+l+1e-6)), 20)'
# Optional: --horizon N --quantiles N --methods rank_icir,quantile_plot --output-dir ... --close-file ...

# Alpha101 batch: dry-run precheck, then run (resumable; skips already-successful matching runs)
python scripts/run_alpha101_evaluations.py --select all --dry-run
python scripts/run_alpha101_evaluations.py --select all
# --set exact (52) | vwap_proxy (30) | market_cap (Alpha056) | runnable (all 83)
# --select 1-4,28,101   --methods <names>   --rerun (force)   --no-cache-data (low memory)

# Validate a factor-registry batch JSON
python scripts/validate_factor_registry.py factor_registry/<batch_id>.json

# Rebuild dashboard (outputs/factor_evaluation/<set>/index.html)
python scripts/build_factor_dashboard.py [--results-dir ... --title ...]
```

## Tests

Tests are standalone scripts (plain `assert` + `main()`, not pytest). Run one file directly:

```bash
python tests/test_evaluation_methods.py     # method composition & dependency resolution
python tests/test_factor_evaluation.py      # synthetic end-to-end artifact contract
python tests/test_extended_evaluators.py    # non-default evaluators
python tests/test_alpha101_registry.py      # names/expressions/exports of all 83 factors
python tests/test_market_data_contract.py   # requires real data/ parquet files
```

Most tests use synthetic data; only the market-data contract test needs the real `data/` matrices.

Webapp contract tests (same standalone style, need the `rdagent` env which has fastapi):

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python webapp/server/tests/test_api.py               # API + registry + funnel-gating contract
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python webapp/server/tests/test_worker_lifecycle.py  # worker success/crash/restart on synthetic data
```

## Architecture

Core modules and subpackages live directly under `src/`; `scripts/` are thin CLI wrappers.

- **`engine.py`** — the hub: AST-whitelist expression validation (only registered operators, listed symbols, and safe `np.*` calls; arbitrary Python is rejected), parquet loading (only matrices whose symbols appear in the expression, all aligned to the close matrix), evaluation orchestration, and persistence. New reusable operators go here (plus its namespace and tests) — never in ad hoc per-factor helpers.
- **`evaluators/`** — composable pipeline. Each method is a function taking `EvaluationState`, registered via `@evaluation_method` (in `base.py`) and contributing metrics/details/artifacts without the orchestrator hard-coding any schema. `registry.py` holds `DEFAULT_EVALUATION_METHODS` (`rank_ic → rank_icir → quantile_returns → quantile_cumulative → quantile_plot`) and `resolve_evaluation_methods`, which auto-inserts dependencies for `--methods` subsets. Extended methods (registered but not default): `future_data_perturbation`, `newey_west_ic_significance`, `top_quantile_performance`, `rolling_sharpe`, `rolling_drawdown`, `market_cap_neutralize` — see `docs/EXTENDED_EVALUATORS.md`. To add a method: one function in `evaluators/`, decorate, export from the package; add to the default tuple only if every standard run should include it.
  - **Current factor state**: `EvaluationContext.factor` is the immutable original factor; `EvaluationState.factor` is the current working factor. A transform such as neutralization reads the current factor and calls `state.replace_factor(...)`. Every later method that directly reads `state.factor` automatically uses the transformed factor. Earlier IC/group-return details are not recomputed automatically: rerun `rank_ic` or `quantile_returns` after the transform before running their summaries or downstream backtests.
  - **Repeated methods & versioned outputs**: `run_evaluation_methods` executes the list exactly as ordered and a method may appear more than once (e.g. `rank_ic` before and after `market_cap_neutralize`); each `requires` entry only needs some earlier occurrence. Repeated productions of an existing metric/detail/artifact name get versioned keys (`ic_mean`, `ic_mean__2`, …; files `<factor>__ic__2.csv`, `<factor>__2.png`), and `require_detail`/`require_cache` return the most recent version. Duplicate-free pipelines are unaffected. Do not reintroduce dedupe/reorder into `run_evaluation_methods` or the generated `code/*.py` snapshots.
- **`returns.py`** — canonical forward-open return label used by every method.
- **`model_training/`** — registered, independently replaceable multi-factor trainers used by
  `/models`. Each concrete trainer lives in one file and consumes `ModelTrainingContext`, then
  returns frozen weights, an executable expression, and diagnostics through
  `ModelTrainingResult`. The worker calls only `run_model_training`; add future OLS/Lasso/rolling
  trainers through the decorator and registry rather than branching in the router or worker. See
  `docs/MODEL_TRAINING.md`.
- **top-level `factor_registry/`** — the canonical and only runtime source of factor definitions. `alpha101_runnable_factors.json` contains 52 exact, 30 VWAP-proxy, and 1 market-cap factor. Do not duplicate formulas under `src/`. The CSV is for human inspection only.
- **`batch.py`** — resumable batch runner; a prior run is skipped only when expression, `return_definition`, horizon, quantiles, and method list all match.
- **`factor_registry.py`** + top-level `factor_registry/` — factor *definitions only* (one JSON per batch, filename = `batch_id`, schema/template inside the dir). Never write evaluation metrics (`ic_mean`, `ir`, returns) into a registry batch. Preserve per-factor `entered_at` across reruns/exports; it is distinct from `evaluated_at`.
- **`paths.py`** — project-root discovery with `FACTOR_MINING_ROOT` env override.
- **`reporting/dashboard.py`** — self-contained HTML dashboard per result set.

### Data contract (`data/`, see `docs/MARKET_DATA.md`)

Wide DataFrames: ascending `DatetimeIndex` rows × 6-digit security-code columns, all axis-aligned to `close_df.pq`; missing values stay `NaN` (never forward-filled). Expression symbols: `c/o/h/l` (adjusted OHLC), `vol` (volume), `amt` (traded amount), `vwap`, `cap`.

- **Never swap `vol` and `amt`.** Alpha101 `volume` maps to `vol`; paper `advN` maps to `adv(amt, N)`.
- `vwap` is a proxy `(high+low)/2`, not exchange VWAP — factors using it must be labeled as proxy and never compared against real-input factors as if they shared the same data basis (口径).

### Output layout (`outputs/factor_evaluation/<set>/`)

`metrics.csv` (upserted per factor, latest run), `metrics_history.csv` (append-only), `plots/<factor>.png`, `details/` (daily IC, group returns, cumulative), `code/<factor>.py` (current rerunnable test) + `code/history/` (immutable timestamped copies), `batch_status.csv` for batch runs. Treat missing or empty artifacts as a failed run. Preserve failed generated test code for diagnosis; rerun under the same factor name only if it is a correction.

## Webapp compatibility (webapp/) — REQUIRED CHECK after touching evaluation code

`webapp/` (FastAPI + React) is a thin wrapper that imports `src/` directly and reads evaluation outputs from disk. **Any change to the evaluation modules in `src/` (engine.py, evaluators/, returns.py, funnel.py, factor_registry.py) must keep its outputs consumable by the webapp, and must be verified against the webapp before the change is considered done.**

The webapp depends on this exact surface — breaking any item below breaks the platform:

1. **Import API** (`webapp/server/app/` imports these; renaming/re-signaturing requires updating the webapp in the same change):
   - `engine`: `evaluate_factor_expression(*, factor_name, expression, data_dir, output_dir, horizon, n_quantiles, preloaded_data, evaluation_methods)` returning a dict with a `metrics` mapping; `parse_and_validate_expression`, `expression_data_symbols`, `DEFAULT_FILES`
   - `evaluators`: `DEFAULT_EVALUATION_METHODS`, `evaluation_method_names`, `resolve_evaluation_methods`; `evaluators.base`: `REGISTERED_EVALUATION_METHODS`, `method_metadata` (method `name`/`requires`/`required_data_symbols`)
   - `funnel`: `FUNNEL_STAGES` (stage `.name`, `.method_names`) and `evaluate_gate(stage, metrics, significance_level=...)`
   - `factor_registry`: `load_factor_batch`, `validate_factor_batch`
2. **Result contract on disk** (webapp resolves and serves these per run):
   - `metrics["evaluation_details"]` / `metrics["evaluation_artifacts"]` must remain dicts (or JSON strings) of paths **relative to `output_dir`**; the webapp refuses paths escaping the run dir
   - Detail CSVs: `utf-8-sig`, first column = date index, numeric cells; NaN allowed (webapp sanitizes NaN/inf → null — never emit values that require custom JSON encoders)
   - `cumulative_returns`/`group_returns` must keep a `Long-Short` column (the compare page matches a column containing both "long" and "short")
   - Rolling details keep the `rolling_sharpe_<window>` / `rolling_drawdown_<window>` naming pattern
   - Repeated-method outputs keep the `__<n>` version-suffix pattern (`ic__2`, `ic_mean__2`); the webapp worker executes stored pipelines exactly as ordered via `evaluators.base.REGISTERED_EVALUATION_METHODS` (duplicates allowed, no re-resolution)
3. **New evaluators require a complete webpage definition with rendered LaTeX.** For every registered evaluation method, add its Chinese name, category, valid LaTeX formula, precise implementation definition, interpretation, and limitations to `webapp/frontend/src/lib/methodDefinitions.ts`. Render formulas through the shared `webapp/frontend/src/components/LatexFormula.tsx` KaTeX component—Unicode/plain-text approximations or formula screenshots are not acceptable. The `/methods` evaluation-module library is the canonical UI for full method definitions: it must expose the method through search/filter, render its `.katex-display` formula with MathML output, show dependencies and required data from `/api/methods`, and must not report “定义缺失” or a KaTeX parse error. Keep `/pipelines` focused on template composition; do not pile full method-definition cards back onto the template page. Add new metric/detail keys and their format/direction/title to `webapp/frontend/src/lib/metrics.ts` (`METRIC_SPECS` / `DETAIL_TITLES`). Rebuild the frontend (`cd webapp/frontend && npm run build`). A method is not complete until its evaluator, tests, LaTeX webpage definition, labels, and webapp rendering all agree.

**Verification checklist (run after every evaluation-code change):**

```bash
# 1) library-level contracts still hold
python tests/test_evaluation_methods.py && python tests/test_factor_evaluation.py

# 2) webapp contracts still hold (uses fastapi env)
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python webapp/server/tests/test_api.py
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python webapp/server/tests/test_worker_lifecycle.py

# 3) if the change altered metrics/details/artifacts: one real end-to-end run through the webapp
#    (start server, submit a job in the UI or POST /api/jobs, confirm the run page renders
#     metrics and charts without errors)
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python -m uvicorn --app-dir webapp/server app.main:app --port 8000
```

## Conventions

- Factor names are stable and unique: lowercase letters/digits/underscores/hyphens or concise Chinese; no path separators.
- Compare `metrics.csv` rows across factors only when `return_definition`, horizon, universe, and date range all match.
- Never select a factor on absolute IR alone — check signed IC, IC stability, group ordering (not just endpoints), long-short direction, and economic rationale; flag non-finite metrics instead of ranking them.
- Detailed specs: `docs/FACTOR_EVALUATION_CONTRACT.md` (metrics schema, operator namespace, review checklist), `docs/ALPHA101_EVALUATION.md`, `docs/FACTOR_REGISTRY.md`, `docs/MARKET_DATA.md`. `skills/evaluate-stock-factors/SKILL.md` contains the same workflow rules for agents.

## Project Skills

Project-local skills provide the required workflow for specialized tasks. Read the matching
`SKILL.md` completely before acting:

- **`evaluate-stock-factors`** — `skills/evaluate-stock-factors/SKILL.md`; use when generating,
  running, comparing, or rerunning factor evaluations, and when adding or modifying evaluator
  modules, state transforms, dependencies, outputs, pipeline registration, frontend labels, or
  webapp contracts. For evaluator code changes, the skill routes to its
  `references/evaluator-extension.md`.

## Imported Claude Cowork project instructions
