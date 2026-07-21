# Evaluator Extension Reference

Use this reference when adding, modifying, or reviewing evaluation modules, transforms, metrics,
details, artifacts, required data inputs, or pipeline behavior.

## Inspect and Classify

1. Inspect `src/evaluators/base.py`, `src/evaluators/registry.py`, a similar evaluator, its tests,
   and `webapp/frontend/src/lib/metrics.ts`.
2. Classify the method as:
   - a direct factor consumer;
   - a derived-result consumer;
   - a factor-state transform;
   - an artifact producer;
   - a new market-data input or reusable numerical transform.
3. Confirm it preserves the return definition and cannot use future information.

## Preserve State Semantics

- Treat `EvaluationContext.factor` as immutable.
- Read the current `state.factor` in direct consumers and transforms.
- Make transforms preserve the exact index and columns, then call
  `state.replace_factor(transformed)`.
- Do not restore the original factor implicitly.
- Remember that `requires` only proves some earlier occurrence ran. It does not prove the producer
  ran after the latest state transform.
- Rerun producers such as `rank_ic` or `quantile_returns` after a transform before their summaries.
- Use `require_detail()` and `require_cache()` for the newest stored prerequisite.
- Never create `__2` suffixes manually; `EvaluationState` versions repeated names.

When removing several correlated exposures, prefer one joint cross-sectional regression over
sequential residualizations unless order dependence is intended and tested.

## Implement

1. Put reusable numerical transformations under `src/transforms/`.
2. Put the pipeline wrapper under `src/evaluators/`.
3. Register one function taking `EvaluationState`:

```python
@evaluation_method(
    "method_name",
    requires=("prerequisite",),
    required_data_symbols=("symbol",),
)
def evaluate_method(state: EvaluationState) -> None:
    ...
```

4. Declare only true execution prerequisites and every required market-data symbol.
5. If adding a data symbol, update engine file mappings, validation, fixtures, data contracts, and
   documentation.
6. Publish outputs through:
   - `state.add_metrics()` for JSON-safe scalars;
   - `state.add_detail()` for numeric date-indexed Series/DataFrames;
   - `state.add_artifact()` for files inside the run directory;
   - `state.add_cache()` for non-persisted intermediates.
7. Export the evaluator from `src/evaluators/__init__.py`.
8. Import it in `src/evaluators/registry.py` so decorator registration occurs.
9. Add it to `DEFAULT_EVALUATION_METHODS` only when universally required.
10. Update `src/funnel.py` and gate tests only when the method belongs to a funnel stage.

Keep persisted paths relative to `output_dir`, detail CSVs compatible with `utf-8-sig`, and detail
cells numeric apart from the date index. Preserve the `Long-Short` column in group and cumulative
returns.

## Keep the Webapp Compatible

For each new method, metric, or detail:

1. Add every registered evaluation method to
   `webapp/frontend/src/lib/methodDefinitions.ts` with:
   - its Chinese name and category;
   - the mathematical formula actually implemented, written as valid LaTeX;
   - a precise method definition, including sample requirements and timing;
   - result interpretation and known limitations.
2. Render formulas only through
   `webapp/frontend/src/components/LatexFormula.tsx` and KaTeX. Do not substitute Unicode,
   monospace text, screenshots, or hand-built HTML for mathematical rendering.
3. Confirm the `/methods` evaluation-module library:
   - exposes the new method through search/filter and the method list;
   - renders the selected method with a `.katex-display` formula;
   - contains KaTeX MathML accessibility output;
   - has no KaTeX parse errors or “定义缺失” warning;
   - shows dependencies and required data from `/api/methods`.
4. Keep `/pipelines` focused on pipeline-template composition. It may link to `/methods`, but must
   not reintroduce the full mathematical definition catalog on the template page.
5. Add each new metric/detail key's Chinese label, format, direction, description, or title to
   `webapp/frontend/src/lib/metrics.ts`.
6. When changing an existing evaluator's semantics, search the frontend for stale method names,
   metric keys, tooltip fallbacks, detail titles, table labels, and `/methods` descriptions. Do not
   leave historical internal field names visible to users unless the UI explicitly labels them as
   implementation compatibility details.
7. Preserve `__<n>` repeated-output parsing.
8. Update server imports and API consumers in the same change if a public API changes.
9. Keep stored pipelines ordered and duplicate-preserving.
10. Rebuild the frontend after changing its source:

```bash
cd webapp/frontend && npm run build
```

## Test

Add focused coverage for:

- numerical behavior, NaNs, insufficient observations, and exact axes;
- registration, dependencies, and required data symbols;
- execution before and after a state transform;
- repeated keys and newest-detail/cache resolution;
- engine persistence and relative output paths;
- complete webpage method definitions matching the implemented formulas;
- valid KaTeX rendering and MathML output for every method formula;
- server and frontend contracts.

Run the full verification checklist from `AGENTS.md`. If output shapes change, complete a real
webapp evaluation and confirm metrics and charts render. Open the evaluation-module page and
verify the new method definition visually. A registered method without its webpage definition is
an incomplete change. Do not report completion when required checks were skipped or failed.
