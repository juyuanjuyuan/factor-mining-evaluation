---
name: mine-stock-factors
description: "Use the local LLM and this repository's Webapp-parity CLI to design, generate, validate, train-period screen, out-of-sample test, compare, and archive Chinese A-share daily cross-sectional factor expressions. Trigger for agent factor mining, 因子挖掘, training/testing period, 批量候选生成, 经济假设转表达式, 因子去重, evaluation jobs, custom ordered pipelines, funnel tests, metric gates, or saving candidates to the test factor library. Apply the human researcher's prompt-defined evaluation protocol; do not invent acceptance thresholds or submit factors to the formal library without explicit authorization."
---

# Mine Stock Factors

Use the local model for hypothesis generation and use the repository CLI for every deterministic
operation. Keep factor design flexible; keep validation, execution, persistence, and timing
contracts strict.

Read `AGENTS.md` first. Read
[the research contract](references/research-contract.md) before the first mining round in a project
or whenever data, timing, naming, outcome classification, or result evidence is unclear. Read
[the CLI reference](references/cli-reference.md) before using templates, gates, batch/tag sources,
job control, or formal-library operations.

## Treat the Prompt as the Research Protocol

Extract the following from the human researcher's prompt when present:

- economic theme and hypotheses to explore;
- allowed inputs, operators, lookback ranges, and candidate count;
- train/test or sample boundary, if the requested workflow defines one;
- evaluation kind, exact ordered methods or template, horizon, quantiles, and significance level;
- metric gates, comparison rules, diversity requirements, and stopping condition;
- whether to save candidates to the test library or promote anything further.

Apply those choices exactly. Treat CLI defaults as execution defaults, never as evidence that a
factor is acceptable. If the prompt does not define an acceptance rule, run the requested tests and
report evidence without labeling factors good, bad, passed, or rejected.

Never smuggle fixed IC, IR, return, drawdown, turnover, or correlation thresholds into the mining
process. Do not copy WorldQuant/BRAIN thresholds, fields, neutralization settings, or submission
rules into this A-share project.

## Keep a Durable Research Journal

Treat a stable `--research-id` as the identity of one economic research thread, for example
`short-term-reversal-v1`. The CLI appends its evidence to
`<state-dir>/factor_research_journal.jsonl`; this is not the formal factor registry and does not
change a candidate's promotion status.

Before proposing a new round in an existing thread, read its compact history. Do this even if the
current prompt repeats an earlier direction, so the agent can avoid parameter-only reruns and can
state what changed:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_research.py journal summary \
  --research-id short-term-reversal-v1
```

At the beginning, when changing direction, after an important interpretation, and when stopping a
thread, add a concise researcher/agent note. Include the mechanism, current hypothesis, and why
the next decision follows from prior evidence:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_research.py journal note \
  --research-id short-term-reversal-v1 \
  --research-phase exploration \
  --research-direction '短期价格反转' \
  --hypothesis '近期收益冲击会在下一持有期部分反转' \
  --note '先用训练期检验二日变化；尚未定义通过门槛。'
```

Every evaluation belonging to that thread must pass `--research-id`, `--research-direction`, and
`--hypothesis`; use `--research-phase exploration|training|testing`, `--iteration-note`, and
repeatable `--parent-candidate` to preserve its context and lineage. The CLI records submission,
terminal success/failure/cancellation, requested and actual dates, metrics, and gate evidence
automatically. After each completed training or testing batch, run `journal summary` again and
base the next LLM reasoning on the returned history. The summary is evidence, not an automatic
selection rule: the human's prompt remains the only authority for interpretation and pass/fail.

## Run the Research Loop

1. Choose or recover a stable research ID. Inspect the prompt, `factor_registry/`, relevant prior
   outputs, and `journal summary --research-id ID`. Identify already-tested expressions, outcomes,
   candidate lineage, notes, and close variants before proposing more. For a brand-new thread,
   first create a `journal note` stating its initial direction and hypothesis.
2. Inspect live capabilities instead of relying on a stale operator or method list:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_research.py catalog all
```

3. State each candidate's economic hypothesis, information timing, expected direction, and the
   structural difference from existing candidates. Generate stable lowercase names.
4. Use only day-`t`-or-earlier information. Preserve the next-open label and the distinction between
   `vol`, `amt`, proxy `vwap`, and evaluator-only `industry`.
5. Validate every expression before spending time on evaluation:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_research.py validate \
  --expression 'rank_cs(delta(c, 20) / (ts_std(c, 20) + 1e-6))'
```

6. Run candidates inline unless persistence is requested. Express the researcher-defined protocol
   with `--methods`, `--template`, `--kind funnel`, `--gate`, or an exact `--request-file`:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_research.py run \
  --factor-name momentum_vol_scaled_001 \
  --expression 'rank_cs(delta(c, 20) / (ts_std(c, 20) + 1e-6))' \
  --methods rank_ic,rank_icir,quantile_returns,quantile_cumulative,quantile_plot \
  --horizon 1 --quantiles 10 \
  --signal-start 2019-10-10 --signal-end 2021-12-31 \
  --research-id momentum-volatility-v1 \
  --research-phase training \
  --research-direction '风险调整后的中期动量' \
  --hypothesis '中期上涨且波动较低的股票在下一持有期相对更强' \
  --iteration-note '首个波动缩放版本'
```

7. Parse the returned job JSON. Require terminal `job.status == "succeeded"` and each expected run
   to be `succeeded` or intentionally `skipped` by the funnel. Inspect `run.result`, `gate_outcome`,
   `gate_explanation`, `output_dir`, metrics, details, plots, and generated code. Missing artifacts
   make the evaluation invalid.
8. Classify every attempt as `invalid_expression`, `execution_failed`, or `research_tested`. Use only
   `research_tested` outcomes to support or reject an economic hypothesis.
9. Run `journal summary --research-id ID` after the batch. Revisit its completed/failed evidence,
   sample dates, gate records, counterexamples, and parent-child relationships before writing the
   next iteration. Record a `journal note` for a direction change, selection decision, or stopping
   rationale.
10. Compare candidates only on compatible return definitions, horizons, samples, data bases, and
    method order. Preserve failures and counterexamples; do not report only winners.
11. Stop according to the prompt. If no stop rule is supplied, complete the requested candidate set
    and return the evidence rather than silently launching an open-ended search.

## Enforce Training Then Testing

When the prompt defines training and testing periods, treat them as a strict two-stage protocol:

1. Run all hypothesis generation, expression variants, parameter choices, comparisons, gates, and
   eliminations only on the training period using paired `--signal-start` and `--signal-end` flags
   plus `--research-phase training`. Review and note the training summary before selecting a winner.
2. Move only candidates that satisfy the researcher's training rule to testing. Before the first
   testing run, freeze the factor name, exact expression, operator parameters, horizon, quantiles,
   ordered method pipeline, transformations, gates, and expected direction. Record the frozen
   candidate in `journal note`; use `--parent-candidate` where a testing name derives from the
   training candidate.
3. Run the frozen candidate on the non-overlapping testing period. Do not generate or select a
   variant using testing-period results. Pass `--research-phase testing` and then review the same
   research ID's journal summary.
4. Label a candidate truly passed only when both the training result and the testing result satisfy
   the prompt-defined rules. A successful process exit only means the computation completed.
5. If testing fails, report the failure. Any later tuning informed by that result contaminates this
   holdout; use a genuinely fresh testing period for another unbiased test or explicitly label the
   later result as reused-holdout evidence.

The date flags name requested signal-date boundaries. The engine evaluates rolling expressions on
full earlier history, then slices the sample and removes the final `horizon + 1` signal rows so the
next-open entry and exit stay inside that period. Verify both `run.run_params.signal_start/end` and
`run.result.sample_start_day/end_day` in the returned JSON.

## Keep Test and Formal Libraries Separate

Use `factor add` only when the researcher asks for persistent candidates or when the requested
workflow explicitly uses the Webapp test library. Record project, tags, paper expression, and proxy
status honestly.

Do not call `factor submit`, `factor delete`, template deletion, or job deletion unless the prompt
explicitly authorizes that state change. Formal-library promotion is not an automatic consequence
of a metric gate or funnel pass.

## Report a Mining Round

Return a compact candidate table containing:

- factor name, exact expression, hypothesis, expected sign, and required symbols;
- research ID, phase, iteration note, parent candidate(s), and the prior-history finding that
  motivated this iteration;
- requested evaluation configuration and observed execution status;
- the metrics and gate evidence named by the researcher;
- artifact/run paths and any proxy-data warning;
- outcome classification and the next decision under the prompt-defined rule.

Separate implementation errors from research evidence. State when the researcher supplied no
acceptance rule. Never infer economic validity from absolute IR alone, a single endpoint, or a
successful process exit.

## Use the Existing Evaluation Skill for Evaluator Changes

If the task changes an evaluator, dependency, output, metric, state transform, or Webapp method
definition, stop the mining loop and follow
[`evaluate-stock-factors`](../evaluate-stock-factors/SKILL.md). Do not patch evaluator behavior just
to make a candidate pass the requested protocol.

## Completion Check

- Validate every expression with the AST whitelist.
- Preserve `open[t+1+horizon] / open[t+1] - 1` and prohibit future information.
- Record the exact ordered pipeline, including repeats.
- Keep training and testing periods non-overlapping; freeze every candidate before testing.
- Verify requested and actual sample dates in the returned run JSON.
- Read the research journal before each new round and after every completed training/testing batch.
- Journal every research run and record direction/selection/stop decisions as notes.
- Use only prompt-supplied gates and selection rules.
- Verify terminal runs and nonempty persisted artifacts.
- Keep incompatible data bases and samples out of direct rankings.
- Leave formal submission and destructive operations to explicit human authorization.
