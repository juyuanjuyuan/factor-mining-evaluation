# Agent Factor-Mining Research Contract

## Contents

- Authority and scope
- Information timing and data
- Candidate construction
- Evaluation evidence
- Outcome classification
- Research journal and iteration
- Persistence and reporting

## Authority and Scope

Use the human researcher's prompt as the source of evaluation objectives, thresholds, candidate
budget, and stopping rules. Supply tools and evidence; do not replace missing research choices with
memorized industry heuristics.

The skill supports LLM-designed daily cross-sectional A-share factor expressions. It does not turn
the project into a tick strategy, execution simulator, WorldQuant BRAIN client, or autonomous
formal-factor admission system.

## Information Timing and Data

Compute a factor after day `t` close and enter at the next open. Preserve the canonical label:

```text
open[t + 1 + horizon] / open[t + 1] - 1
```

The default `horizon=1` is `open[t+2] / open[t+1] - 1`. Reject expressions that use future rows,
negative lags, future returns, or post-`t` information.

Use the live `catalog operators` output and `docs/FACTOR_EVALUATION_CONTRACT.md` for the current
namespace. Core distinctions:

- `c`, `o`, `h`, `l`: adjusted price matrices;
- `vol`: exchange-reported volume;
- `amt`: traded amount; never substitute it for `vol`;
- `vwap`: `(high+low)/2` proxy in this dataset, not exchange VWAP;
- `cap`: total market capitalization;
- `limit`, `st`: tradability inputs;
- `industry`: evaluator-only point-in-time classification; never place it in an expression.

Missing observations remain missing. Do not forward-fill matrices to increase apparent coverage.

## Candidate Construction

For each candidate, record:

1. economic mechanism and why information at `t` could relate to the requested return horizon;
2. exact expression and required symbols;
3. expected IC/portfolio direction, if the hypothesis implies one;
4. lookback windows and the reason they differ from prior attempts;
5. data-basis caveats, especially proxy inputs;
6. nearest existing or previously tested expression and the structural difference.

Use the model to generate hypotheses and expression variants. Use the CLI, not model arithmetic, to
validate or evaluate them. Add a missing operator only when it is generally reusable and the human
request authorizes an evaluator/engine change.

Avoid parameter-only candidate floods. Vary the economic mechanism, input family, transformation,
or interaction when the prompt asks for diverse hypotheses. Do not claim statistical independence
without a prompt-defined and executed comparison.

## Evaluation Evidence

The CLI's default method list is a runtime default, not a universal research standard. Preserve the
researcher's exact:

- task kind: custom evaluation or four-stage funnel;
- ordered method sequence, including repeated pre/post-transform methods;
- holding horizon and quantile count;
- significance level;
- metric gates and `all`/`any` semantics;
- factor source: inline, explicit saved factors, or tag-selected batch.
- requested training/testing signal-date boundaries.

For a training/testing protocol, candidate creation, selection, parameter tuning, and repeated
experimentation must stop at the training boundary. Freeze the candidate and the full evaluation
configuration before testing. The testing period must not overlap training. Only the human
researcher's prompt defines the pass conditions, but a candidate may be called finally passed only
if it passed those conditions on both periods.

If testing evidence influences a later expression or parameter choice, the same testing period is
no longer an untouched holdout for that later candidate. Use a fresh holdout or describe the result
as reused-holdout evidence rather than sample-out confirmation.

Check the returned job and run objects. A valid completed run must have:

- terminal success rather than a queued/running/failed status;
- the requested expression, horizon, quantiles, and method order;
- requested `run_params.signal_start/end` and actual `result.sample_start_day/end_day`;
- nonempty persisted metrics and the requested details/artifacts;
- relative detail/artifact paths resolving inside the run output directory;
- compatible return/data/sample definitions before cross-factor comparison.

A gate labels results but does not change the computation. A funnel may intentionally mark later
stages `skipped` after a failed canonical stage gate.

## Outcome Classification

Use three evidence states:

| State | Meaning | Research evidence? |
|---|---|---|
| `invalid_expression` | AST rejection, unknown operator/symbol, invalid arguments, or other expression implementation error | No |
| `execution_failed` | Missing data, worker crash, timeout, persistence failure, or missing required artifact | No |
| `research_tested` | Requested evaluation completed and produced its metrics/artifacts, regardless of pass/fail | Yes |

Do not count the first two states as evidence against an economic hypothesis. Correct the
implementation or environment, then rerun under the same factor name when it is a correction.

## Research Journal and Iteration

Use one stable research ID for a coherent economic theme. Before every later round, execute
`journal summary --research-id ID` and use the returned candidate lineage, prior hypotheses,
notes, requested/actual sample dates, terminal status, metrics, and gate evidence to decide what
to try next. Do not restart a direction merely because its previous output is inconvenient.

At the start of a direction, after interpreting material evidence, before testing a frozen training
winner, when changing course, and when ending a thread, append a `journal note`. State the
direction, hypothesis, phase, and a concise decision or observation. Notes document reasoning;
they do not supply a hidden pass/fail rule.

Every research evaluation must include the following CLI metadata:

- `--research-id`: stable thread identity;
- `--research-phase exploration|training|testing`;
- `--research-direction` and `--hypothesis`;
- `--iteration-note` when the candidate changes; and
- repeatable `--parent-candidate` for a deliberate extension, correction, or frozen test variant.

The CLI records submission and known terminal completion automatically in the append-only
`<state-dir>/factor_research_journal.jsonl`. It stores experimental evidence and lineage only; it
is separate from the test and formal factor registries. It never promotes a factor, determines
whether a gate should exist, or converts a metric into an economic conclusion. An asynchronous
`--submit-only` job is closed in the journal when the CLI later observes it through `job wait` or
`job cancel`.

Use a new research ID only for a genuinely new mechanism. A new expression within an existing
mechanism normally remains in the same thread, linked to its parent candidate when applicable.
Preserve failed and cancelled attempts so a subsequent agent can distinguish untested ideas from
negative or inconclusive evidence.

## Persistence and Reporting

Use inline evaluation for disposable candidates. Use the Webapp test library when persistence,
tags, projects, or human follow-up are requested. Preserve one stable name for corrections and use
a new name for a distinct expression.

Formal submission may trigger correlation computation and registry writes. Perform it only after
explicit authorization. Never delete registry entries, jobs, templates, or artifacts as routine
cleanup.

Report all requested candidates, including research-tested failures. Include exact expressions,
configuration, evidence, artifacts, and incompatible-comparison warnings. If the human provided no
acceptance criteria, explicitly say that the round produced measurements but no pass/fail decision.
