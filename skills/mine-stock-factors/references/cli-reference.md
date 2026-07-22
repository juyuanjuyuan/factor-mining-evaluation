# Factor Research CLI Reference

## Contents

- Runtime model
- Webapp-to-CLI control map
- Capability and expression commands
- Test-factor library commands
- Evaluation commands
- Gates and exact request JSON
- Template commands
- Job and comparison commands
- Research journal commands
- Output and exit behavior

## Runtime Model

Use the existing interpreter:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python scripts/run_factor_research.py ...
```

The CLI opens the local FastAPI application in-process. It uses the same request schemas,
expression validator, registry service, templates, SQLite job database, worker, evaluator pipeline,
gates, and funnel as the Webapp. Completed runs are stored under `outputs/webapp/runs/` and appear
in the Webapp.

Place runtime overrides before the subcommand:

```bash
python scripts/run_factor_research.py \
  --data-dir /path/to/data \
  --state-dir /path/to/state \
  --registry-dir /path/to/factor_registry \
  catalog methods
```

Use `--submit-only` when a long-lived Webapp worker is already running and should own the queue.
Poll with `job show`; do not start another waiting worker merely to inspect progress.

## Webapp-to-CLI Control Map

| Webapp operation/control | CLI equivalent |
|---|---|
| Real-time expression validation | `validate --expression ...` |
| Method/operator/funnel library | `catalog methods`, `catalog operators`, `catalog funnel-stages` |
| Save a new test factor | `factor add` |
| Factor name/project/paper expression/tags/proxy fields | matching `factor add` or `factor update` flags |
| Select saved factors | repeat `run --factor BATCH/NAME` |
| Select a tag research set | repeat `run --tag TAG`, plus `--tag-match` and `--library` |
| Custom evaluation vs four-stage funnel | `run --kind evaluate|funnel` |
| Flow template | `run --template ID_OR_EXACT_NAME` |
| Holding horizon | `run --horizon H` |
| Quantile groups | `run --quantiles N` |
| IC significance level | `run --significance-level P` |
| Evaluation signal-date period | `run --signal-start YYYY-MM-DD --signal-end YYYY-MM-DD` |
| Ordered/repeated method pipeline | `run --methods m1,m2,m1,...` |
| Optional result gate | repeat `run --gate METRIC:OP:VALUE` and set `--gate-match` |
| Submit task | `run`; add `--submit-only` to leave it queued |
| Queue/status/cancel/delete | `job list|show|wait|cancel|delete` |
| Compare selected completed runs | `compare --run-id ...` |
| Review research direction and iterations | `journal summary --research-id ID` |
| Record a research decision | `journal note --research-id ID ... --note ...` |
| Create/edit/clone/delete templates | `template create|update|clone|delete` |

Formal factor promotion and destructive commands exist for parity, but require explicit human
authorization.

## Capability and Expression Commands

Show everything an agent may need before generating candidates:

```bash
python scripts/run_factor_research.py catalog all
```

Use narrower outputs to save context:

```bash
python scripts/run_factor_research.py catalog methods
python scripts/run_factor_research.py catalog operators
python scripts/run_factor_research.py catalog funnel-stages
python scripts/run_factor_research.py catalog templates
python scripts/run_factor_research.py catalog market-data
```

Validate an expression and return required symbols/operators:

```bash
python scripts/run_factor_research.py validate \
  --expression 'rank_cs(delta(c, 20) / (ts_std(c, 20) + 1e-6))'
```

## Test-Factor Library Commands

Create a persistent test candidate:

```bash
python scripts/run_factor_research.py factor add \
  --factor-name momentum_vol_scaled_001 \
  --expression 'rank_cs(delta(c, 20) / (ts_std(c, 20) + 1e-6))' \
  --project '动量研究' \
  --paper-expression 'optional source notation' \
  --tag candidate --tag medium-horizon
```

If an expression uses the local VWAP proxy, add both:

```text
--uses-proxy --proxy-description 'vwap uses (high+low)/2 proxy'
```

List/show/update metadata:

```bash
python scripts/run_factor_research.py factor list --library test
python scripts/run_factor_research.py factor show --factor webapp_test_factors/momentum_vol_scaled_001
python scripts/run_factor_research.py factor tags \
  --factor webapp_test_factors/momentum_vol_scaled_001 --tag candidate --tag reviewed
python scripts/run_factor_research.py factor project \
  --factor webapp_test_factors/momentum_vol_scaled_001 --project '动量研究'
```

Use `factor update` for editable test-library definitions. Omitted fields keep their current value;
use `--clear-tags` to clear tags and `--no-uses-proxy` to clear the proxy flag.

Only with explicit authorization:

```bash
python scripts/run_factor_research.py factor submit \
  --factor webapp_test_factors/momentum_vol_scaled_001
python scripts/run_factor_research.py factor delete \
  --factor webapp_test_factors/momentum_vol_scaled_001
```

`factor submit` writes the formal registry and may compute correlation state. `factor delete` is
destructive for the selected test/formal registry record.

## Evaluation Commands

Run an inline candidate with the default evaluation pipeline:

```bash
python scripts/run_factor_research.py run \
  --factor-name momentum_vol_scaled_001 \
  --expression 'rank_cs(delta(c, 20) / (ts_std(c, 20) + 1e-6))'
```

Run an exact pipeline. Order and repeats are preserved:

```bash
python scripts/run_factor_research.py run \
  --factor-name neutralization_comparison_001 \
  --expression 'rank_cs(delta(c, 20))' \
  --methods rank_ic,rank_icir,market_cap_neutralize,rank_ic,rank_icir \
  --horizon 1 --quantiles 10
```

Run the same candidate on a bounded training period:

```bash
/Users/huangjuyuan/miniforge3/envs/rdagent/bin/python \
  scripts/run_factor_research.py run \
  --factor-name candidate_001 \
  --expression 'rank_cs(delta(c, 20))' \
  --methods rank_ic,rank_icir \
  --signal-start 2019-10-10 \
  --signal-end 2021-12-31
```

Both date flags are required together. They are inclusive requested signal-date boundaries. The
factor expression retains all earlier history needed for rolling lookbacks, while the final
`horizon + 1` signal rows are removed so the entry and exit opens remain inside the period. Inspect
`run.run_params.signal_start/end` for the request and `run.result.sample_start_day/end_day` for the
actual evaluated signal range.

For an out-of-sample test, rerun only the frozen training winner with a non-overlapping date pair.
Do not change its expression, parameters, horizon, quantiles, ordered methods, transforms, gates,
or expected direction between the two commands.

For an agent research run, add durable research metadata. `--research-id` requires both
`--research-direction` and `--hypothesis`; the CLI then journals the submitted job and known
terminal outcome. Use `--research-phase training` for the training run and `testing` for the frozen
out-of-sample run:

```bash
python scripts/run_factor_research.py run \
  --factor-name candidate_001 \
  --expression 'rank_cs(delta(c, 20))' \
  --methods rank_ic,rank_icir \
  --signal-start 2019-10-10 --signal-end 2021-12-31 \
  --research-id short-term-reversal-v1 \
  --research-phase training \
  --research-direction '短期价格反转' \
  --hypothesis '近期收益冲击会在下一持有期部分反转' \
  --iteration-note '训练期首个候选'
```

Run saved factors or a tag-selected research set:

```bash
python scripts/run_factor_research.py run \
  --factor alpha101_runnable_factors/alpha001 \
  --factor webapp_test_factors/momentum_vol_scaled_001 \
  --methods rank_ic,rank_icir

python scripts/run_factor_research.py run \
  --tag candidate --tag medium-horizon --tag-match all --library test \
  --template '默认完整评价'
```

Run the standard four-stage funnel:

```bash
python scripts/run_factor_research.py run \
  --kind funnel \
  --factor webapp_test_factors/momentum_vol_scaled_001 \
  --horizon 1 --quantiles 10 --significance-level 0.05
```

The funnel owns fixed stage gates. Customize stage methods through a funnel template rather than a
custom metric gate.

## Gates and Exact Request JSON

Gate syntax:

```text
--gate METRIC:gte:VALUE
--gate METRIC:lte:VALUE
--gate METRIC:gt:VALUE
--gate METRIC:lt:VALUE
--gate METRIC:between:LOW:HIGH
```

Repeat `--gate` and combine conditions with `--gate-match all|any`. Gate metrics must be produced by
the selected pipeline. Repeated methods version outputs with `__2`, `__3`, and later suffixes.

Example using only researcher-supplied conditions:

```bash
python scripts/run_factor_research.py run \
  --factor-name candidate_001 --expression 'rank_cs(delta(c, 20))' \
  --methods rank_ic,rank_icir,quantile_returns,rolling_drawdown \
  --gate ir:gte:0.3 \
  --gate gn_rolling_drawdown_60_worst:gte:-0.4 \
  --gate-match all
```

Do not reuse those example numbers unless the current prompt supplies them.

For generated batches or exact reproducibility, pass the same JSON shape as Webapp `JobCreate`:

```json
{
  "kind": "evaluate",
  "title": "agent mining round 3",
  "factors": [
    {
      "factor_name": "candidate_001",
      "batch_id": "temporary",
      "expression": "rank_cs(delta(c, 20))"
    }
  ],
  "methods": ["rank_ic", "rank_icir"],
  "horizon": 1,
  "n_quantiles": 10,
  "significance_level": 0.05,
  "signal_start": "2019-10-10",
  "signal_end": "2021-12-31",
  "gate": {
    "conditions": [{"metric": "ir", "op": "between", "value": -1.0, "value2": 1.0}],
    "match": "all"
  }
}
```

Run it with:

```bash
python scripts/run_factor_research.py run --request-file /absolute/path/job.json
```

## Template Commands

Create an ordered methods template:

```bash
python scripts/run_factor_research.py template create \
  --name pre-post-neutralization --kind methods \
  --methods rank_ic,rank_icir,market_cap_neutralize,rank_ic,rank_icir
```

Create a funnel template. Omitted stages use the canonical methods; repeat `--stage` to override:

```bash
python scripts/run_factor_research.py template create \
  --name extended-funnel --kind funnel \
  --horizon 1 --quantiles 10 --significance-level 0.05 \
  --stage stage2_ic=rank_ic,newey_west_ic_significance,ic_trend_filter
```

Inspect, update, clone, or explicitly delete:

```bash
python scripts/run_factor_research.py template list
python scripts/run_factor_research.py template show --template extended-funnel
python scripts/run_factor_research.py template update --template extended-funnel --horizon 5
python scripts/run_factor_research.py template clone --template 1 --name copied-default
python scripts/run_factor_research.py template delete --template copied-default
```

Built-in templates are read-only. Template selection by name requires an exact name.

## Job and Comparison Commands

By default `run` starts a local worker, waits for a terminal state, and returns the complete job.
Use `--submit-only` to leave a queued job for a long-lived Webapp worker.

```bash
python scripts/run_factor_research.py job list --limit 50
python scripts/run_factor_research.py job show --job-id JOB_ID
python scripts/run_factor_research.py job wait --job-id JOB_ID --timeout 1800
python scripts/run_factor_research.py job cancel --job-id JOB_ID
python scripts/run_factor_research.py job delete --job-id JOB_ID
python scripts/run_factor_research.py compare --run-id RUN_A --run-id RUN_B
```

`job delete` removes terminal SQLite job/run records but keeps disk artifacts. It is still a
destructive platform action and requires explicit authorization.

## Research Journal Commands

The journal is an append-only JSONL evidence record at
`<state-dir>/factor_research_journal.jsonl`, separate from the factor registries. It lets a later
agent recover why a direction was tried, which candidates are related, what was run in training or
testing, and the observed metrics without inventing a selection rule.

Before a new round, request the compact LLM-readable summary:

```bash
python scripts/run_factor_research.py journal summary \
  --research-id short-term-reversal-v1
```

Record direction changes, selection decisions, conclusions, and stopping rationale as notes:

```bash
python scripts/run_factor_research.py journal note \
  --research-id short-term-reversal-v1 \
  --research-phase training \
  --research-direction '短期价格反转' \
  --hypothesis '近期收益冲击会在下一持有期部分反转' \
  --note '训练期候选已冻结，下一步只在独立 testing 期复核。'
```

Use `journal history --research-id ID --limit 50` for raw append-only records. `summary` defaults
to the most recent 200 records; increase `--limit` when a long-running thread needs more context.
The summary's metrics, status, gate output, and dates are evidence for the agent to interpret
under the current human prompt—not a CLI-generated recommendation. For a `--submit-only` research
run, later call `job wait` or `job cancel` through this CLI so the known terminal result is appended
to the same research thread.

## Output and Exit Behavior

Every successful command prints JSON to stdout. Expected command/API failures print JSON to stderr
and exit `2`. A waited job exits `2` on infrastructure `failed` or `cancelled`, but gate failure is
research output rather than a process failure and remains visible in `gate_outcome`.

For a completed evaluation, inspect:

- `job.status`, `total_runs`, and `finished_runs`;
- each run's `status`, `methods`, `gate_outcome`, and `gate_explanation`;
- `run.result` for persisted metrics and relative detail/artifact maps;
- `run.output_dir` for `metrics.csv`, details, plots, and generated code.

Do not interpret exit code `0` as an economic pass. Apply only the decision rule supplied in the
current research prompt.
