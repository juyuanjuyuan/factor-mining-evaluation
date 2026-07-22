"""Append-only local evidence journal for agent-led factor research."""

from __future__ import annotations

from datetime import datetime
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence


JOURNAL_FILENAME = "factor_research_journal.jsonl"
SCHEMA_VERSION = 1
_METADATA_METRIC_KEYS = {
    "run_id",
    "evaluated_at",
    "factor_name",
    "artifact_name",
    "expression",
    "evaluation_details",
    "evaluation_artifacts",
}


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe(value: Any) -> Any:
    """Convert evaluator/API values into strict JSON without losing finite metrics."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe(item) for item in value]
    return str(value)


def _metric_snapshot(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    metrics: dict[str, Any] = {}
    for key, value in result.items():
        if key in _METADATA_METRIC_KEYS:
            continue
        safe_value = _safe(value)
        if isinstance(safe_value, (str, int, float, bool)) or safe_value is None:
            metrics[str(key)] = safe_value
    return metrics


def _run_snapshot(run: Mapping[str, Any]) -> dict[str, Any]:
    result = run.get("result")
    result_mapping = result if isinstance(result, Mapping) else None
    run_params = run.get("run_params")
    params = run_params if isinstance(run_params, Mapping) else {}
    return {
        "run_id": str(run.get("id", "")),
        "factor_name": str(run.get("factor_name", "")),
        "batch_id": run.get("batch_id"),
        "expression": str(run.get("expression", "")),
        "status": str(run.get("status", "")),
        "stage": run.get("stage"),
        "methods": _safe(run.get("methods", [])),
        "horizon": _safe(run.get("horizon")),
        "n_quantiles": _safe(run.get("n_quantiles")),
        "gate_outcome": run.get("gate_outcome"),
        "gate_explanation": run.get("gate_explanation"),
        "output_dir": run.get("output_dir"),
        "requested_signal_start": params.get("signal_start"),
        "requested_signal_end": params.get("signal_end"),
        "sample_start_day": (
            result_mapping.get("sample_start_day") if result_mapping else None
        ),
        "sample_end_day": (
            result_mapping.get("sample_end_day") if result_mapping else None
        ),
        "return_definition": (
            result_mapping.get("return_definition") if result_mapping else None
        ),
        "metrics": _metric_snapshot(result_mapping),
    }


def _job_snapshot(job: Mapping[str, Any]) -> dict[str, Any]:
    runs = job.get("runs")
    run_list = (
        runs
        if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes, bytearray))
        else []
    )
    return {
        "job_id": str(job.get("id", "")),
        "kind": str(job.get("kind", "")),
        "title": job.get("title"),
        "status": str(job.get("status", "")),
        "params": _safe(job.get("params", {})),
        "runs": [
            _run_snapshot(run)
            for run in run_list
            if isinstance(run, Mapping)
        ],
    }


class ResearchJournal:
    """Keep append-only research evidence next to the CLI/Webapp state database."""

    def __init__(self, state_dir: str | Path):
        self.path = Path(state_dir).expanduser().resolve() / JOURNAL_FILENAME

    def append(
        self,
        event: str,
        *,
        research: Mapping[str, Any],
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "schema_version": SCHEMA_VERSION,
            "event": event,
            "recorded_at": _timestamp(),
            "research": _safe(research),
            **_safe(payload or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
        return record

    def records(
        self,
        *,
        research_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"research journal contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"research journal record at line {line_number} must be an object"
                )
            research = record.get("research")
            if research_id is not None and (
                not isinstance(research, Mapping)
                or research.get("research_id") != research_id
            ):
                continue
            records.append(record)
        return records[-limit:] if limit is not None else records

    def record_submission(
        self,
        research: Mapping[str, Any],
        job: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.append(
            "evaluation_submitted",
            research=research,
            payload={"job": _job_snapshot(job)},
        )

    def record_submission_failure(
        self,
        research: Mapping[str, Any],
        error: str,
    ) -> dict[str, Any]:
        return self.append(
            "evaluation_submission_failed",
            research=research,
            payload={"error": error},
        )

    def record_completion_if_known(
        self,
        job: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Append a terminal update only for a job previously journaled by this CLI."""

        job_id = str(job.get("id", ""))
        if not job_id:
            return None
        records = self.records()
        if any(
            record.get("event") == "evaluation_finished"
            and isinstance(record.get("job"), Mapping)
            and record["job"].get("job_id") == job_id
            for record in records
        ):
            return None
        submission = next(
            (
                record
                for record in reversed(records)
                if record.get("event") == "evaluation_submitted"
                and isinstance(record.get("job"), Mapping)
                and record["job"].get("job_id") == job_id
            ),
            None,
        )
        if submission is None or not isinstance(submission.get("research"), Mapping):
            return None
        return self.append(
            "evaluation_finished",
            research=submission["research"],
            payload={
                "job": _job_snapshot(job),
                "submission_recorded_at": submission.get("recorded_at"),
            },
        )

    def add_note(
        self,
        research: Mapping[str, Any],
        note: str,
    ) -> dict[str, Any]:
        return self.append(
            "research_note",
            research=research,
            payload={"note": note},
        )

    def summary(
        self,
        *,
        research_id: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        records = self.records(research_id=research_id, limit=limit)
        directions: dict[tuple[str, str], dict[str, Any]] = {}
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        notes: list[dict[str, Any]] = []

        for record in records:
            research = record.get("research")
            if not isinstance(research, Mapping):
                continue
            research_thread = str(research.get("research_id") or "未说明研究")
            direction = str(research.get("research_direction") or "未说明方向")
            direction_summary = directions.setdefault(
                (research_thread, direction),
                {
                    "research_id": research_thread,
                    "research_direction": direction,
                    "event_count": 0,
                    "hypotheses": [],
                    "phases": [],
                    "latest_recorded_at": None,
                },
            )
            direction_summary["event_count"] += 1
            hypothesis = research.get("hypothesis")
            if isinstance(hypothesis, str) and hypothesis and hypothesis not in direction_summary["hypotheses"]:
                direction_summary["hypotheses"].append(hypothesis)
            phase = research.get("research_phase")
            if isinstance(phase, str) and phase not in direction_summary["phases"]:
                direction_summary["phases"].append(phase)
            direction_summary["latest_recorded_at"] = record.get("recorded_at")

            if record.get("event") == "research_note":
                notes.append(
                    {
                        "recorded_at": record.get("recorded_at"),
                        "research_id": research_thread,
                        "research_direction": direction,
                        "research_phase": phase,
                        "note": record.get("note"),
                    }
                )

            job = record.get("job")
            if not isinstance(job, Mapping):
                continue
            job_runs = job.get("runs")
            if not isinstance(job_runs, Sequence) or isinstance(
                job_runs, (str, bytes, bytearray)
            ):
                continue
            for run in job_runs:
                if not isinstance(run, Mapping):
                    continue
                name = str(run.get("factor_name") or "unnamed_candidate")
                candidate = candidates.setdefault(
                    (research_thread, name),
                    {
                        "research_id": research_thread,
                        "factor_name": name,
                        "expression": run.get("expression"),
                        "parent_candidates": [],
                        "evaluations": [],
                    },
                )
                parents = research.get("parent_candidates")
                if isinstance(parents, Sequence) and not isinstance(parents, str):
                    for parent in parents:
                        parent_name = str(parent)
                        if parent_name and parent_name not in candidate["parent_candidates"]:
                            candidate["parent_candidates"].append(parent_name)
                evaluation = {
                    "recorded_at": record.get("recorded_at"),
                    "event": record.get("event"),
                    "research_id": research_thread,
                    "research_direction": direction,
                    "research_phase": phase,
                    "job_id": job.get("job_id"),
                    "run_id": run.get("run_id"),
                    "status": run.get("status"),
                    "gate_outcome": run.get("gate_outcome"),
                    "requested_signal_start": run.get("requested_signal_start"),
                    "requested_signal_end": run.get("requested_signal_end"),
                    "sample_start_day": run.get("sample_start_day"),
                    "sample_end_day": run.get("sample_end_day"),
                    "metrics": run.get("metrics", {}),
                }
                existing_index = next(
                    (
                        index
                        for index, item in enumerate(candidate["evaluations"])
                        if item.get("run_id") == evaluation["run_id"]
                    ),
                    None,
                )
                if existing_index is None:
                    candidate["evaluations"].append(evaluation)
                elif record.get("event") == "evaluation_finished":
                    candidate["evaluations"][existing_index] = evaluation

        return {
            "journal_path": str(self.path),
            "research_id": research_id,
            "records_considered": len(records),
            "directions": list(directions.values()),
            "candidates": list(candidates.values()),
            "recent_notes": notes[-20:],
        }
