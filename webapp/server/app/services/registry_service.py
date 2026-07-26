"""Read factor batches and maintain web test/library batches atomically."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from engine import evaluate_factor_expression, expression_data_symbols
from evaluation_standards import PROFITABILITY_METHOD_NAMES, PROFITABILITY_STANDARD_NAME
from evaluators.base import REGISTERED_EVALUATION_METHODS
from factor_correlation import (
    FactorCorrelationService,
    FactorCorrelationThresholdError,
)
from factor_registry import load_factor_batch, validate_factor_batch
from returns import RETURN_DEFINITION

from ..config import Settings
from ..db import Database
from ..tagging import normalize_tags
from .funnel_service import make_run


ADMISSION_PROFITABILITY_TEMPLATE_NAME = "盈利能力测试"
ADMISSION_PROFITABILITY_METHOD_NAMES = tuple(PROFITABILITY_METHOD_NAMES)
ADMISSION_HORIZON = 1
ADMISSION_QUANTILES = 10
_PERFORMANCE_TIE_REL_TOLERANCE = 1e-9
_PERFORMANCE_TIE_ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class _PerformanceSnapshot:
    """One factor's comparable profitability result for library admission."""

    factor_name: str
    batch_id: str
    rolling_sharpe_60_median: float
    fitness: float | None
    source: str
    run_id: str | None
    evaluated_at: str | None
    comparison_signature: tuple[object, ...]

    def as_dict(self) -> dict[str, Any]:
        (
            horizon,
            n_quantiles,
            return_definition,
            signal_start,
            signal_end,
            sample_start_day,
            sample_end_day,
            methods,
        ) = self.comparison_signature
        return {
            "factor_name": self.factor_name,
            "batch_id": self.batch_id,
            "gn_rolling_sharpe_60_median": self.rolling_sharpe_60_median,
            "fitness": self.fitness,
            "source": self.source,
            "run_id": self.run_id,
            "evaluated_at": self.evaluated_at,
            "evaluation_config": {
                "horizon": horizon,
                "n_quantiles": n_quantiles,
                "return_definition": return_definition,
                "signal_start": signal_start,
                "signal_end": signal_end,
                "sample_start_day": sample_start_day,
                "sample_end_day": sample_end_day,
                "methods": list(methods),
            },
        }


class FactorLibraryPerformanceRejectedError(ValueError):
    """A correlated candidate is not strictly better than every conflict."""

    def __init__(self, decision: Mapping[str, Any]):
        self.decision = dict(decision)
        blockers = self.decision.get("blocking_factor_names") or []
        criterion = str(self.decision.get("criterion") or "盈利能力")
        names = "、".join(str(name) for name in blockers) or "已有因子"
        super().__init__(
            f"候选与因子库存在高相关性，且未在 {criterion} 上严格优于 {names}；"
            "因子库保持不变，候选保留在测试库"
        )


class FactorLibraryPerformanceEvaluationError(ValueError):
    """The service could not obtain a finite, comparable profitability result."""


class RegistryService:
    DEFAULT_PROJECT = "自定义因子"
    def __init__(
        self,
        settings: Settings,
        db: Database | None = None,
        correlation: FactorCorrelationService | None = None,
    ):
        self.settings = settings
        self.db = db or Database(settings.db_path)
        if db is None:
            self.db.initialize()
        self.correlation = correlation or FactorCorrelationService(settings)
        self._write_lock = threading.RLock()

    def batches(self, library: str = "all") -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.settings.registry_dir.glob("*.json")):
            batch = load_factor_batch(path)
            if library == "test" and batch["batch_id"] == self.settings.submitted_batch_id:
                continue
            if library == "factor" and batch["batch_id"] != self.settings.submitted_batch_id:
                continue
            result.append(batch)
        return result

    def factors(self, library: str = "test") -> list[dict[str, Any]]:
        submitted_names = self._submitted_names()
        rows: list[dict[str, Any]] = []
        records: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for batch in self.batches(library):
            for factor in batch["factors"]:
                records.append((batch, factor))
        tag_map = self.db.tags_for_factors(
            (str(batch["batch_id"]), str(factor["factor_name"]))
            for batch, factor in records
        )
        for batch, factor in records:
            identity = (str(batch["batch_id"]), str(factor["factor_name"]))
            rows.append(
                self._decorate_factor(
                    batch,
                    factor,
                    submitted_names,
                    tags=tag_map[identity],
                )
            )
        return rows

    def find(self, batch_id: str, factor_name: str) -> dict[str, Any] | None:
        path = self.settings.registry_dir / f"{batch_id}.json"
        if not path.is_file():
            return None
        batch = load_factor_batch(path)
        for factor in batch["factors"]:
            if factor["factor_name"] == factor_name:
                tags = self.db.tags_for_factors([(batch_id, factor_name)])[
                    (batch_id, factor_name)
                ]
                return self._decorate_factor(
                    batch,
                    factor,
                    self._submitted_names(),
                    tags=tags,
                )
        return None

    def factors_by_tags(
        self,
        tags: list[str],
        *,
        match: str = "any",
        library: str = "test",
    ) -> list[dict[str, Any]]:
        """Resolve a tagged research set without relying on table pagination."""
        normalized_tags = normalize_tags(tags)
        if not normalized_tags:
            return []
        if match not in {"any", "all"}:
            raise ValueError("标签匹配方式必须是 any 或 all")
        selected: list[dict[str, Any]] = []
        wanted = {tag.casefold() for tag in normalized_tags}
        for factor in self.factors(library):
            present = {tag.casefold() for tag in factor["tags"]}
            if (match == "all" and wanted <= present) or (
                match == "any" and bool(wanted & present)
            ):
                selected.append(factor)
        return selected

    def tag_summary(self, library: str = "test") -> list[dict[str, Any]]:
        """List available tags and their factor counts within one library."""
        counts: dict[str, int] = {}
        for factor in self.factors(library):
            for tag in factor["tags"]:
                counts[tag] = counts.get(tag, 0) + 1
        return [
            {"tag": tag, "count": count}
            for tag, count in sorted(counts.items(), key=lambda item: item[0].casefold())
        ]

    def set_tags(
        self,
        batch_id: str,
        factor_name: str,
        tags: list[str],
    ) -> dict[str, Any] | None:
        """Attach research metadata without altering a registry definition."""
        factor = self.find(batch_id, factor_name)
        if not factor:
            return None
        self.db.replace_factor_tags(
            batch_id,
            factor_name,
            normalize_tags(tags),
        )
        return self.find(batch_id, factor_name)

    def set_project(
        self,
        batch_id: str,
        factor_name: str,
        project: str,
    ) -> dict[str, Any] | None:
        """Reclassify one factor without changing its formula or evaluations."""
        normalized = project.strip()
        if not normalized:
            raise ValueError("项目不能为空")
        with self._write_lock:
            path = self.settings.registry_dir / f"{batch_id}.json"
            if not path.is_file():
                return None
            payload = load_factor_batch(path)
            index = next(
                (
                    index
                    for index, record in enumerate(payload["factors"])
                    if record["factor_name"] == factor_name
                ),
                None,
            )
            if index is None:
                return None
            record = deepcopy(payload["factors"][index])
            record["project"] = normalized
            record["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            payload["factors"][index] = record
            self._write_batch(payload, path)
            tags = self.db.tags_for_factors([(batch_id, factor_name)])[
                (batch_id, factor_name)
            ]
            return self._decorate_factor(
                payload,
                record,
                self._submitted_names(),
                tags=tags,
            )

    def _load_test(self) -> dict[str, Any]:
        path = self.settings.test_registry_path
        if path.is_file():
            return load_factor_batch(path)
        return self._empty_batch(
            batch_id=self.settings.test_batch_id,
            batch_name="Web 平台测试因子",
            source_notes="User-maintained test factor definitions",
        )

    def _load_submitted(self) -> dict[str, Any]:
        path = self.settings.submitted_registry_path
        if path.is_file():
            return load_factor_batch(path)
        return self._empty_batch(
            batch_id=self.settings.submitted_batch_id,
            batch_name="Web 平台因子库",
            source_notes="User-submitted production factor definitions",
        )

    @staticmethod
    def _empty_batch(
        *,
        batch_id: str,
        batch_name: str,
        source_notes: str,
    ) -> dict[str, Any]:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        return {
            "schema_version": 1,
            "batch_id": batch_id,
            "batch_name": batch_name,
            "batch_version": 1,
            "generated_at": timestamp,
            "source": {
                "type": "webapp",
                "name": "Factor evaluation web platform",
                "notes": source_notes,
            },
            "formula_language": "engine expression",
            "factor_count": 0,
            "factors": [],
        }

    def _write_batch(self, payload: dict[str, Any], path: Path) -> None:
        if not payload["factors"]:
            if path.exists():
                path.unlink()
            return
        payload["factor_count"] = len(payload["factors"])
        payload["batch_version"] = int(payload.get("batch_version", 0)) + 1
        payload["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        validate_factor_batch(payload)
        self.settings.registry_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{payload['batch_id']}.",
            suffix=".tmp",
            dir=self.settings.registry_dir,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def create(self, factor: Mapping[str, Any]) -> dict[str, Any]:
        with self._write_lock:
            name = str(factor["factor_name"])
            if any(row["factor_name"] == name for row in self.factors("all")):
                raise ValueError(f"因子名称已存在: {name}")
            payload = self._load_test()
            record = self._record(
                factor,
                number=self._next_number(payload),
                implementation_set="webapp_test",
            )
            payload["factors"].append(record)
            self._write_batch(payload, self.settings.test_registry_path)
            tags = normalize_tags(list(factor.get("tags") or []))
            self.db.replace_factor_tags(payload["batch_id"], name, tags)
            return self._decorate_factor(
                payload,
                record,
                self._submitted_names(),
                tags=tags,
            )

    def update(
        self,
        batch_id: str,
        factor_name: str,
        factor: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        with self._write_lock:
            if batch_id != self.settings.test_batch_id:
                return None
            payload = self._load_test()
            index = next(
                (
                    index
                    for index, record in enumerate(payload["factors"])
                    if record["factor_name"] == factor_name
                ),
                None,
            )
            if index is None:
                return None
            requested_name = str(factor["factor_name"])
            if requested_name != factor_name and any(
                row["factor_name"] == requested_name for row in self.factors("all")
            ):
                raise ValueError(f"因子名称已存在: {requested_name}")
            previous = payload["factors"][index]
            record = self._record(
                factor,
                number=int(previous["number"]),
                entered_at=str(previous["entered_at"]),
                implementation_set=str(previous.get("implementation_set", "webapp_test")),
            )
            payload["factors"][index] = record
            self._write_batch(payload, self.settings.test_registry_path)
            requested_tags = factor.get("tags")
            tags = (
                normalize_tags(list(requested_tags))
                if requested_tags is not None
                else self.db.tags_for_factors([(batch_id, factor_name)])[
                    (batch_id, factor_name)
                ]
            )
            self.db.replace_factor_tags(batch_id, requested_name, tags)
            if requested_name != factor_name:
                self.db.delete_factor_tags(batch_id, factor_name)
            return self._decorate_factor(
                payload,
                record,
                self._submitted_names(),
                tags=tags,
            )

    def delete(self, batch_id: str, factor_name: str) -> bool:
        with self._write_lock:
            if batch_id != self.settings.test_batch_id:
                return False
            payload = self._load_test()
            filtered = [
                row for row in payload["factors"] if row["factor_name"] != factor_name
            ]
            if len(filtered) == len(payload["factors"]):
                return False
            payload["factors"] = filtered
            self._write_batch(payload, self.settings.test_registry_path)
            self.db.delete_factor_tags(batch_id, factor_name)
            return True

    def remove_from_library(self, batch_id: str, factor_name: str) -> bool:
        """Remove a submitted copy while preserving its source test factor.

        Formal-library admission copies a definition instead of moving it.  A
        withdrawal therefore only removes the submitted definition and its
        formal-library tags; test-library metadata and evaluation history stay
        intact.
        """
        with self._write_lock:
            if batch_id != self.settings.submitted_batch_id:
                return False
            payload = self._load_submitted()
            removed = next(
                (
                    row
                    for row in payload["factors"]
                    if row["factor_name"] == factor_name
                ),
                None,
            )
            if removed is None:
                return False
            payload["factors"] = [
                row for row in payload["factors"] if row["factor_name"] != factor_name
            ]
            self._write_batch(payload, self.settings.submitted_registry_path)
            self.db.delete_factor_tags(batch_id, factor_name)
            self.correlation.invalidate_removed_factor(removed)
            return True

    def submit(
        self,
        batch_id: str,
        factor_name: str,
        *,
        candidate_profitability: _PerformanceSnapshot | None = None,
    ) -> dict[str, Any] | None:
        with self._write_lock:
            factor = self.find(batch_id, factor_name)
            if not factor or factor.get("library_scope") == "factor":
                return None
            existing = self.find(self.settings.submitted_batch_id, factor_name)
            if existing:
                return existing
            payload = self._load_submitted()
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            record = {
                **self._record(
                    factor,
                    number=self._next_number(payload),
                    entered_at=str(factor.get("entered_at") or now),
                    implementation_set="submitted",
                ),
                "submitted_at": now,
                "source_batch_id": batch_id,
                "source_factor_name": factor_name,
            }
            # Keep the factor registry definition-only.  The correlation
            # service writes its derived matrix only after the registry write
            # succeeds, while reusing cached exposures for the old factors.
            try:
                prepared_correlation = self.correlation.prepare_candidate(
                    record,
                    payload["factors"],
                )
                admission_decision: dict[str, Any] | None = None
                payload["factors"].append(record)
            except FactorCorrelationThresholdError as exc:
                (
                    prepared_correlation,
                    payload["factors"],
                    admission_decision,
                ) = self._prepare_correlated_replacement(
                    candidate_record=record,
                    candidate_source=factor,
                    candidate_profitability=candidate_profitability,
                    submitted_payload=payload,
                    violation=exc,
                )
            self._write_batch(payload, self.settings.submitted_registry_path)
            self.correlation.commit_candidate(prepared_correlation)
            tags = list(factor.get("tags", []))
            self.db.replace_factor_tags(payload["batch_id"], factor_name, tags)
            if admission_decision is not None:
                for displaced_name in admission_decision["replaced_factor_names"]:
                    self.db.delete_factor_tags(
                        self.settings.submitted_batch_id,
                        displaced_name,
                    )
            decorated = self._decorate_factor(
                payload,
                record,
                self._submitted_names(),
                tags=tags,
            )
            if admission_decision is not None:
                decorated["admission_decision"] = admission_decision
            return decorated

    @staticmethod
    def _finite_metric(metrics: Mapping[str, Any], name: str) -> float | None:
        try:
            value = float(metrics.get(name))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @classmethod
    def _performance_snapshot(
        cls,
        *,
        factor_name: str,
        batch_id: str,
        metrics: Mapping[str, Any],
        horizon: int,
        n_quantiles: int,
        methods: tuple[str, ...],
        source: str,
        run_id: str | None,
        evaluated_at: str | None,
    ) -> _PerformanceSnapshot | None:
        sharpe = cls._finite_metric(metrics, "gn_rolling_sharpe_60_median")
        if sharpe is None or metrics.get("return_definition") != RETURN_DEFINITION:
            return None
        fitness = cls._finite_metric(metrics, "fitness")
        signature = (
            int(horizon),
            int(n_quantiles),
            RETURN_DEFINITION,
            metrics.get("signal_start"),
            metrics.get("signal_end"),
            metrics.get("sample_start_day"),
            metrics.get("sample_end_day"),
            methods,
        )
        return _PerformanceSnapshot(
            factor_name=factor_name,
            batch_id=batch_id,
            rolling_sharpe_60_median=sharpe,
            fitness=fitness,
            source=source,
            run_id=run_id,
            evaluated_at=evaluated_at,
            comparison_signature=signature,
        )

    def _performance_source(self, factor: Mapping[str, Any]) -> dict[str, Any]:
        """Use a formal factor's original test definition and run history."""

        source_batch_id = str(factor.get("source_batch_id") or "").strip()
        source_factor_name = str(factor.get("source_factor_name") or "").strip()
        if source_batch_id and source_factor_name:
            source = self.find(source_batch_id, source_factor_name)
            if source and str(source.get("expression")) == str(factor.get("expression")):
                return source
        batch_id = str(factor.get("batch_id") or self.settings.submitted_batch_id)
        return {**factor, "batch_id": batch_id}

    def _existing_profitability_snapshots(
        self,
        factor: Mapping[str, Any],
    ) -> list[_PerformanceSnapshot]:
        source = self._performance_source(factor)
        factor_name = str(source["factor_name"])
        batch_id = str(source["batch_id"])
        snapshots: list[_PerformanceSnapshot] = []
        for run in self.db.list_runs(factor_name=factor_name, limit=500):
            if run.get("batch_id") != batch_id or run.get("status") != "succeeded":
                continue
            if tuple(run.get("methods") or []) != ADMISSION_PROFITABILITY_METHOD_NAMES:
                continue
            result = run.get("result")
            if not isinstance(result, Mapping):
                continue
            snapshot = self._performance_snapshot(
                factor_name=factor_name,
                batch_id=batch_id,
                metrics=result,
                horizon=int(run["horizon"]),
                n_quantiles=int(run["n_quantiles"]),
                methods=ADMISSION_PROFITABILITY_METHOD_NAMES,
                source="existing_profitability_run",
                run_id=str(run["id"]),
                evaluated_at=run.get("finished_at"),
            )
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    @staticmethod
    def _matching_snapshots(
        candidate: list[_PerformanceSnapshot],
        incumbent: list[_PerformanceSnapshot],
    ) -> tuple[_PerformanceSnapshot, _PerformanceSnapshot] | None:
        for candidate_snapshot in candidate:
            for incumbent_snapshot in incumbent:
                if candidate_snapshot.comparison_signature == incumbent_snapshot.comparison_signature:
                    return candidate_snapshot, incumbent_snapshot
        return None

    @staticmethod
    def _strictly_higher(left: float, right: float) -> bool:
        return left > right and not math.isclose(
            left,
            right,
            rel_tol=_PERFORMANCE_TIE_REL_TOLERANCE,
            abs_tol=_PERFORMANCE_TIE_ABS_TOLERANCE,
        )

    @classmethod
    def _compare_profitability(
        cls,
        candidate: _PerformanceSnapshot,
        incumbent: _PerformanceSnapshot,
    ) -> dict[str, Any] | None:
        """Compare by 60-day Sharpe, then Fitness only for a Sharpe tie."""

        if cls._strictly_higher(
            candidate.rolling_sharpe_60_median,
            incumbent.rolling_sharpe_60_median,
        ):
            return {
                "winner": "candidate",
                "criterion": "gn_rolling_sharpe_60_median",
                "candidate": candidate.as_dict(),
                "incumbent": incumbent.as_dict(),
            }
        if cls._strictly_higher(
            incumbent.rolling_sharpe_60_median,
            candidate.rolling_sharpe_60_median,
        ):
            return {
                "winner": "incumbent",
                "criterion": "gn_rolling_sharpe_60_median",
                "candidate": candidate.as_dict(),
                "incumbent": incumbent.as_dict(),
            }
        if candidate.fitness is None or incumbent.fitness is None:
            return None
        if cls._strictly_higher(candidate.fitness, incumbent.fitness):
            winner = "candidate"
        elif cls._strictly_higher(incumbent.fitness, candidate.fitness):
            winner = "incumbent"
        else:
            winner = "tie"
        return {
            "winner": winner,
            "criterion": "fitness",
            "candidate": candidate.as_dict(),
            "incumbent": incumbent.as_dict(),
        }

    def _run_admission_profitability(
        self,
        factor: Mapping[str, Any],
    ) -> _PerformanceSnapshot:
        """Persist and run the canonical profitability template synchronously.

        This path is intentionally rare: it runs only after a correlation
        conflict has no directly comparable completed profitability runs.  The
        run is written to the normal Webapp history so the automatic admission
        comparison remains inspectable just like a user-started evaluation.
        """

        source = self._performance_source(factor)
        factor_name = str(source["factor_name"])
        batch_id = str(source["batch_id"])
        job_id = uuid4().hex
        run = make_run(
            job_id=job_id,
            factor={
                "factor_name": factor_name,
                "batch_id": batch_id,
                "expression": str(source["expression"]),
            },
            stage_name=None,
            methods=list(ADMISSION_PROFITABILITY_METHOD_NAMES),
            horizon=ADMISSION_HORIZON,
            n_quantiles=ADMISSION_QUANTILES,
            runs_dir=self.settings.runs_dir,
            # Do not let the asynchronous worker claim a run being evaluated
            # below in the submit request.
            status="running",
            run_params={
                "template_name": ADMISSION_PROFITABILITY_TEMPLATE_NAME,
                "admission_comparison": True,
            },
        )
        self.db.create_job(
            {
                "id": job_id,
                "kind": "evaluate",
                "title": f"因子库相关性冲突盈利能力复核 · {factor_name}",
                "params": {
                    "template_name": ADMISSION_PROFITABILITY_TEMPLATE_NAME,
                    "methods": list(ADMISSION_PROFITABILITY_METHOD_NAMES),
                    "horizon": ADMISSION_HORIZON,
                    "n_quantiles": ADMISSION_QUANTILES,
                    "admission_comparison": True,
                },
                "created_at": run["created_at"],
            },
            [run],
        )
        try:
            # The normal worker sets this before plotting.  This direct,
            # on-demand path needs the same writable Matplotlib cache.
            os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
            methods = tuple(
                REGISTERED_EVALUATION_METHODS[name]
                for name in ADMISSION_PROFITABILITY_METHOD_NAMES
            )
            result = evaluate_factor_expression(
                factor_name=factor_name,
                expression=str(source["expression"]),
                data_dir=self.settings.data_dir,
                output_dir=run["output_dir"],
                horizon=ADMISSION_HORIZON,
                n_quantiles=ADMISSION_QUANTILES,
                evaluation_methods=methods,
            )
            metrics = result["metrics"]
            self.db.complete_run(run["id"], metrics)
        except Exception as exc:
            self.db.fail_run(run["id"], f"{type(exc).__name__}: {exc}")
            raise FactorLibraryPerformanceEvaluationError(
                f"因子 {factor_name} 缺少可比较的盈利能力结果，且自动运行"
                f"「{ADMISSION_PROFITABILITY_TEMPLATE_NAME}」失败：{exc}"
            ) from exc
        snapshot = self._performance_snapshot(
            factor_name=factor_name,
            batch_id=batch_id,
            metrics=metrics,
            horizon=ADMISSION_HORIZON,
            n_quantiles=ADMISSION_QUANTILES,
            methods=ADMISSION_PROFITABILITY_METHOD_NAMES,
            source="admission_profitability_template",
            run_id=str(run["id"]),
            evaluated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        if snapshot is None:
            raise FactorLibraryPerformanceEvaluationError(
                f"因子 {factor_name} 的「{ADMISSION_PROFITABILITY_TEMPLATE_NAME}」"
                "没有产生有限的 60 日窗口 Sharpe 中位数"
            )
        return snapshot

    def _gp_profitability_snapshot(
        self,
        request: Mapping[str, Any],
        test_factor: Mapping[str, Any],
    ) -> _PerformanceSnapshot | None:
        """Read the GP candidate's frozen test result for this admission only.

        Factor definitions remain definition-only and the Webapp run history is
        the single durable source for evaluations.  If the incumbent has no
        comparable normal run, both factors are evaluated via the standard
        profitability template below.
        """

        evaluation = request.get("profitability_evaluation")
        if not isinstance(evaluation, Mapping):
            return None
        metrics = evaluation.get("metrics")
        methods = tuple(evaluation.get("methods") or [])
        if (
            evaluation.get("standard") != PROFITABILITY_STANDARD_NAME
            or not isinstance(metrics, Mapping)
            or methods != ADMISSION_PROFITABILITY_METHOD_NAMES
        ):
            return None
        try:
            horizon = int(evaluation["horizon"])
            n_quantiles = int(evaluation["n_quantiles"])
        except (KeyError, TypeError, ValueError):
            return None
        return self._performance_snapshot(
            factor_name=str(test_factor["factor_name"]),
            batch_id=str(test_factor["batch_id"]),
            metrics=metrics,
            horizon=horizon,
            n_quantiles=n_quantiles,
            methods=methods,
            source="gp_frozen_profitability_test",
            run_id=None,
            evaluated_at=str(request.get("requested_at") or "") or None,
        )

    def _resolve_profitability_comparison(
        self,
        candidate: Mapping[str, Any],
        incumbent: Mapping[str, Any],
        candidate_profitability: _PerformanceSnapshot | None = None,
    ) -> dict[str, Any]:
        candidate_snapshots = self._existing_profitability_snapshots(candidate)
        if candidate_profitability is not None:
            candidate_snapshots.insert(0, candidate_profitability)
        matched = self._matching_snapshots(
            candidate_snapshots,
            self._existing_profitability_snapshots(incumbent),
        )
        if matched is not None:
            comparison = self._compare_profitability(*matched)
            if comparison is not None:
                return comparison

        # A missing, stale, non-comparable, or Sharpe-tied-without-Fitness run
        # cannot safely decide admission. Re-run both factors under exactly the
        # same canonical profitability template and full current data history.
        comparison = self._compare_profitability(
            self._run_admission_profitability(candidate),
            self._run_admission_profitability(incumbent),
        )
        if comparison is None:
            raise FactorLibraryPerformanceEvaluationError(
                "60 日窗口 Sharpe 中位数相同，但自动盈利能力测试未产生双方可比较的有限 Fitness"
            )
        return comparison

    def _prepare_correlated_replacement(
        self,
        *,
        candidate_record: Mapping[str, Any],
        candidate_source: Mapping[str, Any],
        candidate_profitability: _PerformanceSnapshot | None,
        submitted_payload: Mapping[str, Any],
        violation: FactorCorrelationThresholdError,
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        """Replace every correlated incumbent only when candidate beats all.

        The candidate must strictly win each conflicting pair.  Removing only a
        subset would either leave a correlation violation or make admission
        depend on arbitrary ordering, so a loss/tie against any incumbent keeps
        the formal library unchanged.
        """

        candidate_name = str(candidate_record["factor_name"])
        conflicting_names = sorted(
            {
                str(item["factor_b"])
                if str(item.get("factor_a")) == candidate_name
                else str(item["factor_a"])
                for item in violation.violations
                if candidate_name in {
                    str(item.get("factor_a")),
                    str(item.get("factor_b")),
                }
            }
        )
        if not conflicting_names:
            # Defensive: a stale pre-existing library violation is not a
            # candidate-vs-incumbent replacement decision.
            raise violation
        records_by_name = {
            str(record["factor_name"]): dict(record)
            for record in submitted_payload["factors"]
        }
        missing = [name for name in conflicting_names if name not in records_by_name]
        if missing:
            raise RuntimeError(
                "相关性服务返回了不在正式因子库中的冲突因子：" + "、".join(missing)
            )

        comparisons = [
            self._resolve_profitability_comparison(
                candidate_source,
                records_by_name[name],
                candidate_profitability,
            )
            for name in conflicting_names
        ]
        blockers = [
            comparison
            for comparison in comparisons
            if comparison["winner"] != "candidate"
        ]
        if blockers:
            criteria = sorted({str(item["criterion"]) for item in blockers})
            raise FactorLibraryPerformanceRejectedError(
                {
                    "action": "kept_incumbents",
                    "candidate_factor_name": candidate_name,
                    "blocking_factor_names": sorted(
                        str(item["incumbent"]["factor_name"]) for item in blockers
                    ),
                    "criterion": "、".join(criteria),
                    "comparisons": comparisons,
                    "correlation_violations": [dict(item) for item in violation.violations],
                }
            )

        remaining_records = [
            dict(record)
            for record in submitted_payload["factors"]
            if str(record["factor_name"]) not in set(conflicting_names)
        ]
        # Re-check after removing all beaten conflicts: the candidate may join
        # only if it is below the correlation threshold with every survivor.
        prepared = self.correlation.prepare_candidate(candidate_record, remaining_records)
        return (
            prepared,
            [*remaining_records, dict(candidate_record)],
            {
                "action": "replaced_correlated_factors",
                "candidate_factor_name": candidate_name,
                "replaced_factor_names": conflicting_names,
                "criterion": "gn_rolling_sharpe_60_median_then_fitness",
                "comparisons": comparisons,
                "correlation_violations": [dict(item) for item in violation.violations],
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )

    def submit_gp_candidate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Submit one GP-tested candidate through the formal-library service.

        The GP runner only writes a durable handoff request after its test gates
        pass.  This service owns test-library persistence, correlation checking,
        and the resulting formal-library decision.
        """

        factor_name = str(request.get("factor_name") or "").strip()
        expression = str(request.get("expression") or "").strip()
        if not factor_name or not expression:
            raise ValueError("GP 因子库提交请求缺少 factor_name 或 expression")

        with self._write_lock:
            formal = self.find(self.settings.submitted_batch_id, factor_name)
            if formal:
                if str(formal["expression"]) != expression:
                    return {
                        "status": "name_conflict",
                        "factor_name": factor_name,
                        "explanation": "正式因子库已有同名但不同表达式的因子",
                    }
                return {
                    "status": "admitted",
                    "factor_name": factor_name,
                    "already_present": True,
                    "correlation_checked": True,
                    "correlation_passed": True,
                    "formal_batch_id": self.settings.submitted_batch_id,
                    "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }

            test_factor = self.find(self.settings.test_batch_id, factor_name)
            if test_factor and str(test_factor["expression"]) != expression:
                return {
                    "status": "name_conflict",
                    "factor_name": factor_name,
                    "explanation": "测试库已有同名但不同表达式的因子",
                }
            if test_factor is None:
                existing = next(
                    (
                        item
                        for item in self.factors("all")
                        if str(item["factor_name"]) == factor_name
                    ),
                    None,
                )
                if existing is not None:
                    return {
                        "status": "name_conflict",
                        "factor_name": factor_name,
                        "explanation": "其他注册批次已有同名因子",
                    }
                symbols = expression_data_symbols(expression)
                test_factor = self.create(
                    {
                        "factor_name": factor_name,
                        "expression": expression,
                        "project": str(request.get("project") or "遗传规划"),
                        "uses_proxy": "vwap" in symbols,
                        "proxy_description": (
                            "vwap is vwap_proxy_df.pq = (high + low) / 2"
                            if "vwap" in symbols
                            else ""
                        ),
                        "tags": ["genetic-programming", "test-passed"],
                    }
                )
            candidate_profitability = self._gp_profitability_snapshot(
                request,
                test_factor,
            )

            try:
                formal = self.submit(
                    self.settings.test_batch_id,
                    factor_name,
                    candidate_profitability=candidate_profitability,
                )
            except FactorLibraryPerformanceRejectedError as exc:
                return {
                    "status": "rejected_performance",
                    "factor_name": factor_name,
                    "correlation_checked": True,
                    "correlation_passed": False,
                    "replacement_decision": exc.decision,
                    "explanation": str(exc),
                    "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            except FactorLibraryPerformanceEvaluationError as exc:
                return {
                    "status": "failed",
                    "factor_name": factor_name,
                    "correlation_checked": True,
                    "explanation": str(exc),
                    "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            except FactorCorrelationThresholdError as exc:
                return {
                    "status": "rejected_correlation",
                    "factor_name": factor_name,
                    "correlation_checked": True,
                    "correlation_passed": False,
                    "correlation_threshold": exc.threshold,
                    "violations": exc.violations,
                    "explanation": str(exc),
                    "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
            if formal is None:
                raise RuntimeError("GP 候选已写入测试库，但无法提交正式因子库")
            return {
                "status": "admitted",
                "factor_name": factor_name,
                "already_present": False,
                "correlation_checked": True,
                "correlation_passed": True,
                "formal_batch_id": self.settings.submitted_batch_id,
                **(
                    {
                        "replaced_factor_names": formal["admission_decision"][
                            "replaced_factor_names"
                        ],
                        "replacement_decision": formal["admission_decision"],
                    }
                    if formal.get("admission_decision")
                    else {}
                ),
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }

    def correlation_matrix(self) -> dict[str, Any]:
        """Read or rebuild the submitted-library correlation matrix."""
        with self._write_lock:
            return self.correlation.matrix(self._load_submitted()["factors"])

    def correlation_pair(self, factor_a: str, factor_b: str) -> dict[str, Any]:
        """Calculate non-overlapping correlation windows for a submitted pair."""
        with self._write_lock:
            return self.correlation.pair_detail(
                factor_a,
                factor_b,
                self._load_submitted()["factors"],
            )

    def _submitted_names(self) -> set[str]:
        payload = self._load_submitted()
        return {str(row["factor_name"]) for row in payload["factors"]}

    def _decorate_factor(
        self,
        batch: Mapping[str, Any],
        factor: Mapping[str, Any],
        submitted_names: set[str],
        *,
        tags: list[str],
    ) -> dict[str, Any]:
        batch_id = str(batch["batch_id"])
        library_scope = (
            "factor" if batch_id == self.settings.submitted_batch_id else "test"
        )
        row = {
            **deepcopy(factor),
            "batch_id": batch_id,
            "batch_name": batch["batch_name"],
            "library_scope": library_scope,
            "project": self._project_for(batch, factor),
            "factor_category": self._factor_category(batch_id, factor),
            "editable": batch_id == self.settings.test_batch_id,
            "submitted": library_scope == "factor"
            or str(factor["factor_name"]) in submitted_names,
            "tags": tags,
        }
        return row

    @classmethod
    def _project_for(
        cls,
        batch: Mapping[str, Any],
        factor: Mapping[str, Any],
    ) -> str:
        """Resolve legacy definitions while keeping new project values explicit."""
        for candidate in (factor.get("project"), batch.get("project")):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        source_batch_id = str(factor.get("source_batch_id", ""))
        batch_id = str(batch.get("batch_id", ""))
        if source_batch_id.startswith("alpha101_") or batch_id.startswith("alpha101_"):
            return "Alpha101"
        return cls.DEFAULT_PROJECT

    def _factor_category(self, batch_id: str, factor: Mapping[str, Any]) -> str:
        implementation_set = str(factor.get("implementation_set", ""))
        if batch_id == self.settings.submitted_batch_id:
            return "已提交因子"
        if batch_id in {self.settings.test_batch_id, self.settings.legacy_custom_batch_id}:
            return "用户测试因子"
        if implementation_set == "exact":
            return "Alpha101 精确输入"
        if implementation_set == "vwap_proxy":
            return "Alpha101 VWAP 代理"
        if implementation_set == "market_cap":
            return "Alpha101 市值因子"
        return implementation_set or "未分类"

    @staticmethod
    def _next_number(payload: Mapping[str, Any]) -> int:
        return max((int(row["number"]) for row in payload["factors"]), default=0) + 1

    @staticmethod
    def _record(
        factor: Mapping[str, Any],
        *,
        number: int,
        entered_at: str | None = None,
        implementation_set: str,
    ) -> dict[str, Any]:
        expression = str(factor["expression"]).strip()
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        return {
            "number": number,
            "factor_name": str(factor["factor_name"]).strip(),
            "entered_at": entered_at or now,
            "updated_at": now,
            "implementation_set": implementation_set,
            "project": str(factor.get("project") or RegistryService.DEFAULT_PROJECT).strip(),
            "expression": expression,
            "paper_expression": str(factor.get("paper_expression", "")),
            "required_symbols": sorted(expression_data_symbols(expression)),
            "uses_proxy": bool(factor.get("uses_proxy", False)),
            "proxy_description": str(factor.get("proxy_description", "")),
        }
