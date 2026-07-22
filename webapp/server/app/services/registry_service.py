"""Read factor batches and maintain web test/library batches atomically."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from engine import expression_data_symbols
from factor_correlation import (
    FactorCorrelationService,
    FactorCorrelationThresholdError,
)
from factor_registry import load_factor_batch, validate_factor_batch

from ..config import Settings
from ..db import Database
from ..tagging import normalize_tags


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

    def submit(self, batch_id: str, factor_name: str) -> dict[str, Any] | None:
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
            prepared_correlation = self.correlation.prepare_candidate(
                record,
                payload["factors"],
            )
            payload["factors"].append(record)
            self._write_batch(payload, self.settings.submitted_registry_path)
            self.correlation.commit_candidate(prepared_correlation)
            tags = list(factor.get("tags", []))
            self.db.replace_factor_tags(payload["batch_id"], factor_name, tags)
            return self._decorate_factor(
                payload,
                record,
                self._submitted_names(),
                tags=tags,
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

            try:
                formal = self.submit(self.settings.test_batch_id, factor_name)
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
