"""Correlation-gated, definition-only admission into the formal factor library."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from engine import expression_data_symbols, parse_and_validate_expression
from factor_correlation import (
    CORRELATION_THRESHOLD,
    FactorCorrelationService,
    FactorCorrelationThresholdError,
)
from factor_registry import load_factor_batch, validate_factor_batch


@dataclass(frozen=True)
class CorrelationSettings:
    data_dir: Path
    state_dir: Path


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    already_present: bool
    factor_name: str
    correlation_passed: bool
    correlation_threshold: float
    explanation: str
    violations: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = [dict(item) for item in self.violations]
        return payload


def _empty_library(batch_id: str) -> dict[str, Any]:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "batch_name": "Web 平台因子库",
        "batch_version": 1,
        "generated_at": timestamp,
        "source": {
            "type": "webapp_and_cli",
            "name": "Factor evaluation factor library",
            "notes": "Definitions admitted by Webapp or evaluation-standard CLI",
        },
        "formula_language": "engine expression",
        "factor_count": 0,
        "factors": [],
    }


def _atomic_write_batch(path: Path, payload: Mapping[str, Any]) -> None:
    validate_factor_batch(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def admit_factor_to_library(
    *,
    factor_name: str,
    expression: str,
    library_file: str | Path,
    data_dir: str | Path,
    state_dir: str | Path,
    project: str = "遗传规划",
    source_batch_id: str = "genetic_programming",
    source_factor_name: str | None = None,
    notes: str = "",
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> AdmissionResult:
    """Admit exactly one candidate after the existing library correlation gate."""

    name = factor_name.strip()
    expression = expression.strip()
    if not name:
        raise ValueError("factor_name cannot be empty")
    parse_and_validate_expression(expression)
    path = Path(library_file).expanduser().resolve()
    payload = load_factor_batch(path) if path.is_file() else _empty_library(path.stem)
    existing = next(
        (item for item in payload["factors"] if item["factor_name"] == name),
        None,
    )
    if existing is not None:
        if str(existing["expression"]) != expression:
            raise ValueError(f"Factor name already exists with another expression: {name}")
        return AdmissionResult(
            admitted=False,
            already_present=True,
            factor_name=name,
            correlation_passed=True,
            correlation_threshold=float(correlation_threshold),
            explanation="因子已以相同表达式存在于正式因子库；按幂等成功处理",
        )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    record = {
        "number": max(
            (int(item.get("number", 0)) for item in payload["factors"]),
            default=0,
        )
        + 1,
        "factor_name": name,
        "entered_at": now,
        "updated_at": now,
        "implementation_set": "genetic_programming",
        "project": project.strip() or "遗传规划",
        "expression": expression,
        "paper_expression": "",
        "required_symbols": sorted(expression_data_symbols(expression)),
        "uses_proxy": "vwap" in expression_data_symbols(expression),
        "proxy_description": (
            "vwap is vwap_proxy_df.pq = (high + low) / 2"
            if "vwap" in expression_data_symbols(expression)
            else ""
        ),
        "notes": notes,
        "submitted_at": now,
        "source_batch_id": source_batch_id,
        "source_factor_name": source_factor_name or name,
    }
    service = FactorCorrelationService(
        CorrelationSettings(
            data_dir=Path(data_dir).expanduser().resolve(),
            state_dir=Path(state_dir).expanduser().resolve(),
        ),
        threshold=correlation_threshold,
    )
    try:
        prepared = service.prepare_candidate(record, payload["factors"])
    except FactorCorrelationThresholdError as exc:
        return AdmissionResult(
            admitted=False,
            already_present=False,
            factor_name=name,
            correlation_passed=False,
            correlation_threshold=float(correlation_threshold),
            explanation=str(exc),
            violations=tuple(exc.violations),
        )

    payload["factors"].append(record)
    payload["factor_count"] = len(payload["factors"])
    payload["batch_version"] = int(payload.get("batch_version", 0)) + 1
    payload["generated_at"] = now
    _atomic_write_batch(path, payload)
    service.commit_candidate(prepared)
    return AdmissionResult(
        admitted=True,
        already_present=False,
        factor_name=name,
        correlation_passed=True,
        correlation_threshold=float(correlation_threshold),
        explanation="评价标准与正式因子库相关性门槛均通过，已加入因子库",
    )
