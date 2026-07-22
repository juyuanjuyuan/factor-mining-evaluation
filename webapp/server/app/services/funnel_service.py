"""Create and advance one factor through a validated four-stage funnel."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from evaluators.base import REGISTERED_EVALUATION_METHODS, method_metadata
from funnel import FUNNEL_STAGES, evaluate_gate


REQUIRED_STAGE_SEQUENCES: dict[str, tuple[str, ...]] = {
    "stage1_validity": ("future_data_perturbation",),
    "stage2_ic": ("rank_ic", "newey_west_ic_significance"),
    "stage2b_neutral": (
        "market_cap_neutralize",
        "rank_ic",
        "newey_west_ic_significance",
    ),
    "stage3_portfolio": ("quantile_returns",),
}


def default_funnel_stages() -> list[dict[str, Any]]:
    return [
        {"name": stage.name, "methods": list(stage.method_names)}
        for stage in FUNNEL_STAGES
    ]


def _contains_ordered_sequence(
    methods: list[str],
    required: tuple[str, ...],
) -> bool:
    cursor = 0
    for method in methods:
        if cursor < len(required) and method == required[cursor]:
            cursor += 1
    return cursor == len(required)


def normalize_funnel_stages(value: Any) -> list[dict[str, Any]]:
    """Validate editable methods while preserving canonical stage gates/order."""

    if value is None:
        return default_funnel_stages()
    if not isinstance(value, list) or len(value) != len(FUNNEL_STAGES):
        raise ValueError("漏斗模板必须包含标准四个阶段")

    normalized: list[dict[str, Any]] = []
    for index, (item, canonical) in enumerate(zip(value, FUNNEL_STAGES), start=1):
        if not isinstance(item, dict) or item.get("name") != canonical.name:
            raise ValueError(f"漏斗第 {index} 阶段必须是 {canonical.name}")
        raw_methods = item.get("methods")
        if not isinstance(raw_methods, list) or not raw_methods:
            raise ValueError(f"漏斗阶段 {canonical.name} 至少需要一个评价方法")
        methods = [str(name).strip() for name in raw_methods]
        if any(not name for name in methods):
            raise ValueError(f"漏斗阶段 {canonical.name} 包含空方法名")
        unknown = [name for name in methods if name not in REGISTERED_EVALUATION_METHODS]
        if unknown:
            raise ValueError(
                f"漏斗阶段 {canonical.name} 包含未知方法: {', '.join(unknown)}"
            )

        executed: set[str] = set()
        for position, name in enumerate(methods, start=1):
            metadata = method_metadata(REGISTERED_EVALUATION_METHODS[name])
            missing = [required for required in metadata.requires if required not in executed]
            if missing:
                raise ValueError(
                    f"漏斗阶段 {canonical.name} 第 {position} 步 {name} "
                    f"需要先执行 {', '.join(missing)}"
                )
            executed.update(metadata.provides)

        required_sequence = REQUIRED_STAGE_SEQUENCES[canonical.name]
        if not _contains_ordered_sequence(methods, required_sequence):
            raise ValueError(
                f"漏斗阶段 {canonical.name} 必须按顺序保留: "
                f"{' → '.join(required_sequence)}"
            )
        normalized.append({"name": canonical.name, "methods": methods})
    return normalized


def make_run(
    *,
    job_id: str,
    factor: Mapping[str, Any],
    stage_name: str | None,
    methods: list[str],
    horizon: int,
    n_quantiles: int,
    runs_dir: Path,
    status: str = "queued",
    run_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = uuid4().hex
    return {
        "id": run_id,
        "job_id": job_id,
        "factor_name": factor["factor_name"],
        "batch_id": factor.get("batch_id"),
        "expression": factor["expression"],
        "stage": stage_name,
        "status": status,
        "horizon": horizon,
        "n_quantiles": n_quantiles,
        "methods": methods,
        "run_params": dict(run_params or {}),
        "output_dir": str((runs_dir / run_id).resolve()),
        "created_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
    }


def first_funnel_run(
    *,
    job_id: str,
    factor: Mapping[str, Any],
    horizon: int,
    n_quantiles: int,
    runs_dir: Path,
    stages: list[dict[str, Any]] | None = None,
    run_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    configured = normalize_funnel_stages(stages)
    stage = configured[0]
    return make_run(
        job_id=job_id,
        factor=factor,
        stage_name=stage["name"],
        methods=list(stage["methods"]),
        horizon=horizon,
        n_quantiles=n_quantiles,
        runs_dir=runs_dir,
        run_params=run_params,
    )


def gate_result(
    stage_name: str,
    metrics: Mapping[str, Any],
    significance_level: float,
) -> tuple[str, Any, str]:
    stage = next(stage for stage in FUNNEL_STAGES if stage.name == stage_name)
    return evaluate_gate(stage, metrics, significance_level=significance_level)


def next_stage_index(
    stage_name: str,
    stages: list[dict[str, Any]] | None = None,
) -> int | None:
    configured = normalize_funnel_stages(stages)
    index = next(
        index for index, stage in enumerate(configured) if stage["name"] == stage_name
    )
    return index + 1 if index + 1 < len(configured) else None
