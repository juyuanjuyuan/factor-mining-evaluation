"""Batch job creation, progress, and cancellation."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from engine import parse_and_validate_expression
from ..config import settings
from ..db import Database
from ..models import JobCreate
from ..services.funnel_service import (
    first_funnel_run,
    make_run,
    normalize_funnel_stages,
)
from ..services.registry_service import RegistryService
from ..services.pipeline_service import validate_ordered_pipeline
from ..worker.supervisor import WorkerSupervisor
from .dependencies import get_db, get_registry, get_supervisor

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _validated_pipeline(names: list[str] | None) -> list[str]:
    """Validate an ordered pipeline, preserving order and duplicates.

    The pipeline is stored and executed exactly as given (a method may repeat,
    e.g. rank_ic before and after market_cap_neutralize). For duplicate-free
    lists whose order misses dependencies, fall back to the classic
    auto-resolution so template/API shorthand like ["rolling_sharpe"] keeps
    working.
    """

    try:
        return validate_ordered_pipeline(names)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("", status_code=201)
def create_job(
    request: JobCreate,
    db: Database = Depends(get_db),
    registry: RegistryService = Depends(get_registry),
) -> dict:
    kind = request.kind
    methods = request.methods
    template = db.get_template(request.template_id) if request.template_id else None
    if request.template_id and template is None:
        raise HTTPException(404, "模板不存在")
    if template:
        kind = "funnel" if template["kind"] == "funnel" else "evaluate"
        methods = template["methods"]
    resolved_factors = []
    if request.tags:
        try:
            factor_inputs = registry.factors_by_tags(
                request.tags,
                match=request.tag_match,
                library=request.library,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not factor_inputs:
            joined = "、".join(request.tags)
            raise HTTPException(422, f"标签「{joined}」没有匹配的因子")
    else:
        factor_inputs = request.factors

    for item in factor_inputs:
        if isinstance(item, dict):
            factor = {
                "factor_name": item["factor_name"],
                "batch_id": item["batch_id"],
                "expression": item["expression"],
            }
        elif item.expression is not None:
            factor = {
                "factor_name": item.factor_name,
                "batch_id": item.batch_id or "temporary",
                "expression": item.expression,
            }
        else:
            factor = registry.find(item.batch_id or "", item.factor_name)
            if not factor:
                raise HTTPException(
                    404, f"因子不存在: {item.batch_id}/{item.factor_name}"
                )
        try:
            parse_and_validate_expression(factor["expression"])
        except (SyntaxError, ValueError) as exc:
            raise HTTPException(
                422, f"{factor['factor_name']}: {exc}"
            ) from exc
        resolved_factors.append(factor)

    if kind == "evaluate":
        try:
            resolved_methods = _validated_pipeline(methods)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    else:
        resolved_methods = []
    try:
        funnel_stages = (
            normalize_funnel_stages(
                (template.get("params") or {}).get("stages") if template else None
            )
            if kind == "funnel"
            else []
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    # A screening gate only applies to custom evaluate pipelines; the funnel
    # carries its own canonical per-stage gates. Fall back to the template's
    # saved gate when the request does not send one.
    gate = None
    if kind == "evaluate":
        if request.gate is not None:
            gate = request.gate.model_dump()
        elif template:
            template_gate = (template.get("params") or {}).get("gate")
            if isinstance(template_gate, dict) and template_gate.get("conditions"):
                gate = template_gate

    job_id = uuid4().hex
    created_at = datetime.now().astimezone().isoformat(timespec="microseconds")
    params = {
        "horizon": request.horizon,
        "n_quantiles": request.n_quantiles,
        "methods": resolved_methods,
        "significance_level": request.significance_level,
        "template_id": request.template_id,
        "funnel_stages": funnel_stages,
        "gate": gate,
        "signal_start": request.signal_start,
        "signal_end": request.signal_end,
    }
    run_params = {
        "signal_start": request.signal_start,
        "signal_end": request.signal_end,
    } if request.signal_start is not None else {}
    if request.tags:
        params["tag_selection"] = {
            "tags": request.tags,
            "match": request.tag_match,
            "library": request.library,
        }
    if kind == "funnel":
        runs = [
            first_funnel_run(
                job_id=job_id,
                factor=factor,
                horizon=request.horizon,
                n_quantiles=request.n_quantiles,
                runs_dir=settings.runs_dir,
                stages=funnel_stages,
                run_params=run_params,
            )
            for factor in resolved_factors
        ]
    else:
        runs = [
            make_run(
                job_id=job_id,
                factor=factor,
                stage_name=None,
                methods=resolved_methods,
                horizon=request.horizon,
                n_quantiles=request.n_quantiles,
                runs_dir=settings.runs_dir,
                run_params=run_params,
            )
            for factor in resolved_factors
        ]
    db.create_job(
        {
            "id": job_id,
            "kind": kind,
            "title": request.title
            or (
                f"{'漏斗' if kind == 'funnel' else '评价'} · 标签「{'、'.join(request.tags)}」"
                f" · {len(runs)} 个因子"
                if request.tags
                else f"{'漏斗' if kind == 'funnel' else '评价'} · {len(runs)} 个因子"
            ),
            "params": params,
            "created_at": created_at,
        },
        runs,
    )
    return db.get_job(job_id)  # type: ignore[return-value]


@router.get("")
def list_jobs(
    limit: int = Query(default=100, ge=1, le=500),
    db: Database = Depends(get_db),
) -> list[dict]:
    return db.list_jobs(limit)


@router.get("/{job_id}")
def get_job(job_id: str, db: Database = Depends(get_db)) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str, db: Database = Depends(get_db)) -> None:
    try:
        found = db.delete_job(job_id)
    except PermissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not found:
        raise HTTPException(404, "任务不存在")


@router.post("/{job_id}/cancel")
def cancel_job(
    job_id: str,
    force: bool = False,
    db: Database = Depends(get_db),
    supervisor: WorkerSupervisor | None = Depends(get_supervisor),
) -> dict:
    if not db.cancel_job(job_id):
        raise HTTPException(404, "任务不存在")
    if force and supervisor:
        supervisor.force_cancel_job(job_id)
    return db.get_job(job_id)  # type: ignore[return-value]
