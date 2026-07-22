"""Expression, evaluation-method, and factor catalog endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from engine import (
    expression_data_symbols,
    expression_operator_names,
    parse_and_validate_expression,
)
from funnel import FUNNEL_STAGES
from operator_causality import operator_causality_catalog
from evaluators import (
    DEFAULT_EVALUATION_METHODS,
    evaluation_method_names,
    resolve_evaluation_methods,
)
from evaluators.base import REGISTERED_EVALUATION_METHODS, method_metadata

from ..db import Database
from ..models import (
    ExpressionRequest,
    FactorCreate,
    FactorProjectUpdate,
    FactorTagsUpdate,
    MethodsRequest,
)
from ..services.funnel_service import REQUIRED_STAGE_SEQUENCES, default_funnel_stages
from ..services.market_timeline import timeline_payload
from ..services.registry_service import RegistryService
from .dependencies import get_db, get_registry

router = APIRouter(tags=["catalog"])

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def _last_modified(factor: dict) -> datetime:
    """Sort key: newest edit first, falling back to first-entry time."""
    stamp = factor.get("updated_at") or factor.get("entered_at")
    if not isinstance(stamp, str):
        return _EPOCH
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return _EPOCH


@router.get("/methods")
def list_methods() -> list[dict]:
    default_names = list(evaluation_method_names(DEFAULT_EVALUATION_METHODS))
    defaults = set(default_names)
    ordered_names = default_names + sorted(
        set(REGISTERED_EVALUATION_METHODS) - defaults
    )
    return [
        {
            "name": metadata.name,
            "requires": list(metadata.requires),
            "provides": list(metadata.provides),
            "required_data_symbols": list(metadata.required_data_symbols),
            "is_default": metadata.name in defaults,
        }
        for metadata in (
            method_metadata(REGISTERED_EVALUATION_METHODS[name])
            for name in ordered_names
        )
    ]


@router.get("/operators/causality")
def list_operator_causality() -> list[dict]:
    return operator_causality_catalog()


@router.get("/funnel-stages")
def list_funnel_stages() -> list[dict]:
    gates = {stage.name: stage.gate_metric for stage in FUNNEL_STAGES}
    return [
        {
            **stage,
            "gate_metric": gates[stage["name"]],
            "required_methods": list(REQUIRED_STAGE_SEQUENCES[stage["name"]]),
        }
        for stage in default_funnel_stages()
    ]


@router.post("/methods/resolve")
def resolve_methods(request: MethodsRequest) -> dict:
    try:
        methods = resolve_evaluation_methods(request.names)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"methods": list(evaluation_method_names(methods))}


@router.post("/expressions/validate")
def validate_expression(request: ExpressionRequest) -> dict:
    try:
        parse_and_validate_expression(request.expression)
        symbols = sorted(expression_data_symbols(request.expression))
        operators = sorted(expression_operator_names(request.expression))
    except (SyntaxError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"valid": True, "symbols": symbols, "operators": operators}


@router.get("/market-data/timeline")
def market_data_timeline(
    registry: RegistryService = Depends(get_registry),
) -> dict:
    """Trading-day positions used by the model train/test range sliders."""

    try:
        return timeline_payload(registry.settings)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/factors")
def list_factors(
    search: str | None = Query(default=None),
    registry: RegistryService = Depends(get_registry),
    db: Database = Depends(get_db),
) -> list[dict]:
    factors = registry.factors("factor")
    return _attach_latest_runs(factors, search, db)


@router.get("/factor-correlation")
def get_factor_correlation(
    registry: RegistryService = Depends(get_registry),
) -> dict:
    """Return the persisted, submitted-library factor correlation matrix."""
    try:
        return registry.correlation_matrix()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/factor-correlation/pair")
def get_factor_correlation_pair(
    factor_a: str = Query(min_length=1),
    factor_b: str = Query(min_length=1),
    registry: RegistryService = Depends(get_registry),
) -> dict:
    """Return 60-day non-overlapping correlations for two submitted factors."""
    try:
        return registry.correlation_pair(factor_a, factor_b)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/test-factors")
def list_test_factors(
    search: str | None = Query(default=None),
    registry: RegistryService = Depends(get_registry),
    db: Database = Depends(get_db),
) -> list[dict]:
    factors = registry.factors("test")
    return _attach_latest_runs(factors, search, db)


@router.get("/factor-tags")
def list_factor_tags(
    library: str = Query(default="test", pattern="^(test|factor)$"),
    registry: RegistryService = Depends(get_registry),
) -> list[dict]:
    return registry.tag_summary(library)


def _attach_latest_runs(
    factors: list[dict],
    search: str | None,
    db: Database,
) -> list[dict]:
    if search:
        term = search.casefold()
        factors = [
            factor
            for factor in factors
            if term in factor["factor_name"].casefold()
            or term in factor["expression"].casefold()
        ]
    factors.sort(key=_last_modified, reverse=True)
    for factor in factors:
        latest = db.latest_run(factor["batch_id"], factor["factor_name"])
        factor["latest_run"] = latest
    return factors


@router.get("/factors/{batch_id}/{factor_name}")
def get_factor(
    batch_id: str,
    factor_name: str,
    registry: RegistryService = Depends(get_registry),
    db: Database = Depends(get_db),
) -> dict:
    factor = registry.find(batch_id, factor_name)
    if not factor:
        raise HTTPException(404, "因子不存在")
    factor["runs"] = db.list_runs(factor_name=factor_name, limit=100)
    factor["runs"] = [run for run in factor["runs"] if run["batch_id"] == batch_id]
    return factor


@router.post("/factors", status_code=201)
def create_factor(
    request: FactorCreate,
    registry: RegistryService = Depends(get_registry),
) -> dict:
    try:
        return registry.create(request.model_dump())
    except (SyntaxError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/test-factors", status_code=201)
def create_test_factor(
    request: FactorCreate,
    registry: RegistryService = Depends(get_registry),
) -> dict:
    try:
        return registry.create(request.model_dump())
    except (SyntaxError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.put("/factors/{batch_id}/{factor_name}")
def update_factor(
    batch_id: str,
    factor_name: str,
    request: FactorCreate,
    registry: RegistryService = Depends(get_registry),
) -> dict:
    if batch_id != registry.settings.test_batch_id:
        raise HTTPException(403, "只有测试库中新建的因子可编辑")
    try:
        factor = registry.update(batch_id, factor_name, request.model_dump())
    except (SyntaxError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if not factor:
        raise HTTPException(404, "因子不存在")
    return factor


@router.put("/test-factors/{batch_id}/{factor_name}")
def update_test_factor(
    batch_id: str,
    factor_name: str,
    request: FactorCreate,
    registry: RegistryService = Depends(get_registry),
) -> dict:
    return update_factor(batch_id, factor_name, request, registry)


@router.put("/factors/{batch_id}/{factor_name}/tags")
def update_factor_tags(
    batch_id: str,
    factor_name: str,
    request: FactorTagsUpdate,
    registry: RegistryService = Depends(get_registry),
) -> dict:
    factor = registry.set_tags(batch_id, factor_name, request.tags)
    if not factor:
        raise HTTPException(404, "因子不存在")
    return factor


@router.put("/factors/{batch_id}/{factor_name}/project")
def update_factor_project(
    batch_id: str,
    factor_name: str,
    request: FactorProjectUpdate,
    registry: RegistryService = Depends(get_registry),
) -> dict:
    """Allow project reclassification for both test and formal library factors."""
    try:
        factor = registry.set_project(batch_id, factor_name, request.project)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not factor:
        raise HTTPException(404, "因子不存在")
    return factor


@router.delete("/factors/{batch_id}/{factor_name}", status_code=204)
def delete_factor(
    batch_id: str,
    factor_name: str,
    registry: RegistryService = Depends(get_registry),
) -> None:
    if batch_id == registry.settings.test_batch_id:
        deleted = registry.delete(batch_id, factor_name)
    elif batch_id == registry.settings.submitted_batch_id:
        deleted = registry.remove_from_library(batch_id, factor_name)
    else:
        raise HTTPException(403, "只有测试库中新建的因子可删除，或可将已提交因子移出因子库")
    if not deleted:
        raise HTTPException(404, "因子不存在")


@router.delete("/test-factors/{batch_id}/{factor_name}", status_code=204)
def delete_test_factor(
    batch_id: str,
    factor_name: str,
    registry: RegistryService = Depends(get_registry),
) -> None:
    delete_factor(batch_id, factor_name, registry)


@router.post("/test-factors/{batch_id}/{factor_name}/submit", status_code=201)
def submit_test_factor(
    batch_id: str,
    factor_name: str,
    registry: RegistryService = Depends(get_registry),
) -> dict:
    try:
        factor = registry.submit(batch_id, factor_name)
    except (SyntaxError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if not factor:
        raise HTTPException(404, "测试因子不存在")
    return factor
