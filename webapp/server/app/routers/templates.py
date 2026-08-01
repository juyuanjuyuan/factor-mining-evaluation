"""Pipeline template CRUD."""

from fastapi import APIRouter, Depends, HTTPException

from ..db import Database
from ..models import TemplateInput
from ..services.funnel_service import normalize_funnel_stages
from ..services.pipeline_service import validate_ordered_pipeline
from .dependencies import get_db

router = APIRouter(prefix="/templates", tags=["templates"])


def _validate(payload: TemplateInput) -> dict:
    data = payload.model_dump()
    if data["kind"] == "methods":
        try:
            data["methods"] = validate_ordered_pipeline(data["methods"])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        data["params"] = {}
    else:
        data["methods"] = []
        params = dict(data.get("params") or {})
        horizon = params.get("horizon", 1)
        n_quantiles = params.get("n_quantiles", 10)
        decay = params.get("decay", 1)
        significance_level = params.get("significance_level", 0.05)
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or not 1 <= horizon <= 60
        ):
            raise HTTPException(422, "漏斗模板持有期必须是 1–60 的整数")
        if (
            isinstance(n_quantiles, bool)
            or not isinstance(n_quantiles, int)
            or not 2 <= n_quantiles <= 20
        ):
            raise HTTPException(422, "漏斗模板分位组数必须是 2–20 的整数")
        if (
            isinstance(decay, bool)
            or not isinstance(decay, int)
            or decay < 1
        ):
            raise HTTPException(422, "漏斗模板 Decay 必须是正整数")
        if (
            isinstance(significance_level, bool)
            or not isinstance(significance_level, (int, float))
            or not 0 < float(significance_level) < 1
        ):
            raise HTTPException(422, "漏斗模板显著性水平必须在 0 和 1 之间")
        try:
            stages = normalize_funnel_stages(params.get("stages"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        data["params"] = {
            "horizon": horizon,
            "n_quantiles": n_quantiles,
            "decay": decay,
            "significance_level": float(significance_level),
            "stages": stages,
        }
    return data


@router.get("")
def list_templates(db: Database = Depends(get_db)) -> list[dict]:
    return db.list_templates()


@router.post("", status_code=201)
def create_template(
    payload: TemplateInput, db: Database = Depends(get_db)
) -> dict:
    try:
        return db.create_template(_validate(payload))
    except Exception as exc:
        if "UNIQUE" in str(exc):
            raise HTTPException(409, "模板名称已存在") from exc
        raise


@router.put("/{template_id}")
def update_template(
    template_id: int,
    payload: TemplateInput,
    db: Database = Depends(get_db),
) -> dict:
    try:
        result = db.update_template(template_id, _validate(payload))
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not result:
        raise HTTPException(404, "模板不存在")
    return result


@router.delete("/{template_id}", status_code=204)
def delete_template(template_id: int, db: Database = Depends(get_db)) -> None:
    try:
        found = db.delete_template(template_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    if not found:
        raise HTTPException(404, "模板不存在")
