"""Train-then-holdout linear multi-factor model-test sessions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from model_training import (
    available_model_training_methods,
    resolve_model_training_method,
    training_method_metadata,
)

from ..db import Database
from ..models import ModelTestInput
from ..services.model_test_service import (
    make_model_job,
    make_model_run,
    resolve_model_input,
)
from ..services.market_timeline import validate_model_windows
from ..services.registry_service import RegistryService
from .dependencies import get_db, get_registry
from .jobs import _validated_pipeline


router = APIRouter(prefix="/models", tags=["model-tests"])


def _with_runs(model: dict, db: Database) -> dict:
    result = dict(model)
    training_id = result.get("training_run_id")
    testing_id = result.get("testing_run_id")
    result["training_run"] = db.get_run(training_id) if training_id else None
    result["testing_run"] = db.get_run(testing_id) if testing_id else None
    training_metadata = training_method_metadata(
        resolve_model_training_method(result.get("training_method"))
    )
    fit_ready = not training_metadata.requires_fitting or bool(
        result.get("fit_result")
        and result["fit_result"].get("method") == training_metadata.name
    )
    result["can_run_testing"] = bool(
        result["training_run"]
        and result["training_run"].get("status") == "succeeded"
        and fit_ready
        and not testing_id
    )
    return result


def _configuration(request: ModelTestInput, registry: RegistryService) -> dict:
    try:
        validate_model_windows(
            registry.settings,
            train_start=request.train_start,
            train_end=request.train_end,
            test_start=request.test_start,
            test_end=request.test_end,
            horizon=request.horizon,
        )
        configuration = resolve_model_input(request, registry)
        configuration["methods"] = _validated_pipeline(request.methods)
        return configuration
    except HTTPException:
        raise
    except (SyntaxError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("")
def list_models(
    limit: int = Query(default=100, ge=1, le=500),
    db: Database = Depends(get_db),
) -> list[dict]:
    return [_with_runs(model, db) for model in db.list_model_tests(limit)]


@router.get("/training-methods")
def list_training_methods() -> list[dict]:
    return [
        {
            "name": metadata.name,
            "label": metadata.label,
            "description": metadata.description,
            "requires_fitting": metadata.requires_fitting,
            "term_weight_editable": metadata.term_weight_editable,
            "parameters": [
                {
                    "name": parameter.name,
                    "label": parameter.label,
                    "default": parameter.default,
                    "minimum": parameter.minimum,
                    "maximum": parameter.maximum,
                    "step": parameter.step,
                    "description": parameter.description,
                }
                for parameter in metadata.parameters
            ],
        }
        for metadata in available_model_training_methods()
    ]


@router.post("", status_code=201)
def create_model(
    request: ModelTestInput,
    registry: RegistryService = Depends(get_registry),
    db: Database = Depends(get_db),
) -> dict:
    configuration = _configuration(request, registry)
    from uuid import uuid4

    configuration["id"] = uuid4().hex
    return _with_runs(db.create_model_test(configuration), db)


@router.get("/{model_id}")
def get_model(model_id: str, db: Database = Depends(get_db)) -> dict:
    model = db.get_model_test(model_id)
    if not model:
        raise HTTPException(404, "模型测试不存在")
    return _with_runs(model, db)


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: str, db: Database = Depends(get_db)) -> None:
    if not db.delete_model_test(model_id):
        raise HTTPException(404, "模型测试不存在")


@router.put("/{model_id}")
def update_model(
    model_id: str,
    request: ModelTestInput,
    registry: RegistryService = Depends(get_registry),
    db: Database = Depends(get_db),
) -> dict:
    configuration = _configuration(request, registry)
    try:
        model = db.update_model_test(model_id, configuration)
    except PermissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not model:
        raise HTTPException(404, "模型测试不存在")
    return _with_runs(model, db)


def _start_model_section(model_id: str, section: str, db: Database, registry: RegistryService) -> dict:
    model = db.get_model_test(model_id)
    if not model:
        raise HTTPException(404, "模型测试不存在")
    if model.get("testing_run_id"):
        raise HTTPException(409, "样本外测试已经提交，模型配置已锁定")
    if section == "training":
        previous_id = model.get("training_run_id")
        previous = db.get_run(previous_id) if previous_id else None
        if previous and previous.get("status") in {"queued", "running"}:
            raise HTTPException(409, "训练集评价正在执行，请等待完成后再调整或重跑")
    else:
        training_id = model.get("training_run_id")
        training = db.get_run(training_id) if training_id else None
        if not training or training.get("status") != "succeeded":
            raise HTTPException(409, "请先完成一次成功的训练集评价，再执行样本外测试")

    job = make_model_job(model, section=section)
    run = make_model_run(
        model,
        section=section,
        job_id=job["id"],
        settings=registry.settings,
    )
    db.create_job(job, [run])
    if not db.link_model_run(model_id, run["id"], section=section):
        raise HTTPException(409, "模型状态已变化，未能关联本次运行")
    current = db.get_model_test(model_id)
    if not current:
        raise HTTPException(404, "模型测试不存在")
    return _with_runs(current, db)


@router.post("/{model_id}/train", status_code=201)
def run_training(
    model_id: str,
    db: Database = Depends(get_db),
    registry: RegistryService = Depends(get_registry),
) -> dict:
    return _start_model_section(model_id, "training", db, registry)


@router.post("/{model_id}/test", status_code=201)
def run_testing(
    model_id: str,
    db: Database = Depends(get_db),
    registry: RegistryService = Depends(get_registry),
) -> dict:
    return _start_model_section(model_id, "testing", db, registry)
