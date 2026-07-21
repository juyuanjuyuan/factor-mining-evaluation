"""Pure model-test helpers shared by the train/test API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uuid import uuid4

from engine import parse_and_validate_expression
from model_training import (
    DEFAULT_MODEL_TRAINING_METHOD,
    ModelTerm,
    build_model_expression,
    normalize_training_parameters,
    resolve_model_training_method,
)

from ..config import Settings
from ..models import ModelTestInput
from .funnel_service import make_run
from .registry_service import RegistryService


def build_linear_expression(
    terms: list[Mapping[str, Any]],
    *,
    training_method: str = DEFAULT_MODEL_TRAINING_METHOD,
    training_params: Mapping[str, Any] | None = None,
) -> str:
    """Build the selected trainer's auditable provisional score expression."""
    frozen = tuple(
        ModelTerm(
            batch_id=str(term["batch_id"]),
            factor_name=str(term["factor_name"]),
            expression=str(term["expression"]),
            weight=float(term["weight"]),
        )
        for term in terms
    )
    return build_model_expression(
        training_method,
        frozen,
        tuple(term.weight for term in frozen),
        training_params or {},
    )


def resolve_model_input(
    request: ModelTestInput,
    registry: RegistryService,
) -> dict[str, Any]:
    """Freeze submitted-library definitions and produce the executable blend."""

    terms: list[dict[str, Any]] = []
    for requested in request.terms:
        factor = registry.find(requested.batch_id, requested.factor_name)
        if not factor:
            raise ValueError(
                f"因子不存在: {requested.batch_id}/{requested.factor_name}"
            )
        if factor.get("library_scope") != "factor":
            raise ValueError(
                f"只能选择因子库中的已提交因子: {requested.factor_name}"
            )
        expression = str(factor["expression"])
        parse_and_validate_expression(expression)
        terms.append(
            {
                "batch_id": requested.batch_id,
                "factor_name": requested.factor_name,
                "weight": requested.weight,
                "expression": expression,
            }
        )
    resolve_model_training_method(request.training_method)
    training_params = normalize_training_parameters(
        request.training_method, request.training_params
    )
    expression = build_linear_expression(
        terms,
        training_method=request.training_method,
        training_params=training_params,
    )
    return {
        **request.model_dump(exclude={"terms"}),
        "terms": terms,
        "expression": expression,
        "training_params": training_params,
    }


def make_model_run(
    model: Mapping[str, Any],
    *,
    section: str,
    job_id: str,
    settings: Settings,
) -> dict[str, Any]:
    if section not in {"training", "testing"}:
        raise ValueError(f"Unknown model-test section: {section}")
    is_training = section == "training"
    signal_start = str(model["train_start"] if is_training else model["test_start"])
    signal_end = str(model["train_end"] if is_training else model["test_end"])
    label = "训练集" if is_training else "样本外测试集"
    run_params: dict[str, Any] = {
        "model_test_id": model["id"],
        "model_section": section,
        "signal_start": signal_start,
        "signal_end": signal_end,
        "training_method": model["training_method"],
        "training_params": model["training_params"],
    }
    if is_training:
        run_params["model_training"] = {
            "method": model["training_method"],
            "parameters": model["training_params"],
            "terms": model["terms"],
        }
    elif model.get("fit_result"):
        run_params["fit_result"] = model["fit_result"]
    return make_run(
        job_id=job_id,
        factor={
            "factor_name": f"{model['model_name']} · {label}",
            "expression": model["expression"],
        },
        stage_name=label,
        methods=list(model["methods"]),
        horizon=int(model["horizon"]),
        n_quantiles=int(model["n_quantiles"]),
        runs_dir=settings.runs_dir,
        run_params=run_params,
    )


def make_model_job(model: Mapping[str, Any], *, section: str) -> dict[str, Any]:
    label = "训练集" if section == "training" else "样本外测试集"
    return {
        "id": uuid4().hex,
        "kind": "evaluate",
        "title": f"模型测试 · {model['model_name']} · {label}",
        "params": {
            "model_test_id": model["id"],
            "model_section": section,
            "signal_start": model["train_start"] if section == "training" else model["test_start"],
            "signal_end": model["train_end"] if section == "training" else model["test_end"],
            "horizon": model["horizon"],
            "n_quantiles": model["n_quantiles"],
            "methods": model["methods"],
            "training_method": model["training_method"],
            "training_params": model["training_params"],
        },
        "created_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
    }
