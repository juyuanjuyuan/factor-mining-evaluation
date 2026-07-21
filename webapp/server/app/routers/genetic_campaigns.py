"""Create and control genetic-programming factor-mining campaigns."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..models import GeneticCampaignCreate
from ..services.market_timeline import trading_days, validate_model_windows
from ..worker.genetic_supervisor import GeneticMiningSupervisor
from .dependencies import get_genetic_supervisor

from genetic_mining import fitness_backend_statuses
from genetic_mining.fitness_backends import require_fitness_backend


router = APIRouter(prefix="/genetic-campaigns", tags=["genetic-mining"])


def _required_supervisor(
    supervisor: GeneticMiningSupervisor | None,
) -> GeneticMiningSupervisor:
    if supervisor is None:
        raise HTTPException(503, "遗传挖掘服务当前未启用")
    return supervisor


@router.get("")
def list_campaigns(
    supervisor: GeneticMiningSupervisor | None = Depends(get_genetic_supervisor),
) -> list[dict]:
    return _required_supervisor(supervisor).list_campaigns()


@router.get("/backends")
def list_fitness_backends() -> list[dict]:
    return list(fitness_backend_statuses())


@router.post("", status_code=201)
def create_campaign(
    request: GeneticCampaignCreate,
    supervisor: GeneticMiningSupervisor | None = Depends(get_genetic_supervisor),
) -> dict:
    service = _required_supervisor(supervisor)
    try:
        validate_model_windows(
            service.settings,
            train_start=request.train_start,
            train_end=request.train_end,
            test_start=request.test_start,
            test_end=request.test_end,
            horizon=request.horizon,
        )
        require_fitness_backend(request.compute_backend)
        days = trading_days(service.settings)
        positions = {day: position for position, day in enumerate(days)}
        minimum_days = 60 + request.horizon + 1
        for label, start, end in (
            ("训练集", request.train_start, request.train_end),
            ("测试集", request.test_start, request.test_end),
        ):
            count = positions[end] - positions[start] + 1
            if count < minimum_days:
                raise ValueError(
                    f"{label}至少需要 {minimum_days} 个交易日，才能形成 60 个有效 IC/分段夏普观测"
                )
        return service.create_campaign(request.model_dump())
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{campaign}")
def get_campaign(
    campaign: str,
    supervisor: GeneticMiningSupervisor | None = Depends(get_genetic_supervisor),
) -> dict:
    try:
        return _required_supervisor(supervisor).get_campaign(campaign)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "遗传挖掘任务不存在") from exc


@router.post("/{campaign}/start")
def start_campaign(
    campaign: str,
    supervisor: GeneticMiningSupervisor | None = Depends(get_genetic_supervisor),
) -> dict:
    try:
        return _required_supervisor(supervisor).start_campaign(campaign)
    except KeyError as exc:
        raise HTTPException(404, "遗传挖掘任务不存在") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/{campaign}/stop")
def stop_campaign(
    campaign: str,
    supervisor: GeneticMiningSupervisor | None = Depends(get_genetic_supervisor),
) -> dict:
    try:
        return _required_supervisor(supervisor).stop_campaign(campaign)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "遗传挖掘任务不存在") from exc
