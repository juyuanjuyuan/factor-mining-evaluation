"""Immutable run results, detail series, artifacts, and comparisons."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from market_cycles import market_cycle_backgrounds

from ..db import Database
from ..models import CompareRequest
from ..services.run_reader import RunReader
from .dependencies import get_db, get_reader

router = APIRouter(tags=["runs"])


@router.get("/runs")
def list_runs(
    status: str | None = None,
    factor_name: str | None = None,
    job_id: str | None = None,
    limit: int = Query(default=300, ge=1, le=1000),
    db: Database = Depends(get_db),
) -> list[dict]:
    return db.list_runs(
        status=status, factor_name=factor_name, job_id=job_id, limit=limit
    )


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    db: Database = Depends(get_db),
    reader: RunReader = Depends(get_reader),
) -> dict:
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "结果不存在")
    run["detail_names"] = reader.detail_names(run) if run["result"] else []
    if "cycle_context" in run.get("methods", []):
        # Presentation metadata only: it is intentionally not persisted in
        # evaluation metrics/details and is shared with the matplotlib plot.
        run["market_cycle_backgrounds"] = market_cycle_backgrounds()
    return run


@router.get("/runs/{run_id}/details")
def list_details(
    run_id: str,
    db: Database = Depends(get_db),
    reader: RunReader = Depends(get_reader),
) -> dict:
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "结果不存在")
    return {"names": reader.detail_names(run)}


@router.get("/runs/{run_id}/details/{name}")
def get_detail(
    run_id: str,
    name: str,
    db: Database = Depends(get_db),
    reader: RunReader = Depends(get_reader),
) -> dict:
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "结果不存在")
    try:
        return reader.get_detail(run, name)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, f"明细不存在: {name}") from exc


@router.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(
    run_id: str,
    name: str,
    db: Database = Depends(get_db),
    reader: RunReader = Depends(get_reader),
) -> FileResponse:
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "结果不存在")
    try:
        path = reader.artifact_path(run, name)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, f"产物不存在: {name}") from exc
    return FileResponse(path, filename=Path(path).name)


@router.post("/compare")
def compare(
    request: CompareRequest,
    db: Database = Depends(get_db),
    reader: RunReader = Depends(get_reader),
) -> dict:
    runs = []
    for run_id in request.run_ids:
        run = db.get_run(run_id)
        if not run:
            raise HTTPException(404, f"结果不存在: {run_id}")
        if run["status"] != "succeeded" or not run["result"]:
            raise HTTPException(422, f"结果尚未成功完成: {run_id}")
        runs.append(run)
    excluded = {
        "run_id",
        "factor_name",
        "artifact_name",
        "expression",
        "return_definition",
        "evaluated_at",
        "evaluation_methods",
        "evaluation_details",
        "evaluation_artifacts",
        "quantile_net_commission_rate",
        "quantile_net_handling_fee_rate",
        "quantile_net_regulatory_fee_rate",
        "quantile_net_transfer_fee_rate",
        "quantile_net_stamp_tax_rate",
        "quantile_net_buy_cost_rate",
        "quantile_net_sell_cost_rate",
        "quantile_net_round_trip_cost_rate",
    }
    metric_names = sorted(
        {
            key
            for run in runs
            for key, value in run["result"].items()
            if key not in excluded and isinstance(value, (int, float, bool))
        }
    )
    metric_rows = [
        {
            "metric": metric,
            "values": {run["id"]: run["result"].get(metric) for run in runs},
        }
        for metric in metric_names
    ]
    curves = {"top_quantile": [], "cumulative_ic": []}
    for run in runs:
        label = run["factor_name"]
        names = reader.detail_names(run)
        cumulative_name = reader.latest_detail_name(names, "cumulative_returns")
        if cumulative_name:
            detail = reader.get_detail(run, cumulative_name)
            quantile_columns = [
                (int(name[1:]), index)
                for index, name in enumerate(detail["columns"])
                if name.startswith("G") and name[1:].isdigit()
            ]
            column = (
                max(quantile_columns, key=lambda item: item[0])[1]
                if quantile_columns
                else None
            )
            if column is not None:
                curves["top_quantile"].append(
                    {
                        "run_id": run["id"],
                        "name": label,
                        "data": [
                            [date, row[column]]
                            for date, row in zip(detail["index"], detail["data"])
                        ],
                    }
                )
        ic_name = reader.latest_detail_name(names, "ic")
        if ic_name:
            detail = reader.get_detail(run, ic_name)
            total = 0.0
            data = []
            for date, row in zip(detail["index"], detail["data"]):
                value = row[0] if row else None
                if value is not None:
                    total += value
                data.append([date, total])
            curves["cumulative_ic"].append(
                {"run_id": run["id"], "name": label, "data": data}
            )
    return {
        "runs": [
            {"id": run["id"], "factor_name": run["factor_name"]} for run in runs
        ],
        "metric_rows": metric_rows,
        "curves": curves,
    }
