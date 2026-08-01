"""Immutable run results, detail series, artifacts, and comparisons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from market_cycles import market_cycle_backgrounds

from ..db import Database
from ..sanitize import sanitize
from ..models import CompareRequest
from ..services.holding_reference_data import HoldingReferenceData
from ..services.run_reader import RunReader
from .dependencies import get_db, get_holding_reference_data, get_reader

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


def _holding_audit_summary_name(reader: RunReader, run: dict) -> str:
    name = reader.latest_detail_name(
        reader.detail_names(run), "holding_audit_summary"
    )
    if name is None or reader.latest_artifact_name(run, "holding_audit") is None:
        raise KeyError("holding_audit")
    return name


@router.get("/runs/{run_id}/holding-audit/dates")
def list_holding_audit_dates(
    run_id: str,
    db: Database = Depends(get_db),
    reader: RunReader = Depends(get_reader),
) -> dict:
    """Return the small per-day index for an optional holdings-audit artifact."""

    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "结果不存在")
    try:
        summary = reader.get_detail(run, _holding_audit_summary_name(reader, run))
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "该运行未启用查看仓单模块") from exc
    columns = summary["columns"]
    dates = [
        {
            "signal_day": day,
            **{
                column: row[position] if position < len(row) else None
                for position, column in enumerate(columns)
            },
        }
        for day, row in zip(summary["index"], summary["data"])
    ]
    return {
        "top_group": f"G{run['n_quantiles']}",
        "dates": dates,
    }


@router.get("/runs/{run_id}/holding-audit")
def get_holding_audit(
    run_id: str,
    signal_day: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    db: Database = Depends(get_db),
    reader: RunReader = Depends(get_reader),
    reference_data: HoldingReferenceData = Depends(get_holding_reference_data),
) -> dict:
    """Read one signal day's highest-quantile positions without full preload."""

    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "结果不存在")
    try:
        normalized_day = pd.Timestamp(signal_day).date().isoformat()
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "signal_day 必须是 YYYY-MM-DD") from exc
    try:
        summary_name = _holding_audit_summary_name(reader, run)
        artifact_name = reader.latest_artifact_name(run, "holding_audit")
        if artifact_name is None:  # defensive: checked by _holding_audit_summary_name
            raise KeyError("holding_audit")
        artifact_path = reader.artifact_path(run, artifact_name)
        summary = reader.get_detail(run, summary_name)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "该运行未启用查看仓单模块") from exc

    summary_by_day = {
        day: {
            column: row[position] if position < len(row) else None
            for position, column in enumerate(summary["columns"])
        }
        for day, row in zip(summary["index"], summary["data"])
    }
    if normalized_day not in summary_by_day:
        raise HTTPException(404, "该信号日没有可审计的最高分位持仓")

    try:
        frame = pd.read_parquet(
            artifact_path,
            filters=[("signal_day", "=", normalized_day)],
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "仓单产物无法读取") from exc
    if frame.empty:
        raise HTTPException(404, "该信号日没有可审计的最高分位持仓")
    frame = frame.sort_values("rank_in_top_group", kind="stable")
    total = len(frame)
    offset = (page - 1) * page_size
    page_frame = reference_data.enrich(frame.iloc[offset : offset + page_size])
    rows = sanitize(page_frame.to_dict(orient="records"))
    first = frame.iloc[0]
    return {
        "signal_day": normalized_day,
        "entry_day": sanitize(first["entry_day"]),
        "exit_day": sanitize(first["exit_day"]),
        "top_group": f"G{run['n_quantiles']}",
        "summary": sanitize(summary_by_day[normalized_day]),
        "total": total,
        "page": page,
        "page_size": page_size,
        "reference_data": reference_data.metadata(),
        "rows": rows,
    }


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
