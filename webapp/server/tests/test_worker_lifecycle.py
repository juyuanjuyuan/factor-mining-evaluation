#!/usr/bin/env python3
"""Synthetic end-to-end worker success, crash isolation, and restart checks."""

from __future__ import annotations

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pandas as pd


SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_DIR.parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app.config import Settings
from app.db import Database
from app.services.funnel_service import make_run
from app.services.model_test_service import make_model_job, make_model_run
from app.services.run_reader import RunReader
from app.worker.supervisor import WorkerSupervisor


def wait_until(predicate, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise TimeoutError("condition did not become true")


def write_market_data(data_dir: Path) -> None:
    from engine import DEFAULT_FILES

    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-02", periods=80, freq="B")
    columns = [f"{index:06d}" for index in range(1, 9)]
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 1, (len(index), len(columns))), axis=0),
        index=index,
        columns=columns,
    ).clip(lower=5)
    frames = {
        "c": close,
        "o": close * (1 + rng.normal(0, 0.002, close.shape)),
        "h": close * 1.01,
        "l": close * 0.99,
        "vol": pd.DataFrame(rng.uniform(1e5, 1e6, close.shape), index=index, columns=columns),
        "amt": pd.DataFrame(rng.uniform(1e7, 1e8, close.shape), index=index, columns=columns),
        "vwap": close,
        "cap": pd.DataFrame(rng.uniform(1e9, 1e10, close.shape), index=index, columns=columns),
        "limit": pd.DataFrame(0.10, index=index, columns=columns),
        "st": pd.DataFrame(False, index=index, columns=columns),
    }
    frames["industry"] = pd.DataFrame(
        {
            "trade_date": np.repeat(index, len(columns)),
            "security_code": list(columns) * len(index),
            "industry_l1_code": np.tile(["A"] * 4 + ["B"] * 4, len(index)),
        }
    )
    data_dir.mkdir()
    for symbol, filename in DEFAULT_FILES.items():
        frames[symbol].to_parquet(data_dir / filename)


def add_job(
    db: Database,
    settings: Settings,
    title: str,
    *,
    expression: str = "rank_cs(delta(c, 2))",
    methods: list[str] | None = None,
    run_params: dict[str, str] | None = None,
) -> tuple[str, str]:
    job_id = uuid4().hex
    run = make_run(
        job_id=job_id,
        factor={
            "factor_name": f"worker_test_{title}",
            "batch_id": "synthetic",
            "expression": expression,
        },
        stage_name=None,
        methods=methods or ["rank_ic", "rank_icir"],
        horizon=1,
        n_quantiles=5,
        runs_dir=settings.runs_dir,
        run_params=run_params,
    )
    db.create_job(
        {
            "id": job_id,
            "kind": "evaluate",
            "title": title,
            "params": {},
            "created_at": datetime.now().astimezone().isoformat(),
        },
        [run],
    )
    return job_id, run["id"]


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        write_market_data(data_dir)
        settings = Settings(
            project_root=PROJECT_ROOT,
            data_dir=data_dir,
            registry_dir=PROJECT_ROOT / "factor_registry",
            state_dir=root / "state",
            frontend_dist=PROJECT_ROOT / "webapp" / "frontend" / "dist",
        )
        settings.runs_dir.mkdir(parents=True)
        db = Database(settings.db_path)
        db.initialize()
        supervisor = WorkerSupervisor(db, settings)
        supervisor.start()
        try:
            _, first_run = add_job(db, settings, "success")
            wait_until(lambda: db.get_run(first_run)["status"] == "succeeded")
            result = db.get_run(first_run)
            assert result and result["result"]["return_definition"]
            assert Path(result["output_dir"], "metrics.csv").is_file()

            _, industry_run = add_job(
                db,
                settings,
                "industry_neutralization",
                methods=["industry_neutralize", "rank_ic", "rank_icir"],
            )
            wait_until(lambda: db.get_run(industry_run)["status"] == "succeeded")
            industry_result = db.get_run(industry_run)
            assert industry_result
            industry_details = json.loads(
                industry_result["result"]["evaluation_details"]
            )
            assert "industry_neutralization" in industry_details

            _, joint_neutralization_run = add_job(
                db,
                settings,
                "industry_market_cap_neutralization",
                methods=["industry_market_cap_neutralize", "rank_ic", "rank_icir"],
            )
            wait_until(
                lambda: db.get_run(joint_neutralization_run)["status"] == "succeeded"
            )
            joint_neutralization_result = db.get_run(joint_neutralization_run)
            assert joint_neutralization_result
            joint_neutralization_details = json.loads(
                joint_neutralization_result["result"]["evaluation_details"]
            )
            assert "industry_market_cap_neutralization" in joint_neutralization_details

            # Model-test runs carry their train/test signal window in the
            # immutable run params. The worker must pass it through to the
            # evaluator, which keeps open-to-open labels inside that split.
            _, split_run = add_job(
                db,
                settings,
                "model_split",
                run_params={
                    "model_section": "training",
                    "signal_start": "2024-01-10",
                    "signal_end": "2024-03-15",
                },
            )
            wait_until(lambda: db.get_run(split_run)["status"] == "succeeded")
            split_result = db.get_run(split_run)
            assert split_result
            assert split_result["result"]["signal_start"] == "2024-01-10"
            assert split_result["result"]["signal_end"] == "2024-03-15"
            assert split_result["result"]["sample_start_day"] == "2024-01-10"
            assert split_result["result"]["sample_end_day"] == "2024-03-13"
            split_detail = RunReader().get_detail(split_result, "ic")
            assert split_detail["index"][0] == "2024-01-10"
            assert split_detail["index"][-1] == "2024-03-13"

            # A registered automatic trainer runs inside the preloaded worker,
            # atomically replaces the provisional model expression, and leaves
            # the holdout run with only the frozen fitted expression.
            model_id = uuid4().hex
            model = db.create_model_test(
                {
                    "id": model_id,
                    "model_name": "worker ridge model",
                    "terms": [
                        {
                            "batch_id": "synthetic",
                            "factor_name": "price",
                            "expression": "delta(c, 2)",
                            "weight": 1.0,
                        },
                        {
                            "batch_id": "synthetic",
                            "factor_name": "volume",
                            "expression": "delta(vol, 2)",
                            "weight": 1.0,
                        },
                    ],
                    "expression": (
                        "(1 * zscore_cs(winsorize_cs((delta(c, 2)), 0.01, 0.99))) + "
                        "(1 * zscore_cs(winsorize_cs((delta(vol, 2)), 0.01, 0.99)))"
                    ),
                    "train_start": "2024-01-10",
                    "train_end": "2024-03-15",
                    "test_start": "2024-03-18",
                    "test_end": "2024-04-19",
                    "horizon": 1,
                    "n_quantiles": 5,
                    "methods": ["rank_ic", "rank_icir"],
                    "training_method": "winsorized_zscore_ridge",
                    "training_params": {
                        "ridge_alpha": 1e-6,
                        "winsor_lower_quantile": 0.01,
                        "winsor_upper_quantile": 0.99,
                    },
                }
            )
            training_job = make_model_job(model, section="training")
            model_training_run = make_model_run(
                model,
                section="training",
                job_id=training_job["id"],
                settings=settings,
            )
            db.create_job(training_job, [model_training_run])
            assert db.link_model_run(model_id, model_training_run["id"], section="training")
            wait_until(lambda: db.get_run(model_training_run["id"])["status"] == "succeeded")
            fitted_model = db.get_model_test(model_id)
            fitted_run = db.get_run(model_training_run["id"])
            assert fitted_model and fitted_run
            assert fitted_model["fit_result"]["method"] == "winsorized_zscore_ridge"
            assert len(fitted_model["fit_result"]["terms"]) == 2
            assert fitted_model["expression"] == fitted_run["expression"]
            assert fitted_model["expression"] != model["expression"]
            assert "zscore_cs(winsorize_cs" in fitted_model["expression"]
            assert fitted_run["result"]["model_training_method"] == "winsorized_zscore_ridge"
            assert Path(fitted_run["output_dir"], "model_training.json").is_file()

            testing_job = make_model_job(fitted_model, section="testing")
            model_testing_run = make_model_run(
                fitted_model,
                section="testing",
                job_id=testing_job["id"],
                settings=settings,
            )
            assert model_testing_run["expression"] == fitted_model["expression"]
            assert "model_training" not in model_testing_run["run_params"]
            db.create_job(testing_job, [model_testing_run])
            assert db.link_model_run(model_id, model_testing_run["id"], section="testing")
            wait_until(lambda: db.get_run(model_testing_run["id"])["status"] == "succeeded")
            assert db.get_model_test(model_id)["expression"] == fitted_model["expression"]

            _, prefix_run = add_job(
                db,
                settings,
                "prefix",
                expression="ts_mean(c, 3)",
                methods=["prefix_truncation_consistency"],
            )
            wait_until(lambda: db.get_run(prefix_run)["status"] == "succeeded")
            prefix_result = db.get_run(prefix_run)
            assert prefix_result["result"]["prefix_truncation_passed"]
            details = json.loads(prefix_result["result"]["evaluation_details"])
            detail_path = details["prefix_truncation_consistency"]
            assert Path(prefix_result["output_dir"], detail_path).is_file()

            _, crash_run = add_job(db, settings, "crash")
            wait_until(lambda: supervisor._current_run_id == crash_run)
            assert supervisor._process
            supervisor._process.kill()
            wait_until(lambda: db.get_run(crash_run)["status"] == "failed")

            _, recovery_run = add_job(db, settings, "recovery")
            wait_until(lambda: db.get_run(recovery_run)["status"] == "succeeded")
        finally:
            supervisor.stop()

    print("webapp worker lifecycle passed")


if __name__ == "__main__":
    main()
