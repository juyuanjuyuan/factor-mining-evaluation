#!/usr/bin/env python3
"""Web API and detached-process contracts for GP factor mining."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient


SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_DIR.parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app.config import Settings
from app.routers import genetic_campaigns
from app.routers.dependencies import get_genetic_supervisor
from app.worker.genetic_supervisor import GeneticMiningSupervisor
from genetic_mining.tree import ExpressionTree


class FakeProcess:
    next_pid = 70000

    def __init__(self):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode or 0


class FakeProcessFactory:
    def __init__(self):
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, command: list[str], **kwargs: Any) -> FakeProcess:
        process = FakeProcess()
        self.calls.append((command, kwargs))
        self.processes.append(process)
        return process


class FakeFactorLibrary:
    def __init__(self):
        self.requests: list[dict[str, Any]] = []

    def submit_gp_candidate(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(dict(request))
        return {
            "status": "admitted",
            "factor_name": request["factor_name"],
            "correlation_checked": True,
            "correlation_passed": True,
            "formal_batch_id": "webapp_factor_library",
        }


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        registry_dir = root / "factor_registry"
        data_dir.mkdir()
        registry_dir.mkdir()
        days = pd.date_range("2024-01-02", periods=140, freq="B")
        pd.DataFrame(index=days, columns=["000001"], dtype=float).to_parquet(
            data_dir / "close_df.pq"
        )
        settings = Settings(
            project_root=PROJECT_ROOT,
            data_dir=data_dir,
            registry_dir=registry_dir,
            state_dir=root / "outputs" / "webapp",
            frontend_dist=PROJECT_ROOT / "webapp" / "frontend" / "dist",
        )
        factory = FakeProcessFactory()
        factor_library = FakeFactorLibrary()
        supervisor = GeneticMiningSupervisor(
            settings,
            process_factory=factory,
            registry=factor_library,
        )
        app = FastAPI()
        app.include_router(genetic_campaigns.router, prefix="/api")
        app.dependency_overrides[get_genetic_supervisor] = lambda: supervisor
        payload = {
            "campaign": "web_gp_contract",
            "train_start": days[0].date().isoformat(),
            "train_end": days[69].date().isoformat(),
            "test_start": days[70].date().isoformat(),
            "test_end": days[-1].date().isoformat(),
            "horizon": 1,
            "n_quantiles": 5,
            "preprocess_mode": "market_cap_industry",
            "population_size": 4,
            "generations": 1,
            "hall_of_fame": 4,
            "components": 1,
            "tournament_size": 2,
            "n_jobs": 1,
            "compute_backend": "cpu",
            "seed": 7,
            "continuous": False,
            "pause_seconds": 0,
        }
        with TestClient(app) as client:
            backends = client.get("/api/genetic-campaigns/backends")
            assert backends.status_code == 200, backends.text
            backend_names = {item["name"] for item in backends.json()}
            assert backend_names == {"cpu", "mps"}
            created = client.post("/api/genetic-campaigns", json=payload)
            assert created.status_code == 201, created.text
            record = created.json()
            assert record["status"] == "running"
            assert record["cycle_baseline"] == 0
            assert record["cycle_target"] == 1
            assert record["config"]["train_end"] == payload["train_end"]
            assert record["output_dir"].endswith("gp_factor_mining/web_gp_contract")
            command = factory.calls[0][0]
            assert command[command.index("--campaign") + 1] == "web_gp_contract"
            assert command[command.index("--compute-backend") + 1] == "cpu"
            assert command[command.index("--preprocess-mode") + 1] == "market_cap_industry"
            assert "--library-file" not in command
            assert "--correlation-state-dir" not in command
            assert "--no-admit" not in command
            assert "--forever" not in command
            assert factory.calls[0][1]["start_new_session"] is True

            campaign_root = settings.genetic_mining_dir / "web_gp_contract"
            cycle_root = campaign_root / "cycles" / "cycle_000001"
            cycle_root.mkdir(parents=True)
            trees = [ExpressionTree("terminal", "c"), ExpressionTree("terminal", "o")]
            (campaign_root / "active_cycle.json").write_text(
                '{"cycle": 1, "status": "running"}\n',
                encoding="utf-8",
            )
            (cycle_root / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "cycle": 1,
                        "stage": "evolution",
                        "next_generation": 0,
                        "population": [tree.to_dict() for tree in trees],
                    }
                ),
                encoding="utf-8",
            )
            (campaign_root / "fitness_cache.jsonl").write_text(
                json.dumps({"expression": "c", "error": ""})
                + "\n",
                encoding="utf-8",
            )
            candidate_name = "gp_web_contract_001"
            candidate_root = cycle_root / "candidates" / candidate_name
            candidate_root.mkdir(parents=True)
            (candidate_root / "factor_library_submission_request.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "factor_name": candidate_name,
                        "expression": "c",
                        "project": "遗传规划",
                    }
                ),
                encoding="utf-8",
            )
            (cycle_root / "cycle_summary.json").write_text(
                json.dumps(
                    {
                        "cycle": 1,
                        "status": "completed",
                        "test_passed_count": 1,
                        "failed_count": 0,
                        "candidates": [
                            {
                                "factor_name": candidate_name,
                                "expression": "c",
                                "test_overall_passed": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            progress = client.get("/api/genetic-campaigns/web_gp_contract").json()
            assert progress["current_generation_completed"] == 1
            assert progress["current_generation_total"] == 2
            assert progress["current_generation_progress"] == 0.5
            assert factor_library.requests == [
                {
                    "schema_version": 1,
                    "factor_name": candidate_name,
                    "expression": "c",
                    "project": "遗传规划",
                }
            ]
            assert progress["factor_library_admitted_count"] == 1
            assert progress["latest_cycle"]["candidates"][0][
                "factor_library_submission"
            ]["status"] == "admitted"

            listed = client.get("/api/genetic-campaigns")
            assert listed.status_code == 200
            assert listed.json()[0]["campaign"] == "web_gp_contract"
            conflict = client.post(
                "/api/genetic-campaigns",
                json={**payload, "campaign": "second_campaign"},
            )
            assert conflict.status_code == 409
            assert "正在运行" in conflict.json()["detail"]

            stopped = client.post("/api/genetic-campaigns/web_gp_contract/stop")
            assert stopped.status_code == 200
            assert stopped.json()["status"] == "stopped"
            restarted = client.post("/api/genetic-campaigns/web_gp_contract/start")
            assert restarted.status_code == 200
            assert restarted.json()["status"] == "running"
            factory.processes[-1].returncode = 0
            completed = client.get("/api/genetic-campaigns/web_gp_contract")
            assert completed.status_code == 200
            assert completed.json()["status"] == "succeeded"

            invalid = client.post(
                "/api/genetic-campaigns",
                json={
                    **payload,
                    "campaign": "invalid_split",
                    "train_end": payload["test_start"],
                },
            )
            assert invalid.status_code == 422
            assert "不能重叠" in invalid.text

        metadata = settings.genetic_mining_dir / "web_gp_contract" / "web_campaign.json"
        assert metadata.is_file()

    print("webapp genetic mining contracts passed")


if __name__ == "__main__":
    main()
