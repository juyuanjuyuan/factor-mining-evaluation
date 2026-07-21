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
        supervisor = GeneticMiningSupervisor(settings, process_factory=factory)
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
            "preprocess_mode": "none",
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
            "admit": False,
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
            assert "--no-admit" in command
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
            progress = client.get("/api/genetic-campaigns/web_gp_contract").json()
            assert progress["current_generation_completed"] == 1
            assert progress["current_generation_total"] == 2
            assert progress["current_generation_progress"] == 0.5

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
