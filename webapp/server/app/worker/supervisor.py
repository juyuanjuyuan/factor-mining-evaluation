"""API-side supervisor for one isolated, restartable evaluation process."""

from __future__ import annotations

import multiprocessing as mp
import threading
from queue import Empty
from typing import Any

from ..config import Settings
from ..db import Database
from ..sanitize import sanitize
from ..services.funnel_service import (
    gate_result,
    make_run,
    next_stage_index,
    normalize_funnel_stages,
)
from ..services.gate_service import evaluate_metric_gate
from .executor import worker_main


class WorkerSupervisor:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self._context = mp.get_context("spawn")
        self._commands: Any = None
        self._events: Any = None
        self._process: mp.Process | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._current_run_id: str | None = None
        self._ready = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._spawn_worker()
        self._thread = threading.Thread(
            target=self._loop, name="factor-worker-supervisor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._commands is not None:
            try:
                self._commands.put_nowait(None)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        self._terminate_worker()

    def _spawn_worker(self) -> None:
        self._commands = self._context.Queue()
        self._events = self._context.Queue()
        self._process = self._context.Process(
            target=worker_main,
            args=(self._commands, self._events, str(self.settings.data_dir)),
            name="factor-evaluation-worker",
            daemon=True,
        )
        self._process.start()
        self._ready = False

    def _terminate_worker(self) -> None:
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
        self._process = None
        self._ready = False

    def force_cancel_job(self, job_id: str) -> None:
        current = self.db.get_run(self._current_run_id) if self._current_run_id else None
        if current and current["job_id"] == job_id and current["status"] == "running":
            run_id = current["id"]
            self._terminate_worker()
            self.db.cancel_run(run_id)
            self._current_run_id = None
            if not self._stop.is_set():
                self._spawn_worker()

    def _loop(self) -> None:
        while not self._stop.wait(0.25):
            self._detect_crash()
            self._drain_events()
            if self._ready and self._current_run_id is None:
                run = self.db.claim_next_run()
                if run:
                    self._current_run_id = run["id"]
                    self._commands.put(run)

    def _detect_crash(self) -> None:
        if self._process is None or self._process.is_alive():
            return
        if self._current_run_id:
            self.db.fail_run(
                self._current_run_id,
                f"评价 worker 异常退出（exitcode={self._process.exitcode}）",
            )
            self._current_run_id = None
        if not self._stop.is_set():
            self._spawn_worker()

    def _drain_events(self) -> None:
        while self._events is not None:
            try:
                event = self._events.get_nowait()
            except Empty:
                return
            event_type = event.get("type")
            if event_type == "ready":
                self._ready = True
            elif event_type == "startup_failed":
                self._ready = False
            elif event_type == "succeeded":
                training = sanitize(event.get("model_training"))
                if training:
                    self.db.apply_model_training(
                        model_id=training["model_id"],
                        run_id=event["run_id"],
                        terms=training["result"]["terms"],
                        expression=training["expression"],
                        fit_result=training["result"],
                    )
                self._handle_success(event["run_id"], sanitize(event["metrics"]))
                self._current_run_id = None
            elif event_type == "failed":
                self.db.fail_run(event["run_id"], event["error"])
                self._current_run_id = None

    def _handle_success(self, run_id: str, metrics: dict[str, Any]) -> None:
        run = self.db.get_run(run_id)
        if not run:
            return
        job = self.db.get_job(run["job_id"], include_runs=False)
        if not job or job["kind"] != "funnel":
            gate = (job["params"] or {}).get("gate") if job else None
            verdict = evaluate_metric_gate(gate, metrics)
            if verdict is None:
                self.db.complete_run(run_id, metrics)
            else:
                outcome, details, explanation = verdict
                self.db.complete_run(
                    run_id,
                    metrics,
                    gate_outcome=outcome,
                    gate_value=details,
                    gate_explanation=explanation,
                )
            return
        significance = float(job["params"].get("significance_level", 0.05))
        stages = normalize_funnel_stages(job["params"].get("funnel_stages"))
        outcome, value, explanation = gate_result(run["stage"], metrics, significance)
        self.db.complete_run(
            run_id,
            metrics,
            gate_outcome=outcome,
            gate_value=value,
            gate_explanation=explanation,
        )
        next_index = next_stage_index(run["stage"], stages)
        if next_index is None:
            return

        factor = {
            "factor_name": run["factor_name"],
            "batch_id": run["batch_id"],
            "expression": run["expression"],
        }
        if outcome != "passed":
            skipped = [
                make_run(
                    job_id=run["job_id"],
                    factor=factor,
                    stage_name=stages[index]["name"],
                    methods=list(stages[index]["methods"]),
                    horizon=run["horizon"],
                    n_quantiles=run["n_quantiles"],
                    runs_dir=self.settings.runs_dir,
                    status="skipped",
                )
                for index in range(next_index, len(stages))
            ]
            self.db.add_runs(run["job_id"], skipped)
            return
        next_run = make_run(
            job_id=run["job_id"],
            factor=factor,
            stage_name=stages[next_index]["name"],
            methods=list(stages[next_index]["methods"]),
            horizon=run["horizon"],
            n_quantiles=run["n_quantiles"],
            runs_dir=self.settings.runs_dir,
        )
        self.db.add_runs(run["job_id"], [next_run])
