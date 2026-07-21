"""Independent process supervisor for resumable genetic-mining campaigns."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from genetic_mining.tree import ExpressionTree

from ..config import Settings


ProcessFactory = Callable[..., subprocess.Popen[Any]]
_CAMPAIGN_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _tail(path: Path, *, lines: int = 20, characters: int = 6000) -> str:
    if not path.is_file():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(content.splitlines()[-lines:])[-characters:]


def _generation_progress(
    root: Path,
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, int | float | None]:
    """Derive in-generation progress from the durable expression cache."""

    if not checkpoint or checkpoint.get("stage") != "evolution":
        return {
            "current_generation_completed": None,
            "current_generation_total": None,
            "current_generation_failed": None,
            "current_generation_progress": None,
        }
    expressions: set[str] = set()
    for payload in checkpoint.get("population", []):
        try:
            expressions.add(ExpressionTree.from_dict(payload).to_expression())
        except (KeyError, TypeError, ValueError):
            continue
    cached: dict[str, bool] = {}
    cache_path = root / "fitness_cache.jsonl"
    if cache_path.is_file():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    expression = payload.get("expression")
                    if isinstance(expression, str):
                        cached[expression] = bool(payload.get("error"))
        except (OSError, ValueError):
            pass
    completed = sum(expression in cached for expression in expressions)
    failed = sum(cached.get(expression, False) for expression in expressions)
    total = len(expressions)
    return {
        "current_generation_completed": completed,
        "current_generation_total": total,
        "current_generation_failed": failed,
        "current_generation_progress": (completed / total) if total else None,
    }


class GeneticMiningSupervisor:
    """Launch GP outside the evaluation queue and expose file-backed progress."""

    def __init__(
        self,
        settings: Settings,
        *,
        process_factory: ProcessFactory = subprocess.Popen,
        poll_interval: float = 1.0,
    ):
        self.settings = settings
        self.process_factory = process_factory
        self.poll_interval = poll_interval
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.settings.genetic_mining_dir.mkdir(parents=True, exist_ok=True)
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._monitor,
            name="genetic-mining-supervisor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring only; detached mining processes keep running."""

        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _monitor(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                self.list_campaigns()
            except Exception:
                # A damaged status file must not take down the API process.
                continue

    def _root(self, campaign: str) -> Path:
        if not _CAMPAIGN_RE.fullmatch(campaign):
            raise ValueError("无效的 campaign 名称")
        return self.settings.genetic_mining_dir / campaign

    def _metadata_path(self, campaign: str) -> Path:
        return self._root(campaign) / "web_campaign.json"

    def _load_metadata(self, campaign: str) -> dict[str, Any]:
        payload = _read_json(self._metadata_path(campaign))
        if payload is None:
            raise KeyError(campaign)
        return payload

    def _write_metadata(self, campaign: str, payload: Mapping[str, Any]) -> None:
        _atomic_json(self._metadata_path(campaign), payload)

    def _build_command(self, config: Mapping[str, Any]) -> list[str]:
        command = [
            sys.executable,
            str(self.settings.project_root / "scripts" / "run_gp_factor_mining.py"),
            "--campaign",
            str(config["campaign"]),
            "--train-start",
            str(config["train_start"]),
            "--train-end",
            str(config["train_end"]),
            "--test-start",
            str(config["test_start"]),
            "--test-end",
            str(config["test_end"]),
            "--data-dir",
            str(self.settings.data_dir),
            "--output-dir",
            str(self.settings.genetic_mining_dir),
            "--library-file",
            str(self.settings.submitted_registry_path),
            "--correlation-state-dir",
            str(self.settings.state_dir),
            "--horizon",
            str(config["horizon"]),
            "--quantiles",
            str(config["n_quantiles"]),
            "--preprocess-mode",
            str(config["preprocess_mode"]),
            "--population-size",
            str(config["population_size"]),
            "--generations",
            str(config["generations"]),
            "--hall-of-fame",
            str(config["hall_of_fame"]),
            "--components",
            str(config["components"]),
            "--tournament-size",
            str(config["tournament_size"]),
            "--n-jobs",
            str(config["n_jobs"]),
            "--compute-backend",
            str(config.get("compute_backend", "cpu")),
            "--seed",
            str(config["seed"]),
            "--project",
            "遗传规划",
        ]
        if not config.get("admit", True):
            command.append("--no-admit")
        if config.get("continuous"):
            command.extend(["--forever", "--pause-seconds", str(config["pause_seconds"])])
            if config.get("max_cycles") is not None:
                command.extend(["--max-cycles", str(config["max_cycles"])])
        return command

    def _active_campaign(self, *, excluding: str | None = None) -> str | None:
        for row in self.list_campaigns():
            if row["campaign"] != excluding and row["status"] == "running":
                return str(row["campaign"])
        return None

    def create_campaign(self, config: Mapping[str, Any]) -> dict[str, Any]:
        campaign = str(config["campaign"])
        with self._lock:
            if self._metadata_path(campaign).exists():
                raise FileExistsError(f"Campaign {campaign} 已存在；请换名称或继续原任务")
            active = self._active_campaign()
            if active:
                raise RuntimeError(f"已有遗传挖掘任务正在运行：{active}")
            root = self._root(campaign)
            root.mkdir(parents=True, exist_ok=False)
            timestamp = _now()
            metadata = {
                "campaign": campaign,
                "status": "created",
                "config": dict(config),
                "pid": None,
                "created_at": timestamp,
                "started_at": None,
                "updated_at": timestamp,
                "finished_at": None,
                "error": None,
            }
            self._write_metadata(campaign, metadata)
            return self._launch(campaign, metadata)

    def start_campaign(self, campaign: str) -> dict[str, Any]:
        with self._lock:
            metadata = self._refresh(campaign)
            if metadata["status"] == "running":
                return metadata
            active = self._active_campaign(excluding=campaign)
            if active:
                raise RuntimeError(f"已有遗传挖掘任务正在运行：{active}")
            return self._launch(campaign, self._load_metadata(campaign))

    def _launch(
        self,
        campaign: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        command = self._build_command(metadata["config"])
        root = self._root(campaign)
        stdout_path = root / "web_stdout.log"
        stderr_path = root / "web_stderr.log"
        environment = dict(os.environ)
        environment.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
        if metadata["config"].get("compute_backend") == "mps":
            # Never conceal an unsupported tensor operation by running it on CPU.
            environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        existing_cycles = len(list((root / "cycles").glob("cycle_*/cycle_summary.json")))
        if metadata["config"].get("continuous"):
            launch_limit = metadata["config"].get("max_cycles")
            cycle_target = (
                existing_cycles + int(launch_limit)
                if launch_limit is not None
                else None
            )
        else:
            cycle_target = existing_cycles + 1
        try:
            with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                process = self.process_factory(
                    command,
                    cwd=str(self.settings.project_root),
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
        except Exception as exc:
            failed = {
                **metadata,
                "status": "failed",
                "pid": None,
                "updated_at": _now(),
                "finished_at": _now(),
                "error": f"启动失败：{type(exc).__name__}: {exc}",
            }
            self._write_metadata(campaign, failed)
            raise RuntimeError(failed["error"]) from exc
        self._processes[campaign] = process
        running = {
            **metadata,
            "status": "running",
            "pid": int(process.pid),
            "command": command,
            "cycle_baseline": existing_cycles,
            "cycle_target": cycle_target,
            "started_at": _now(),
            "updated_at": _now(),
            "finished_at": None,
            "error": None,
        }
        self._write_metadata(campaign, running)
        return self._enrich(running)

    def _pid_alive(self, pid: int | None) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def _process_state(self, campaign: str, metadata: Mapping[str, Any]) -> tuple[bool, int | None]:
        process = self._processes.get(campaign)
        if process is not None:
            return process.poll() is None, process.poll()
        return self._pid_alive(metadata.get("pid")), None

    def _refresh(self, campaign: str) -> dict[str, Any]:
        metadata = self._load_metadata(campaign)
        if metadata.get("status") != "running":
            return self._enrich(metadata)
        alive, exit_code = self._process_state(campaign, metadata)
        if alive:
            return self._enrich(metadata)
        active = _read_json(self._root(campaign) / "active_cycle.json") or {}
        summaries = list((self._root(campaign) / "cycles").glob("cycle_*/cycle_summary.json"))
        completed = len(summaries)
        cycle_target = metadata.get("cycle_target")
        expected_completion = exit_code == 0 or (
            active.get("status") == "completed"
            and cycle_target is not None
            and completed >= int(cycle_target)
        )
        finished = {
            **metadata,
            "status": "succeeded" if expected_completion else "failed",
            "pid": None,
            "updated_at": _now(),
            "finished_at": _now(),
            "error": None if expected_completion else (
                f"遗传挖掘进程已退出{f'（exitcode={exit_code}）' if exit_code is not None else ''}"
            ),
        }
        self._processes.pop(campaign, None)
        self._write_metadata(campaign, finished)
        return self._enrich(finished)

    def _enrich(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        campaign = str(metadata["campaign"])
        root = self._root(campaign)
        active = _read_json(root / "active_cycle.json")
        checkpoint = None
        if active and active.get("cycle") is not None:
            checkpoint = _read_json(
                root
                / "cycles"
                / f"cycle_{int(active['cycle']):06d}"
                / "checkpoint.json"
            )
        summaries = [
            payload
            for path in sorted((root / "cycles").glob("cycle_*/cycle_summary.json"))
            if (payload := _read_json(path)) is not None
        ]
        latest = summaries[-1] if summaries else None
        last_error = _read_json(root / "last_error.json")
        stage = active.get("status") if active else None
        next_generation = int(checkpoint.get("next_generation", 0)) if checkpoint else 0
        configured_generations = int((metadata.get("config") or {}).get("generations", 0))
        current_generation = None
        if checkpoint and stage == "running":
            current_generation = min(next_generation + 1, configured_generations)
        elif checkpoint and stage == "testing":
            current_generation = configured_generations
        progress = _generation_progress(root, checkpoint)
        return {
            **metadata,
            "process_alive": metadata.get("status") == "running",
            "output_dir": str(root),
            "current_cycle": active.get("cycle") if active else None,
            "current_stage": stage,
            "current_generation": current_generation,
            **progress,
            "completed_cycles": len(summaries),
            "test_passed_count": sum(int(item.get("test_passed_count", 0)) for item in summaries),
            "admitted_count": sum(int(item.get("admitted_count", 0)) for item in summaries),
            "failed_candidate_count": sum(int(item.get("failed_count", 0)) for item in summaries),
            "latest_cycle": latest,
            "last_error": last_error,
            "stdout_tail": _tail(root / "web_stdout.log"),
            "stderr_tail": _tail(root / "web_stderr.log"),
        }

    def list_campaigns(self) -> list[dict[str, Any]]:
        self.settings.genetic_mining_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = []
            for path in self.settings.genetic_mining_dir.glob("*/web_campaign.json"):
                try:
                    rows.append(self._refresh(path.parent.name))
                except (KeyError, ValueError):
                    continue
            return sorted(rows, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def get_campaign(self, campaign: str) -> dict[str, Any]:
        with self._lock:
            return self._refresh(campaign)

    def stop_campaign(self, campaign: str) -> dict[str, Any]:
        with self._lock:
            metadata = self._load_metadata(campaign)
            alive, _ = self._process_state(campaign, metadata)
            if metadata.get("status") != "running" or not alive:
                return self._refresh(campaign)
            process = self._processes.get(campaign)
            try:
                if process is not None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                else:
                    os.killpg(int(metadata["pid"]), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            self._processes.pop(campaign, None)
            stopped = {
                **metadata,
                "status": "stopped",
                "pid": None,
                "updated_at": _now(),
                "finished_at": _now(),
                "error": "用户从 Webapp 停止任务",
            }
            self._write_metadata(campaign, stopped)
            return self._enrich(stopped)
