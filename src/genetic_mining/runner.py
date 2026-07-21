"""Resumable train-mine/test-gate/admit orchestration for continuous GP search."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import fcntl
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from engine import load_market_data
from evaluation_standards import evaluate_factor_standards
from factor_correlation import CORRELATION_THRESHOLD

from .admission import admit_factor_to_library
from .evolution import (
    EvolutionConfig,
    ScoredTree,
    evaluate_population,
    evolve_population,
    initial_population,
    select_diverse_components,
    scored_sort_key,
    update_hall_of_fame,
)
from .fitness import (
    FitnessResult,
    fitness_contract,
    prepare_fitness_context,
)
from .tree import ExpressionTree


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "_", value.strip().lower()).strip("_-")
    if not normalized:
        raise ValueError("campaign must contain lowercase letters, digits, _ or -")
    return normalized


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class MiningCampaignConfig:
    campaign: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    data_dir: Path
    output_dir: Path
    library_file: Path
    correlation_state_dir: Path
    evolution: EvolutionConfig = EvolutionConfig()
    horizon: int = 1
    n_quantiles: int = 10
    preprocess_mode: str = "paper_local"
    minimum_ic_days: int = 60
    significance_level: float = 0.05
    minimum_ic_mean: float = 0.0
    minimum_rolling_sharpe_60_median: float = 1.0
    minimum_annualized_return: float = 0.30
    correlation_threshold: float = CORRELATION_THRESHOLD
    seed: int = 20190610
    admit: bool = True
    project: str = "遗传规划"

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign", _slug(self.campaign))
        for name in ("data_dir", "output_dir", "library_file", "correlation_state_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        train_start = np.datetime64(self.train_start)
        train_end = np.datetime64(self.train_end)
        test_start = np.datetime64(self.test_start)
        test_end = np.datetime64(self.test_end)
        if train_start > train_end or test_start > test_end:
            raise ValueError("train/test start must not be after its end")
        if train_end >= test_start:
            raise ValueError("training and test windows must be disjoint and ordered")
        if self.horizon < 1 or self.n_quantiles < 3:
            raise ValueError("horizon must be positive and n_quantiles at least three")
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be between zero and one")
        if not 0 < self.correlation_threshold < 1:
            raise ValueError("correlation_threshold must be between zero and one")

    @property
    def root(self) -> Path:
        return self.output_dir / self.campaign

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["library_file"] = str(self.library_file)
        payload["correlation_state_dir"] = str(self.correlation_state_dir)
        payload["evolution"] = self.evolution.as_dict()
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class GeneticMiningRunner:
    def __init__(self, config: MiningCampaignConfig):
        self.config = config
        self.config.root.mkdir(parents=True, exist_ok=True)
        self._lock_handle = (self.config.root / ".campaign.lock").open("a+")
        try:
            fcntl.flock(
                self._lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            self._lock_handle.close()
            raise RuntimeError(
                f"Campaign {self.config.campaign!r} is already running"
            ) from exc
        self.campaign_path = self.config.root / "campaign.json"
        self.active_path = self.config.root / "active_cycle.json"
        self.cache_path = self.config.root / "fitness_cache.jsonl"
        self._initialize_campaign()
        self.market_data = self._load_market_data()
        self.fitness_context = prepare_fitness_context(
            self.market_data,
            train_start=config.train_start,
            train_end=config.train_end,
            horizon=config.horizon,
            preprocess_mode=config.preprocess_mode,
            minimum_ic_days=config.minimum_ic_days,
        )
        self.fitness_cache = self._load_fitness_cache()

    def close(self) -> None:
        if not self._lock_handle.closed:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _initialize_campaign(self) -> None:
        if self.campaign_path.is_file():
            existing = _read_json(self.campaign_path)
            if existing.get("config_fingerprint") != self.config.fingerprint:
                raise ValueError(
                    "Campaign configuration changed; use a new campaign name to keep "
                    "training/test and cached-fitness provenance immutable"
                )
            return
        _atomic_json(
            self.campaign_path,
            {
                "campaign": self.config.campaign,
                "created_at": _now(),
                "config_fingerprint": self.config.fingerprint,
                "config": self.config.as_dict(),
                "next_cycle": 1,
            },
        )

    def _load_market_data(self):
        terminal_expression = " + ".join(f"({item})" for item in self.config.evolution.terminals)
        return load_market_data(
            self.config.data_dir,
            terminal_expression,
            extra_symbols={"cap", "amt", "limit", "st"},
        )

    def _load_fitness_cache(self) -> dict[str, FitnessResult]:
        cache: dict[str, FitnessResult] = {}
        if not self.cache_path.is_file():
            return cache
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    result = FitnessResult.from_dict(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                cache[result.expression] = result
        return cache

    def _append_fitness(self, result: FitnessResult) -> None:
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.as_dict(), ensure_ascii=False, allow_nan=False))
            handle.write("\n")
            handle.flush()

    def _archive_fitness_cache(self, cycle: int) -> None:
        """Keep resume cache per cycle so continuous mode has bounded RAM."""

        if not self.cache_path.is_file():
            self.fitness_cache.clear()
            return
        archived = self._cycle_dir(cycle) / "fitness_cache.jsonl"
        archived.parent.mkdir(parents=True, exist_ok=True)
        if archived.exists():
            with archived.open("a", encoding="utf-8") as destination, self.cache_path.open(
                "r", encoding="utf-8"
            ) as source:
                shutil.copyfileobj(source, destination)
            self.cache_path.unlink()
        else:
            os.replace(self.cache_path, archived)
        self.fitness_cache.clear()

    def _campaign_record(self) -> dict[str, Any]:
        return _read_json(self.campaign_path)

    def _cycle_number(self) -> int:
        if self.active_path.is_file():
            active = _read_json(self.active_path)
            if active.get("status") in {"running", "testing"}:
                return int(active["cycle"])
        return int(self._campaign_record().get("next_cycle", 1))

    def _cycle_dir(self, cycle: int) -> Path:
        return self.config.root / "cycles" / f"cycle_{cycle:06d}"

    def _checkpoint_path(self, cycle: int) -> Path:
        return self._cycle_dir(cycle) / "checkpoint.json"

    def _new_cycle(self, cycle: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.config.seed + cycle - 1)
        population = initial_population(rng, self.config.evolution)
        checkpoint = {
            "cycle": cycle,
            "stage": "evolution",
            "next_generation": 0,
            "population": [tree.to_dict() for tree in population],
            "hall_of_fame": [],
            "rng_state": rng.bit_generator.state,
            "started_at": _now(),
        }
        _atomic_json(self._checkpoint_path(cycle), checkpoint)
        _atomic_json(
            self.active_path,
            {"cycle": cycle, "status": "running", "updated_at": _now()},
        )
        return checkpoint

    def _load_or_create_checkpoint(self, cycle: int) -> dict[str, Any]:
        path = self._checkpoint_path(cycle)
        return _read_json(path) if path.is_file() else self._new_cycle(cycle)

    def _generation_summary(
        self,
        generation: int,
        scored: tuple[ScoredTree, ...],
    ) -> dict[str, Any]:
        valid = [item for item in scored if np.isfinite(item.fitness.selection_score)]
        ranked = sorted(scored, key=scored_sort_key)
        scores = np.asarray(
            [item.fitness.selection_score for item in valid], dtype=float
        )
        return {
            "generation": generation + 1,
            "population_size": len(scored),
            "valid_programs": len(valid),
            "invalid_programs": len(scored) - len(valid),
            "mean_adjusted_fitness": float(scores.mean()) if len(scores) else None,
            "best": ranked[0].as_dict() if ranked else None,
            "mean_node_count": float(
                np.mean([item.tree.node_count for item in scored])
            ),
            "completed_at": _now(),
        }

    def _run_evolution(
        self,
        cycle: int,
        checkpoint: dict[str, Any],
    ) -> tuple[tuple[ScoredTree, ...], list[dict[str, Any]]]:
        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint["rng_state"]
        population = tuple(
            ExpressionTree.from_dict(item) for item in checkpoint["population"]
        )
        hall = {
            item["fitness"]["expression"]: ScoredTree.from_dict(item)
            for item in checkpoint.get("hall_of_fame", [])
        }
        start_generation = int(checkpoint["next_generation"])
        for generation in range(start_generation, self.config.evolution.generations):
            scored = evaluate_population(
                population,
                self.fitness_context,
                self.config.evolution,
                self.fitness_cache,
                on_result=self._append_fitness,
            )
            update_hall_of_fame(
                hall,
                scored,
                limit=self.config.evolution.hall_of_fame,
            )
            generation_summary = self._generation_summary(generation, scored)
            _atomic_json(
                self._cycle_dir(cycle)
                / "generations"
                / f"generation_{generation + 1:03d}.json",
                generation_summary,
            )
            if generation + 1 < self.config.evolution.generations:
                population = evolve_population(scored, rng, self.config.evolution)
                checkpoint = {
                    **checkpoint,
                    "next_generation": generation + 1,
                    "population": [tree.to_dict() for tree in population],
                    "hall_of_fame": [
                        item.as_dict()
                        for item in sorted(hall.values(), key=scored_sort_key)
                    ],
                    "rng_state": rng.bit_generator.state,
                    "updated_at": _now(),
                }
                _atomic_json(self._checkpoint_path(cycle), checkpoint)

        components, diversity_audit = select_diverse_components(
            hall,
            self.fitness_context,
            self.config.evolution,
        )
        checkpoint = {
            **checkpoint,
            "stage": "testing",
            "next_generation": self.config.evolution.generations,
            "components": [item.as_dict() for item in components],
            "diversity_audit": diversity_audit,
            "hall_of_fame": [
                item.as_dict() for item in sorted(hall.values(), key=scored_sort_key)
            ],
            "updated_at": _now(),
        }
        _atomic_json(self._checkpoint_path(cycle), checkpoint)
        _atomic_json(
            self.active_path,
            {"cycle": cycle, "status": "testing", "updated_at": _now()},
        )
        return components, diversity_audit

    def _candidate_name(self, cycle: int, rank: int, expression: str) -> str:
        digest = hashlib.sha256(expression.encode("utf-8")).hexdigest()[:8]
        return f"gp_{self.config.campaign}_{cycle:06d}_{rank:03d}_{digest}"

    def _evaluate_candidates(
        self,
        cycle: int,
        components: tuple[ScoredTree, ...],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for rank, component in enumerate(components, start=1):
            name = self._candidate_name(
                cycle,
                rank,
                component.fitness.expression,
            )
            candidate_dir = self._cycle_dir(cycle) / "candidates" / name
            result_path = candidate_dir / "candidate_result.json"
            if result_path.is_file():
                records.append(_read_json(result_path))
                continue
            try:
                standards = evaluate_factor_standards(
                    factor_name=name,
                    expression=component.fitness.expression,
                    data_dir=self.config.data_dir,
                    output_dir=candidate_dir / "test_evaluation",
                    signal_start=self.config.test_start,
                    signal_end=self.config.test_end,
                    standards="all",
                    horizon=self.config.horizon,
                    n_quantiles=self.config.n_quantiles,
                    preloaded_data=self.market_data,
                    significance_level=self.config.significance_level,
                    minimum_ic_mean=self.config.minimum_ic_mean,
                    minimum_rolling_sharpe_60_median=(
                        self.config.minimum_rolling_sharpe_60_median
                    ),
                    minimum_annualized_return=self.config.minimum_annualized_return,
                )
                admission: dict[str, Any] | None = None
                if standards["overall_passed"] and self.config.admit:
                    admission = admit_factor_to_library(
                        factor_name=name,
                        expression=component.fitness.expression,
                        library_file=self.config.library_file,
                        data_dir=self.config.data_dir,
                        state_dir=self.config.correlation_state_dir,
                        project=self.config.project,
                        source_batch_id=f"gp_{self.config.campaign}",
                        source_factor_name=name,
                        notes=(
                            f"GP cycle {cycle}; train {self.config.train_start}.."
                            f"{self.config.train_end}; test {self.config.test_start}.."
                            f"{self.config.test_end}; frozen expression"
                        ),
                        correlation_threshold=self.config.correlation_threshold,
                    ).as_dict()
                record = {
                    "factor_name": name,
                    "expression": component.fitness.expression,
                    "training_fitness": component.fitness.as_dict(),
                    "test_overall_passed": standards["overall_passed"],
                    "standards_summary_path": standards["summary_path"],
                    "standard_gates": {
                        key: value["gate"]
                        for key, value in standards["standards"].items()
                    },
                    "admission": admission,
                    "admitted": bool(
                        admission
                        and (admission["admitted"] or admission["already_present"])
                    ),
                    "status": "completed",
                    "completed_at": _now(),
                }
            except Exception as exc:
                record = {
                    "factor_name": name,
                    "expression": component.fitness.expression,
                    "training_fitness": component.fitness.as_dict(),
                    "test_overall_passed": False,
                    "admission": None,
                    "admitted": False,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_at": _now(),
                }
            _atomic_json(result_path, record)
            records.append(record)
        return records

    def run_cycle(self) -> dict[str, Any]:
        cycle = self._cycle_number()
        checkpoint = self._load_or_create_checkpoint(cycle)
        if checkpoint.get("stage") == "testing":
            components = tuple(
                ScoredTree.from_dict(item) for item in checkpoint.get("components", [])
            )
            diversity_audit = list(checkpoint.get("diversity_audit", []))
        else:
            components, diversity_audit = self._run_evolution(cycle, checkpoint)
        candidates = self._evaluate_candidates(cycle, components)
        summary = {
            "campaign": self.config.campaign,
            "cycle": cycle,
            "status": "completed",
            "completed_at": _now(),
            "fitness_contract": fitness_contract(self.fitness_context),
            "component_count": len(components),
            "test_passed_count": sum(
                bool(item.get("test_overall_passed")) for item in candidates
            ),
            "admitted_count": sum(bool(item.get("admitted")) for item in candidates),
            "failed_count": sum(item.get("status") == "failed" for item in candidates),
            "diversity_audit": diversity_audit,
            "candidates": candidates,
        }
        _atomic_json(self._cycle_dir(cycle) / "cycle_summary.json", summary)
        self._archive_fitness_cache(cycle)
        _atomic_json(
            self.active_path,
            {"cycle": cycle, "status": "completed", "updated_at": _now()},
        )
        campaign = self._campaign_record()
        campaign["next_cycle"] = cycle + 1
        campaign["last_completed_cycle"] = cycle
        campaign["updated_at"] = _now()
        _atomic_json(self.campaign_path, campaign)
        return summary

    def run(
        self,
        *,
        forever: bool = False,
        pause_seconds: float = 60.0,
        max_cycles: int | None = None,
        stop_on_error: bool = False,
    ) -> list[dict[str, Any]]:
        if pause_seconds < 0:
            raise ValueError("pause_seconds cannot be negative")
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        summaries: list[dict[str, Any]] = []
        while True:
            try:
                summary = self.run_cycle()
                summaries.append(summary)
                print(
                    json.dumps(
                        {
                            "campaign": summary["campaign"],
                            "cycle": summary["cycle"],
                            "status": summary["status"],
                            "test_passed_count": summary["test_passed_count"],
                            "admitted_count": summary["admitted_count"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failure = {
                    "campaign": self.config.campaign,
                    "cycle": self._cycle_number(),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at": _now(),
                }
                _atomic_json(self.config.root / "last_error.json", failure)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
                if stop_on_error or not forever:
                    raise
            if not forever:
                break
            if max_cycles is not None and len(summaries) >= max_cycles:
                break
            time.sleep(pause_seconds)
        return summaries
