#!/usr/bin/env python3
"""Regression contracts for correlation-conflict factor-library replacement."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4


SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_DIR.parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


def _store_profitability_run(
    *,
    db,
    settings,
    factor: dict,
    sharpe: float,
    fitness: float,
) -> None:
    from app.services.funnel_service import make_run
    from app.services.registry_service import (
        ADMISSION_HORIZON,
        ADMISSION_PROFITABILITY_METHOD_NAMES,
        ADMISSION_QUANTILES,
    )
    from returns import RETURN_DEFINITION
    from evaluators.tradability import TRADABILITY_DEFINITION

    job_id = uuid4().hex
    run = make_run(
        job_id=job_id,
        factor=factor,
        stage_name=None,
        methods=list(ADMISSION_PROFITABILITY_METHOD_NAMES),
        horizon=ADMISSION_HORIZON,
        n_quantiles=ADMISSION_QUANTILES,
        runs_dir=settings.runs_dir,
        status="running",
    )
    db.create_job(
        {
            "id": job_id,
            "kind": "evaluate",
            "title": "测试盈利能力结果",
            "params": {},
            "created_at": run["created_at"],
        },
        [run],
    )
    db.complete_run(
        run["id"],
        {
            "return_definition": RETURN_DEFINITION,
            "tradability_definition": TRADABILITY_DEFINITION,
            "signal_start": "2024-01-02",
            "signal_end": "2024-12-31",
            "sample_start_day": "2024-01-02",
            "sample_end_day": "2024-12-27",
            "gn_rolling_sharpe_60_median": sharpe,
            "fitness": fitness,
        },
    )


class _CorrelationStub:
    """Make only the named candidate conflict with the current formal factor."""

    threshold = 0.75

    def __init__(self) -> None:
        self.committed: list[tuple[str, tuple[str, ...]]] = []

    def prepare_candidate(self, candidate, existing):
        from factor_correlation import FactorCorrelationThresholdError

        names = tuple(str(item["factor_name"]) for item in existing)
        if str(candidate["factor_name"]).startswith("challenger") and names:
            raise FactorCorrelationThresholdError(
                [
                    {
                        "factor_a": names[0],
                        "factor_b": str(candidate["factor_name"]),
                        "correlation": 0.99,
                        "abs_correlation": 0.99,
                    }
                ],
                self.threshold,
            )
        return {"candidate": str(candidate["factor_name"]), "existing": names}

    def commit_candidate(self, prepared) -> None:
        self.committed.append((prepared["candidate"], prepared["existing"]))


def _service(root: Path):
    from app.config import Settings
    from app.db import Database
    from app.services.registry_service import RegistryService

    settings = Settings(
        project_root=PROJECT_ROOT,
        data_dir=root / "data",
        registry_dir=root / "registry",
        state_dir=root / "state",
        frontend_dist=PROJECT_ROOT / "webapp" / "frontend" / "dist",
    )
    db = Database(settings.db_path)
    db.initialize()
    correlation = _CorrelationStub()
    return RegistryService(settings, db=db, correlation=correlation), db, correlation


def test_replaces_only_when_sharpe_is_strictly_higher_and_preserves_test_source() -> None:
    from app.services.registry_service import FactorLibraryPerformanceRejectedError

    with TemporaryDirectory() as temporary:
        service, db, correlation = _service(Path(temporary))
        incumbent = service.create(
            {"factor_name": "incumbent", "expression": "rank_cs(c)", "tags": ["old"]}
        )
        _store_profitability_run(
            db=db,
            settings=service.settings,
            factor=incumbent,
            sharpe=1.0,
            fitness=2.0,
        )
        service.submit(incumbent["batch_id"], incumbent["factor_name"])

        challenger = service.create(
            {"factor_name": "challenger_sharpe", "expression": "rank_cs(c * 2)"}
        )
        _store_profitability_run(
            db=db,
            settings=service.settings,
            factor=challenger,
            sharpe=1.1,
            fitness=2.0,
        )
        admitted = service.submit(challenger["batch_id"], challenger["factor_name"])

        assert admitted and admitted["factor_name"] == "challenger_sharpe"
        assert admitted["admission_decision"]["replaced_factor_names"] == ["incumbent"]
        assert [row["factor_name"] for row in service.factors("factor")] == [
            "challenger_sharpe"
        ]
        assert service.find("webapp_test_factors", "incumbent")
        assert not service.find("webapp_factor_library", "incumbent")
        assert correlation.committed[-1] == ("challenger_sharpe", ())

        tied_sharpe = service.create(
            {"factor_name": "challenger_fitness", "expression": "rank_cs(c * 3)"}
        )
        _store_profitability_run(
            db=db,
            settings=service.settings,
            factor=tied_sharpe,
            sharpe=1.1,
            fitness=1.9,
        )
        try:
            service.submit(tied_sharpe["batch_id"], tied_sharpe["factor_name"])
        except FactorLibraryPerformanceRejectedError as exc:
            assert exc.decision["criterion"] == "fitness"
            assert exc.decision["blocking_factor_names"] == ["challenger_sharpe"]
        else:
            raise AssertionError("a lower-Fitness tied-Sharpe challenger must be rejected")
        assert [row["factor_name"] for row in service.factors("factor")] == [
            "challenger_sharpe"
        ]


def test_missing_metrics_runs_the_profitability_template_for_both_factors() -> None:
    from evaluators.base import REGISTERED_EVALUATION_METHODS
    from app.services.registry_service import (
        ADMISSION_PROFITABILITY_METHOD_NAMES,
    )
    from returns import RETURN_DEFINITION
    from evaluators.tradability import TRADABILITY_DEFINITION

    with TemporaryDirectory() as temporary:
        service, _db, _correlation = _service(Path(temporary))
        incumbent = service.create(
            {"factor_name": "incumbent_auto", "expression": "rank_cs(c)"}
        )
        service.submit(incumbent["batch_id"], incumbent["factor_name"])
        challenger = service.create(
            {"factor_name": "challenger_auto", "expression": "rank_cs(c * 2)"}
        )
        calls: list[dict] = []

        def fake_evaluate_factor_expression(*, factor_name, **kwargs):
            calls.append({"factor_name": factor_name, **kwargs})
            return {
                "metrics": {
                    "return_definition": RETURN_DEFINITION,
                    "tradability_definition": TRADABILITY_DEFINITION,
                    "sample_start_day": "2024-01-02",
                    "sample_end_day": "2024-12-27",
                    "gn_rolling_sharpe_60_median": (
                        1.2 if factor_name == "challenger_auto" else 0.8
                    ),
                    "fitness": 1.0,
                }
            }

        with patch(
            "app.services.registry_service.evaluate_factor_expression",
            side_effect=fake_evaluate_factor_expression,
        ):
            admitted = service.submit(challenger["batch_id"], challenger["factor_name"])

        assert admitted and admitted["admission_decision"]["replaced_factor_names"] == [
            "incumbent_auto"
        ]
        assert [call["factor_name"] for call in calls] == [
            "challenger_auto",
            "incumbent_auto",
        ]
        assert all(
            tuple(
                method.__name__
                for method in call["evaluation_methods"]
            )
            == tuple(
                method.__name__
                for method in (
                    REGISTERED_EVALUATION_METHODS[name]
                    for name in ADMISSION_PROFITABILITY_METHOD_NAMES
                )
            )
            for call in calls
        )
        auto_runs = [
            run
            for run in service.db.list_runs(limit=20)
            if (run.get("run_params") or {}).get("admission_comparison") is True
        ]
        assert len(auto_runs) == 2
        assert all(run["status"] == "succeeded" for run in auto_runs)


def test_gp_handoff_reuses_its_frozen_profitability_metrics() -> None:
    from app.services.registry_service import ADMISSION_PROFITABILITY_METHOD_NAMES
    from returns import RETURN_DEFINITION
    from evaluators.tradability import TRADABILITY_DEFINITION

    def request(name: str, expression: str, sharpe: float, fitness: float) -> dict:
        return {
            "factor_name": name,
            "expression": expression,
            "project": "遗传规划",
            "requested_at": "2026-07-25T10:00:00+08:00",
            "profitability_evaluation": {
                "standard": "profitability_test",
                "methods": list(ADMISSION_PROFITABILITY_METHOD_NAMES),
                "horizon": 1,
                "n_quantiles": 10,
                "signal_start": "2024-01-02",
                "signal_end": "2024-12-31",
                "metrics": {
                    "return_definition": RETURN_DEFINITION,
                    "tradability_definition": TRADABILITY_DEFINITION,
                    "signal_start": "2024-01-02",
                    "signal_end": "2024-12-31",
                    "sample_start_day": "2024-01-02",
                    "sample_end_day": "2024-12-27",
                    "gn_rolling_sharpe_60_median": sharpe,
                    "fitness": fitness,
                },
            },
        }

    with TemporaryDirectory() as temporary:
        service, db, _correlation = _service(Path(temporary))
        incumbent = service.create(
            {"factor_name": "gp_incumbent", "expression": "rank_cs(c)"}
        )
        _store_profitability_run(
            db=db,
            settings=service.settings,
            factor=incumbent,
            sharpe=0.8,
            fitness=0.3,
        )
        service.submit(incumbent["batch_id"], incumbent["factor_name"])
        challenger = service.submit_gp_candidate(
            request("challenger_gp", "rank_cs(c * 2)", 1.2, -2.0)
        )
        assert challenger["status"] == "admitted"
        assert challenger["replaced_factor_names"] == ["gp_incumbent"]
        assert [row["factor_name"] for row in service.factors("factor")] == [
            "challenger_gp"
        ]
        automatic = [
            run
            for run in service.db.list_runs(limit=20)
            if (run.get("run_params") or {}).get("admission_comparison") is True
        ]
        assert automatic == []


def main() -> None:
    test_replaces_only_when_sharpe_is_strictly_higher_and_preserves_test_source()
    test_missing_metrics_runs_the_profitability_template_for_both_factors()
    test_gp_handoff_reuses_its_frozen_profitability_metrics()
    print("factor-library replacement admission contracts passed")


if __name__ == "__main__":
    main()
