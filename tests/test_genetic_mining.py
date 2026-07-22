#!/usr/bin/env python3
"""Genetic-programming search and strict train/test isolation contracts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from engine import DEFAULT_FILES, parse_and_validate_expression
from genetic_mining.evolution import (
    EvolutionConfig,
    ScoredTree,
    evaluate_population,
    evolve_population,
    initial_population,
    select_top_components,
)
from genetic_mining.admission import admit_factor_to_library
from genetic_mining.fitness import (
    FitnessResult,
    evaluate_program_fitness,
    prepare_fitness_context,
)
from genetic_mining.runner import GeneticMiningRunner, MiningCampaignConfig
from genetic_mining.tree import ExpressionTree


def synthetic_market_data(periods: int = 190) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(20190610)
    days = pd.date_range("2020-01-02", periods=periods, freq="B")
    codes = [f"{number:06d}" for number in range(1, 25)]
    returns = rng.normal(0.0005, 0.012, size=(periods, len(codes)))
    close = pd.DataFrame(
        15 * np.cumprod(1 + returns, axis=0),
        index=days,
        columns=codes,
    )
    open_prices = close * pd.DataFrame(
        1 + rng.normal(0, 0.003, close.shape),
        index=days,
        columns=codes,
    )
    high = pd.DataFrame(
        np.maximum(close, open_prices) * 1.01,
        index=days,
        columns=codes,
    )
    low = pd.DataFrame(
        np.minimum(close, open_prices) * 0.99,
        index=days,
        columns=codes,
    )
    volume = pd.DataFrame(
        rng.uniform(1e5, 2e6, close.shape),
        index=days,
        columns=codes,
    )
    amount = volume * close
    industry_labels = np.repeat(np.array(["A", "B", "C"], dtype=object), 8)
    return {
        "c": close,
        "o": open_prices,
        "h": high,
        "l": low,
        "vol": volume,
        "amt": amount,
        "vwap": (high + low) / 2,
        "cap": pd.DataFrame(
            rng.lognormal(22, 0.8, close.shape),
            index=days,
            columns=codes,
        ),
        "limit": pd.DataFrame(0.10, index=days, columns=codes),
        "st": pd.DataFrame(False, index=days, columns=codes),
        "industry": pd.DataFrame(
            np.tile(industry_labels, (periods, 1)),
            index=days,
            columns=codes,
            dtype="string",
        ),
    }


def test_expression_trees_and_evolution() -> None:
    data = synthetic_market_data()
    config = EvolutionConfig(
        generations=2,
        population_size=8,
        hall_of_fame=8,
        n_components=2,
        tournament_size=4,
        init_depth_min=1,
        init_depth_max=2,
        max_depth=5,
        max_nodes=63,
        n_jobs=2,
        terminals=("c", "o", "vol"),
    )
    context = prepare_fitness_context(
        data,
        train_start="2020-01-02",
        train_end="2020-05-29",
        preprocess_mode="none",
        minimum_ic_days=20,
    )
    rng = np.random.default_rng(7)
    population = initial_population(rng, config)
    assert len({tree.to_expression() for tree in population}) == 8
    for tree in population:
        parse_and_validate_expression(tree.to_expression())
    cache = {}
    scored = evaluate_population(population, context, config, cache)
    offspring = evolve_population(scored, rng, config)
    assert len(offspring) == config.population_size
    assert len({tree.to_expression() for tree in offspring}) == len(offspring)


def test_hall_of_fame_selects_top_fitness_without_internal_correlation_gate() -> None:
    config = EvolutionConfig(
        generations=1,
        population_size=4,
        hall_of_fame=4,
        n_components=3,
        tournament_size=2,
    )
    scores = {"c": 0.30, "o": 0.10, "h": 0.20, "l": None}
    hall = {
        expression: ScoredTree(
            tree=ExpressionTree("terminal", expression),
            fitness=FitnessResult(
                expression=expression,
                raw_fitness=score,
                adjusted_fitness=score,
                ic_mean=score,
                ic_std=0.01 if score is not None else None,
                ir=1.0 if score is not None else None,
                ic_count=60 if score is not None else 0,
                pair_count=1200 if score is not None else 0,
                node_count=1,
                depth=1,
                error="" if score is not None else "invalid",
            ),
        )
        for expression, score in scores.items()
    }

    selected, audit = select_top_components(hall, config)

    assert [item.fitness.expression for item in selected] == ["c", "h", "o"]
    assert [item["rank"] for item in audit] == [1, 2, 3]
    assert all(item["selected_for_frozen_test"] for item in audit)


def test_training_fitness_cannot_see_test_returns() -> None:
    data = synthetic_market_data()
    modified = {name: frame.copy() for name, frame in data.items()}
    modified["o"].loc[modified["o"].index > "2020-05-29"] *= 7.0
    kwargs = {
        "train_start": "2020-01-02",
        "train_end": "2020-05-29",
        "preprocess_mode": "none",
        "minimum_ic_days": 20,
    }
    original_context = prepare_fitness_context(data, **kwargs)
    modified_context = prepare_fitness_context(modified, **kwargs)
    tree = ExpressionTree("terminal", "c")
    original = evaluate_program_fitness(
        tree,
        original_context,
        parsimony_coefficient=0.0,
    )
    changed = evaluate_program_fitness(
        tree,
        modified_context,
        parsimony_coefficient=0.0,
    )
    assert original.ic_count == changed.ic_count
    assert np.isclose(original.ic_mean, changed.ic_mean)
    assert np.isclose(original.ic_std, changed.ic_std)


def _write_market_data(root: Path, data: dict[str, pd.DataFrame]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for symbol, frame in data.items():
        frame.to_parquet(root / DEFAULT_FILES[symbol])


def test_resumable_single_cycle_smoke() -> None:
    data = synthetic_market_data()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        _write_market_data(data_dir, data)
        days = data["c"].index
        evolution = EvolutionConfig(
            generations=1,
            population_size=4,
            hall_of_fame=4,
            n_components=1,
            tournament_size=2,
            init_depth_min=1,
            init_depth_max=1,
            max_depth=4,
            max_nodes=31,
            terminals=("c",),
        )
        config = MiningCampaignConfig(
            campaign="synthetic",
            train_start=str(days[0].date()),
            train_end=str(days[104].date()),
            test_start=str(days[105].date()),
            test_end=str(days[-1].date()),
            data_dir=data_dir,
            output_dir=root / "outputs",
            library_file=root / "factor_library.json",
            correlation_state_dir=root / "state",
            evolution=evolution,
            n_quantiles=5,
            preprocess_mode="none",
            minimum_ic_days=20,
            admit=False,
        )
        runner = GeneticMiningRunner(config)
        try:
            GeneticMiningRunner(config)
        except RuntimeError as exc:
            assert "already running" in str(exc)
        else:
            raise AssertionError("campaign lock did not reject a concurrent runner")
        # Persist an active initial checkpoint, close the process, then prove a
        # fresh process resumes that exact cycle instead of starting cycle 2.
        runner._load_or_create_checkpoint(1)
        runner.close()
        resumed = GeneticMiningRunner(config)
        summary = resumed.run_cycle()
        resumed.close()
        assert summary["status"] == "completed"
        assert summary["component_count"] == 1
        assert (
            config.root / "cycles/cycle_000001/fitness_cache.jsonl"
        ).is_file()
        assert not (config.root / "fitness_cache.jsonl").exists()
        assert (config.root / "cycles/cycle_000001/checkpoint.json").is_file()
        assert (config.root / "cycles/cycle_000001/cycle_summary.json").is_file()
        assert summary["candidates"][0]["status"] == "completed"
        assert set(summary["candidates"][0]["standard_gates"]) == {"profitability_test"}
        assert not summary["candidates"][0]["admitted"]


def test_profitability_screen_runs_ic_only_for_survivors() -> None:
    data = synthetic_market_data(periods=130)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        _write_market_data(data_dir, data)
        days = data["c"].index
        config = MiningCampaignConfig(
            campaign="two-stage-screen",
            train_start=str(days[0].date()),
            train_end=str(days[64].date()),
            test_start=str(days[65].date()),
            test_end=str(days[-1].date()),
            data_dir=data_dir,
            output_dir=root / "outputs",
            library_file=root / "factor_library.json",
            correlation_state_dir=root / "state",
            evolution=EvolutionConfig(
                generations=1,
                population_size=2,
                hall_of_fame=2,
                n_components=2,
                tournament_size=1,
                terminals=("c", "o"),
            ),
            preprocess_mode="none",
            minimum_ic_days=20,
            admit=False,
        )
        component = lambda expression: ScoredTree(
            tree=ExpressionTree("terminal", expression),
            fitness=FitnessResult(
                expression=expression,
                raw_fitness=0.1,
                adjusted_fitness=0.1,
                ic_mean=0.1,
                ic_std=0.1,
                ir=1.0,
                ic_count=60,
                pair_count=1000,
                node_count=1,
                depth=1,
            ),
        )
        calls: list[tuple[str, str]] = []

        def fake_evaluate_factor_standards(*, expression, standards, **_kwargs):
            standard = next(iter(standards))
            calls.append((expression, standard))
            passed = expression == "o" and standard == "profitability_test"
            return {
                "overall_passed": passed,
                "summary_path": f"/{expression}/{standard}.json",
                "standards": {standard: {"gate": {"passed": passed}}},
            }

        runner = GeneticMiningRunner(config)
        try:
            with patch(
                "genetic_mining.runner.evaluate_factor_standards",
                side_effect=fake_evaluate_factor_standards,
            ):
                records = runner._evaluate_candidates(
                    1,
                    (component("c"), component("o")),
                )
        finally:
            runner.close()

        assert calls == [
            ("c", "profitability_test"),
            ("o", "profitability_test"),
            ("o", "ic_test"),
        ]
        assert records[0]["profitability_passed"] is False
        assert records[0]["ic_checked"] is False
        assert set(records[0]["standard_gates"]) == {"profitability_test"}
        assert records[1]["profitability_passed"] is True
        assert records[1]["ic_checked"] is True
        assert records[1]["ic_passed"] is False
        assert set(records[1]["standard_gates"]) == {
            "profitability_test",
            "ic_test",
        }


def test_correlation_gated_library_admission() -> None:
    data = synthetic_market_data(periods=90)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        library = root / "factor_registry" / "formal_library.json"
        state = root / "state"
        _write_market_data(data_dir, data)
        admitted = admit_factor_to_library(
            factor_name="gp_admission_one",
            expression="c",
            library_file=library,
            data_dir=data_dir,
            state_dir=state,
        )
        assert admitted.admitted
        assert admit_factor_to_library(
            factor_name="gp_admission_one",
            expression="c",
            library_file=library,
            data_dir=data_dir,
            state_dir=state,
        ).already_present
        rejected = admit_factor_to_library(
            factor_name="gp_admission_duplicate_exposure",
            expression="c",
            library_file=library,
            data_dir=data_dir,
            state_dir=state,
        )
        assert not rejected.admitted
        assert not rejected.correlation_passed
        assert np.isclose(rejected.violations[0]["abs_correlation"], 1.0)
        payload = json.loads(library.read_text(encoding="utf-8"))
        assert payload["factor_count"] == 1
        assert payload["factors"][0]["factor_name"] == "gp_admission_one"
        assert not {
            "ic_mean",
            "ir",
            "top_group_annualized_return",
        } & set(payload["factors"][0])


def main() -> None:
    test_expression_trees_and_evolution()
    test_hall_of_fame_selects_top_fitness_without_internal_correlation_gate()
    test_training_fitness_cannot_see_test_returns()
    test_resumable_single_cycle_smoke()
    test_profitability_screen_runs_ic_only_for_survivors()
    test_correlation_gated_library_admission()
    print("genetic mining contracts passed")


if __name__ == "__main__":
    main()
