#!/usr/bin/env python3
"""Genetic-programming search and strict train/test isolation contracts."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from engine import DEFAULT_FILES, parse_and_validate_expression
from genetic_mining.evolution import (
    EvolutionConfig,
    ScoredTree,
    _unique_parent_pool,
    active_limits,
    dominates,
    evaluate_population,
    evolve_population,
    initial_population,
    pareto_front,
    pareto_fronts,
    select_top_components,
    tournament_select,
    update_hall_of_fame,
)
from genetic_mining.admission import admit_factor_to_library
from genetic_mining.fitness import (
    FitnessResult,
    evaluate_processed_expression,
    evaluate_program_fitness,
    prepare_fitness_context,
)
from genetic_mining.runner import GeneticMiningRunner, MiningCampaignConfig
from genetic_mining.tree import (
    FUNCTION_BY_NAME,
    FUNCTION_SPECS,
    ExpressionTree,
    canonicalize_tree,
    delete_node,
    insert_node,
    mutate_constant,
    mutate_window,
    point_mutation,
)


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
        "delisting": pd.DataFrame(False, index=days, columns=codes),
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


def _scored_tree(tree: ExpressionTree, aligned_ic: float) -> ScoredTree:
    expression = tree.to_expression()
    return ScoredTree(
        tree=tree,
        fitness=FitnessResult(
            expression=expression,
            raw_fitness=aligned_ic,
            adjusted_fitness=aligned_ic - 0.0001 * tree.node_count,
            ic_mean=aligned_ic,
            ic_std=0.01,
            ir=aligned_ic / 0.01,
            ic_count=60,
            pair_count=1200,
            node_count=tree.node_count,
            depth=tree.depth,
            aligned_ic_mean=aligned_ic,
            frozen_expression=expression,
            frozen_node_count=tree.node_count,
        ),
    )


def test_pareto_hall_of_fame_and_complexity_stratified_selection() -> None:
    config = EvolutionConfig(
        generations=1,
        population_size=4,
        hall_of_fame=4,
        n_components=3,
        tournament_size=2,
    )
    close = ExpressionTree("terminal", "c")
    candidate_c = _scored_tree(close, 0.021)
    candidate_a = _scored_tree(ExpressionTree("function", "abs", (close,)), 0.025)
    candidate_d = _scored_tree(
        ExpressionTree(
            "function",
            "log",
            (ExpressionTree("function", "abs", (close,)),),
        ),
        0.023,
    )
    candidate_b = _scored_tree(
        ExpressionTree(
            "function",
            "add",
            (
                ExpressionTree("function", "abs", (close,)),
                ExpressionTree("terminal", "o"),
            ),
        ),
        0.027,
    )
    assert dominates(candidate_a, candidate_d)
    assert not dominates(candidate_a, candidate_b)
    front = pareto_front([candidate_a, candidate_b, candidate_c, candidate_d])
    assert {item.tree.to_expression() for item in front} == {
        candidate_a.tree.to_expression(),
        candidate_b.tree.to_expression(),
        candidate_c.tree.to_expression(),
    }

    hall: dict[str, ScoredTree] = {}
    update_hall_of_fame(
        hall,
        (candidate_a, candidate_b, candidate_c, candidate_d),
        limit=config.hall_of_fame,
    )
    selected, audit = select_top_components(hall, config)

    assert [item.fitness.expression for item in selected] == [
        candidate_b.fitness.expression,
        candidate_a.fitness.expression,
        candidate_c.fitness.expression,
    ]
    assert [item["rank"] for item in audit] == [1, 2, 3]
    assert [item["node_count"] for item in audit] == [4, 2, 1]
    assert all(item["pareto_front"] for item in audit)
    assert all(item["selected_for_frozen_test"] for item in audit)


def test_pareto_hall_of_fame_fills_from_successive_fronts() -> None:
    config = EvolutionConfig(
        generations=1,
        population_size=4,
        hall_of_fame=4,
        n_components=3,
        tournament_size=2,
    )
    candidates = tuple(
        _scored_tree(ExpressionTree("terminal", name), score)
        for name, score in (
            ("c", 0.04),
            ("o", 0.03),
            ("h", 0.02),
            ("l", 0.01),
        )
    )

    fronts = pareto_fronts(list(candidates))
    assert [
        [item.tree.to_expression() for item in front]
        for front in fronts
    ] == [["c"], ["o"], ["h"], ["l"]]

    hall: dict[str, ScoredTree] = {}
    update_hall_of_fame(hall, candidates, limit=config.hall_of_fame)
    selected, audit = select_top_components(hall, config)

    assert len(hall) == config.hall_of_fame
    assert [item.tree.to_expression() for item in selected] == ["c", "o", "h"]
    assert [item["pareto_rank"] for item in audit] == [1, 2, 3]
    assert [item["pareto_front"] for item in audit] == [True, False, False]


def test_pareto_hall_of_fame_freezes_one_hundred_when_available() -> None:
    config = EvolutionConfig(
        generations=1,
        population_size=120,
        hall_of_fame=100,
        n_components=100,
        tournament_size=20,
    )
    candidates = tuple(
        _scored_tree(
            ExpressionTree("terminal", f"synthetic_{index:03d}"),
            0.20 - index * 0.001,
        )
        for index in range(120)
    )

    hall: dict[str, ScoredTree] = {}
    update_hall_of_fame(hall, candidates, limit=config.hall_of_fame)
    selected, audit = select_top_components(hall, config)

    assert len(hall) == 100
    assert len(selected) == 100
    assert len(audit) == 100
    assert [item["pareto_rank"] for item in audit] == list(range(1, 101))


def test_safe_tree_canonicalization_and_single_site_mutations() -> None:
    close = ExpressionTree("terminal", "c")
    zero = ExpressionTree("constant", 0.0)
    one = ExpressionTree("constant", 1.0)
    double_negative = ExpressionTree(
        "function",
        "neg",
        (ExpressionTree("function", "neg", (close,)),),
    )
    assert canonicalize_tree(double_negative) == close
    assert canonicalize_tree(ExpressionTree("function", "add", (zero, close))) == close
    assert canonicalize_tree(ExpressionTree("function", "mul", (one, close))) == close
    assert canonicalize_tree(
        ExpressionTree(
            "function",
            "add",
            (ExpressionTree("terminal", "o"), close),
        )
    ).children == (close, ExpressionTree("terminal", "o"))

    window_tree = ExpressionTree("function", "ts_rank", (close,), 20)
    changed_window = mutate_window(
        window_tree,
        np.random.default_rng(1),
        windows=(10, 20, 40),
    )
    assert changed_window.parameter in {10, 40}

    constant_tree = ExpressionTree(
        "function",
        "add",
        (close, ExpressionTree("constant", 0.3)),
    )
    changed_constant = mutate_constant(
        constant_tree,
        np.random.default_rng(2),
        constant_range=(-1.0, 1.0),
    )
    assert changed_constant.to_expression() != constant_tree.to_expression()

    assert delete_node(
        ExpressionTree("function", "abs", (close,)),
        np.random.default_rng(3),
    ) == close
    inserted = insert_node(
        close,
        np.random.default_rng(4),
        functions=tuple(spec for spec in FUNCTION_SPECS if spec.arity == 1),
    )
    assert inserted.node_count == 2
    assert inserted.children == (close,)

    operator_mutated = point_mutation(
        ExpressionTree("function", "abs", (close,)),
        np.random.default_rng(5),
        replacement_probability=1.0,
        functions=(FUNCTION_BY_NAME["abs"], FUNCTION_BY_NAME["log"]),
    )
    assert operator_mutated.value == "log"
    assert operator_mutated.children == (close,)


def test_complexity_warmup_opens_full_limits_only_for_final_generation() -> None:
    config = EvolutionConfig(
        generations=3,
        population_size=4,
        hall_of_fame=4,
        n_components=2,
        tournament_size=2,
        max_depth=8,
        max_nodes=127,
    )
    limits = [active_limits(config, generation) for generation in range(3)]
    assert [(item.max_depth, item.max_nodes) for item in limits] == [
        (4, 31),
        (6, 63),
        (8, 127),
    ]
    assert active_limits(
        replace(config, generations=1),
        0,
    ).max_nodes == 127
    assert active_limits(
        replace(config, complexity_warmup=False),
        0,
    ).max_nodes == 127


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


def test_negative_training_ic_freezes_a_negated_expression() -> None:
    data = synthetic_market_data()
    context = prepare_fitness_context(
        data,
        train_start="2020-01-02",
        train_end="2020-05-29",
        preprocess_mode="none",
        minimum_ic_days=20,
    )
    tree = ExpressionTree("terminal", "c")
    processed = evaluate_processed_expression(tree.to_expression(), context)
    negative_ic_context = replace(context, forward_return=-processed)

    result = evaluate_program_fitness(
        tree,
        negative_ic_context,
        parsimony_coefficient=0.01,
    )

    assert result.ic_mean is not None and result.ic_mean < 0
    assert result.raw_fitness == result.ic_mean
    assert result.direction_multiplier == -1
    assert result.frozen_expression == "(-(c))"
    assert result.aligned_ic_mean is not None
    assert np.isclose(result.aligned_ic_mean, abs(result.ic_mean))
    assert result.frozen_node_count == tree.node_count + 1
    assert result.selection_score > 0
    assert np.isclose(
        result.adjusted_fitness,
        result.aligned_ic_mean - 0.01 * result.frozen_node_count,
    )

    already_negated = ExpressionTree("function", "neg", (tree,))
    negated_factor = evaluate_processed_expression(
        already_negated.to_expression(), context
    )
    simplified = evaluate_program_fitness(
        already_negated,
        replace(context, forward_return=-negated_factor),
        parsimony_coefficient=0.01,
    )
    assert simplified.direction_multiplier == -1
    assert simplified.frozen_expression == "c"
    assert simplified.frozen_node_count == 1


def test_negative_training_ic_is_normalized_before_tournament_evolution() -> None:
    """A negative raw tree must never be the parent genotype for its generation."""

    data = synthetic_market_data()
    context = prepare_fitness_context(
        data,
        train_start="2020-01-02",
        train_end="2020-05-29",
        preprocess_mode="none",
        minimum_ic_days=20,
    )
    raw_tree = ExpressionTree("terminal", "c")
    processed = evaluate_processed_expression(raw_tree.to_expression(), context)
    negative_ic_context = replace(context, forward_return=-processed)
    config = EvolutionConfig(
        generations=1,
        population_size=1,
        hall_of_fame=1,
        n_components=1,
        tournament_size=1,
        elite_size=1,
        terminals=("c",),
    )

    scored = evaluate_population(
        (raw_tree,),
        negative_ic_context,
        config,
        {},
    )

    assert scored[0].fitness.ic_mean is not None and scored[0].fitness.ic_mean < 0
    assert scored[0].tree.to_expression() == "(-(c))"
    assert scored[0].tree.to_expression() == scored[0].fitness.frozen_expression
    parent = tournament_select(
        scored,
        np.random.default_rng(7),
        tournament_size=1,
    )
    assert parent.to_expression() == "(-(c))"
    offspring = evolve_population(scored, np.random.default_rng(8), config)
    assert offspring[0].to_expression() == "(-(c))"

    hall: dict[str, ScoredTree] = {}
    update_hall_of_fame(hall, scored, limit=1)
    assert list(hall) == ["(-(c))"]

    mirrored_tree = ExpressionTree("function", "neg", (raw_tree,))
    mirrored_config = EvolutionConfig(
        generations=1,
        population_size=2,
        hall_of_fame=2,
        n_components=1,
        tournament_size=1,
        terminals=("c",),
    )
    mirrored_scored = evaluate_population(
        (raw_tree, mirrored_tree),
        negative_ic_context,
        mirrored_config,
        {},
    )
    assert {item.tree.to_expression() for item in mirrored_scored} == {"(-(c))"}
    assert [item.tree.to_expression() for item in _unique_parent_pool(mirrored_scored)] == [
        "(-(c))"
    ]


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
            evolution=evolution,
            n_quantiles=5,
            preprocess_mode="none",
            minimum_ic_days=20,
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
        assert "admission" not in summary["candidates"][0]
        assert "admitted" not in summary["candidates"][0]
        assert "admitted_count" not in summary


def test_three_generation_runner_persists_warmup_and_pareto_state() -> None:
    data = synthetic_market_data(periods=130)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        _write_market_data(data_dir, data)
        days = data["c"].index
        config = MiningCampaignConfig(
            campaign="phase-one-search",
            train_start=str(days[0].date()),
            train_end=str(days[104].date()),
            test_start=str(days[105].date()),
            test_end=str(days[-1].date()),
            data_dir=data_dir,
            output_dir=root / "outputs",
            evolution=EvolutionConfig(
                generations=3,
                population_size=8,
                hall_of_fame=6,
                n_components=3,
                tournament_size=3,
                init_depth_min=1,
                init_depth_max=2,
                max_depth=8,
                max_nodes=127,
                terminals=("c", "o", "vol"),
            ),
            n_quantiles=5,
            preprocess_mode="none",
            minimum_ic_days=20,
        )
        runner = GeneticMiningRunner(config)
        try:
            with patch.object(runner, "_evaluate_candidates", return_value=[]):
                summary = runner.run_cycle()
        finally:
            runner.close()

        cycle = config.root / "cycles/cycle_000001"
        limits = []
        for generation in range(1, 4):
            payload = json.loads(
                (
                    cycle
                    / "generations"
                    / f"generation_{generation:03d}.json"
                ).read_text(encoding="utf-8")
            )
            limits.append(
                (payload["active_max_depth"], payload["active_max_nodes"])
            )
            assert payload["pareto_hall_of_fame_size"] <= config.evolution.hall_of_fame
        assert limits == [(4, 31), (6, 63), (8, 127)]
        checkpoint = json.loads(
            (cycle / "checkpoint.json").read_text(encoding="utf-8")
        )
        hall = [
            ScoredTree.from_dict(item)
            for item in checkpoint["hall_of_fame"]
        ]
        assert len(hall) == config.evolution.hall_of_fame
        assert len(checkpoint["selection_audit"]) == config.evolution.n_components
        assert [item["pareto_rank"] for item in checkpoint["selection_audit"]] == sorted(
            item["pareto_rank"] for item in checkpoint["selection_audit"]
        )
        assert all(
            item["pareto_front"] == (item["pareto_rank"] == 1)
            for item in checkpoint["selection_audit"]
        )
        assert summary["search_protocol"] == config.test_screening_protocol


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
                frozen_expression="(-(c))" if expression == "c" else expression,
                direction_multiplier=-1 if expression == "c" else 1,
            ),
        )
        calls: list[tuple[str, str]] = []

        def fake_evaluate_factor_standards(*, expression, standards, **_kwargs):
            standard = next(iter(standards))
            calls.append((expression, standard))
            passed = expression == "(-(c))"
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
            ("(-(c))", "profitability_test"),
            ("(-(c))", "ic_test"),
            ("o", "profitability_test"),
        ]
        assert records[0]["profitability_passed"] is True
        assert records[0]["ic_checked"] is True
        assert records[0]["ic_passed"] is True
        assert records[0]["test_overall_passed"] is True
        assert records[0]["factor_library_submission_requested"] is True
        assert records[0]["expression"] == "(-(c))"
        assert records[0]["raw_expression"] == "c"
        assert records[0]["direction_multiplier"] == -1
        assert set(records[0]["standard_gates"]) == {
            "profitability_test",
            "ic_test",
        }
        assert records[1]["profitability_passed"] is False
        assert records[1]["ic_checked"] is False
        assert set(records[1]["standard_gates"]) == {"profitability_test"}
        request_path = (
            config.root
            / "cycles/cycle_000001/candidates"
            / records[0]["factor_name"]
            / "factor_library_submission_request.json"
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert request["factor_name"] == records[0]["factor_name"]
        assert request["expression"] == "(-(c))"
        assert request["raw_expression"] == "c"
        assert request["direction_multiplier"] == -1
        assert request["testing_protocol"] == config.test_screening_protocol
        assert not request_path.with_name("factor_library_submission.json").exists()


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
    test_pareto_hall_of_fame_and_complexity_stratified_selection()
    test_pareto_hall_of_fame_fills_from_successive_fronts()
    test_pareto_hall_of_fame_freezes_one_hundred_when_available()
    test_safe_tree_canonicalization_and_single_site_mutations()
    test_complexity_warmup_opens_full_limits_only_for_final_generation()
    test_training_fitness_cannot_see_test_returns()
    test_negative_training_ic_freezes_a_negated_expression()
    test_negative_training_ic_is_normalized_before_tournament_evolution()
    test_resumable_single_cycle_smoke()
    test_three_generation_runner_persists_warmup_and_pareto_state()
    test_profitability_screen_runs_ic_only_for_survivors()
    test_correlation_gated_library_admission()
    print("genetic mining contracts passed")


if __name__ == "__main__":
    main()
