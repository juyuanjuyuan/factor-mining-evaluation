#!/usr/bin/env python3
"""CPU/MPS parity and train/test isolation for GP training fitness."""

from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from genetic_mining.evolution import EvolutionConfig, evaluate_population
from genetic_mining.fitness import (
    evaluate_processed_expression,
    evaluate_program_fitness,
    fitness_contract,
    prepare_fitness_context,
)
from genetic_mining.fitness_backends import MPSFitnessBackend, mps_runtime_status
from genetic_mining.tree import ExpressionTree
from test_genetic_mining import synthetic_market_data


def terminal(value: str) -> ExpressionTree:
    return ExpressionTree("terminal", value)


def constant(value: float) -> ExpressionTree:
    return ExpressionTree("constant", value)


def function(
    name: str,
    *children: ExpressionTree,
    parameter: int | float | None = None,
) -> ExpressionTree:
    return ExpressionTree("function", name, tuple(children), parameter)


def operator_trees() -> dict[str, ExpressionTree]:
    close = terminal("c")
    open_prices = terminal("o")
    return {
        "add": function("add", close, open_prices),
        "sub": function("sub", close, open_prices),
        "mul": function("mul", close, constant(0.3)),
        "div": function("div", close, open_prices),
        "abs": function("abs", function("neg", close)),
        "sqrt": function("sqrt", close),
        "log": function("log", close),
        "inv": function("inv", close),
        "neg": function("neg", close),
        "rank_cs": function("rank_cs", close),
        "scale_cs": function("scale_cs", close),
        "signed_power": function(
            "signed_power",
            function("sub", close, open_prices),
            parameter=0.5,
        ),
        "delay": function("delay", close, parameter=5),
        "delta": function("delta", close, parameter=5),
        "ts_corr": function("ts_corr", close, open_prices, parameter=10),
        "ts_cov": function("ts_cov", close, open_prices, parameter=10),
        "decay_linear": function("decay_linear", close, parameter=10),
        "ts_min": function("ts_min", close, parameter=10),
        "ts_max": function("ts_max", close, parameter=10),
        "ts_argmin": function("ts_argmin", close, parameter=10),
        "ts_argmax": function("ts_argmax", close, parameter=10),
        "ts_rank": function("ts_rank", close, parameter=10),
        "ts_sum": function("ts_sum", close, parameter=10),
        "ts_product": function(
            "ts_product",
            function("div", close, open_prices),
            parameter=5,
        ),
        "ts_std": function("ts_std", close, parameter=10),
        "pct_terminal": terminal("pct(c, 1)"),
    }


def assert_frame_parity(cpu, mps, *, atol: float = 2e-4) -> None:
    cpu_values = cpu.to_numpy(dtype=float)
    mps_values = mps.to_numpy(dtype=float)
    assert np.array_equal(np.isnan(cpu_values), np.isnan(mps_values))
    np.testing.assert_allclose(
        mps_values,
        cpu_values,
        rtol=2e-4,
        atol=atol,
        equal_nan=True,
    )


def require_local_mps() -> bool:
    status = mps_runtime_status()
    if status["available"]:
        return True
    print(f"MPS parity tests skipped: {status['reason']}")
    return False


def test_all_gp_operators_match_cpu() -> None:
    data = synthetic_market_data(110)
    data["c"].iloc[3:7, 0] = np.nan
    data["c"].iloc[10:40, 0] = 7.5
    data["c"].iloc[20, 1:4] = data["c"].iloc[20, 1]
    context = prepare_fitness_context(
        data,
        train_start="2020-01-02",
        train_end="2020-05-15",
        preprocess_mode="none",
        minimum_ic_days=20,
    )
    backend = MPSFitnessBackend(context)
    try:
        for name, tree in operator_trees().items():
            cpu = evaluate_processed_expression(tree.to_expression(), context)
            mps = backend.evaluate_processed_tree(tree)
            try:
                assert_frame_parity(cpu, mps)
            except AssertionError as exc:
                raise AssertionError(f"CPU/MPS mismatch for {name}") from exc
    finally:
        backend.close()


def test_preprocessing_and_fitness_match_cpu() -> None:
    data = synthetic_market_data(155)
    data["st"].iloc[50:55, :2] = True
    data["o"].iloc[70, 2] = data["c"].iloc[69, 2] * 1.10
    data["cap"].iloc[60, 0] = 0.0
    data["industry"].iloc[60, 16:22] = pd.NA
    tree = function("ts_std", terminal("c"), parameter=20)
    for mode in ("market_cap", "market_cap_industry", "paper_local"):
        context = prepare_fitness_context(
            data,
            train_start="2020-01-02",
            train_end="2020-06-30",
            preprocess_mode=mode,
            minimum_ic_days=20,
        )
        cpu_factor = evaluate_processed_expression(tree.to_expression(), context)
        cpu_fitness = evaluate_program_fitness(
            tree,
            context,
            parsimony_coefficient=0.0001,
        )
        backend = MPSFitnessBackend(context)
        try:
            mps_factor = backend.evaluate_processed_tree(tree)
            mps_fitness = backend.evaluate_many(
                [tree],
                parsimony_coefficient=0.0001,
            )[0]
        finally:
            backend.close()
        assert_frame_parity(cpu_factor, mps_factor)
        assert mps_fitness.error == ""
        assert mps_fitness.ic_count == cpu_fitness.ic_count
        assert mps_fitness.pair_count == cpu_fitness.pair_count
        assert np.isclose(mps_fitness.ic_mean, cpu_fitness.ic_mean, atol=2e-6)
        assert np.isclose(mps_fitness.ic_std, cpu_fitness.ic_std, atol=2e-6)
        assert np.isclose(mps_fitness.adjusted_fitness, cpu_fitness.adjusted_fitness, atol=2e-6)
        if mode == "market_cap_industry":
            contract = fitness_contract(context)
            assert contract["neutralization_model"] == (
                "factor ~ intercept + log(total_market_cap) + "
                "industry_l1_fixed_effects"
            )
            assert contract["minimum_industry_observations"] == 3


def test_mps_fitness_cannot_see_test_returns() -> None:
    data = synthetic_market_data(155)
    modified = {name: frame.copy() for name, frame in data.items()}
    modified["o"].loc[modified["o"].index > "2020-05-29"] *= 7.0
    kwargs = {
        "train_start": "2020-01-02",
        "train_end": "2020-05-29",
        "preprocess_mode": "market_cap_industry",
        "minimum_ic_days": 20,
    }
    tree = function("delay", terminal("c"), parameter=2)
    original_context = prepare_fitness_context(data, **kwargs)
    modified_context = prepare_fitness_context(modified, **kwargs)
    original_backend = MPSFitnessBackend(original_context)
    modified_backend = MPSFitnessBackend(modified_context)
    try:
        original = original_backend.evaluate_many(
            [tree], parsimony_coefficient=0.0
        )[0]
        changed = modified_backend.evaluate_many(
            [tree], parsimony_coefficient=0.0
        )[0]
    finally:
        original_backend.close()
        modified_backend.close()
    assert original.ic_count == changed.ic_count
    assert original.pair_count == changed.pair_count
    assert np.isclose(original.ic_mean, changed.ic_mean, atol=1e-7)
    assert np.isclose(original.ic_std, changed.ic_std, atol=1e-7)


def test_mps_population_backend_and_invalid_formula() -> None:
    data = synthetic_market_data(100)
    context = prepare_fitness_context(
        data,
        train_start="2020-01-02",
        train_end="2020-05-01",
        preprocess_mode="none",
        minimum_ic_days=20,
    )
    config = EvolutionConfig(
        generations=1,
        population_size=2,
        hall_of_fame=2,
        n_components=1,
        tournament_size=2,
        n_jobs=1,
        compute_backend="mps",
    )
    trees = (terminal("c"), terminal("unsupported_terminal"))
    backend = MPSFitnessBackend(context)
    try:
        cache = {}
        scored = evaluate_population(
            trees,
            context,
            config,
            cache,
            backend=backend,
        )
        assert scored[0].fitness.error == ""
        assert "does not support terminal" in scored[1].fitness.error or "不支持终端" in scored[1].fitness.error
        assert backend.metadata["implicit_cpu_fallback"] is False
    finally:
        backend.close()
    try:
        EvolutionConfig(n_jobs=2, compute_backend="mps")
    except ValueError as exc:
        assert "n_jobs=1" in str(exc)
    else:
        raise AssertionError("MPS accepted multiple CPU worker threads")


def test_mps_population_reports_generation_progress() -> None:
    """MPS must flush completed candidates before a whole generation ends."""

    data = synthetic_market_data(100)
    context = prepare_fitness_context(
        data,
        train_start="2020-01-02",
        train_end="2020-05-01",
        preprocess_mode="none",
        minimum_ic_days=20,
    )
    config = EvolutionConfig(
        generations=1,
        population_size=2,
        hall_of_fame=2,
        n_components=1,
        tournament_size=2,
        n_jobs=1,
        compute_backend="mps",
    )
    backend = MPSFitnessBackend(context)
    backend.progress_update_interval_seconds = 0.0
    cache = {}
    reported = []
    try:
        scored = evaluate_population(
            (terminal("c"), terminal("o")),
            context,
            config,
            cache,
            on_result=reported.append,
            backend=backend,
        )
    finally:
        backend.close()
    assert len(scored) == 2
    assert len(reported) == 2
    assert set(cache) == {item.fitness.expression for item in scored}


def main() -> None:
    if not require_local_mps():
        return
    test_all_gp_operators_match_cpu()
    test_preprocessing_and_fitness_match_cpu()
    test_mps_fitness_cannot_see_test_returns()
    test_mps_population_backend_and_invalid_formula()
    test_mps_population_reports_generation_progress()
    print("genetic mining MPS parity contracts passed")


if __name__ == "__main__":
    main()
