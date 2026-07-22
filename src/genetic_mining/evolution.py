"""Deterministic, resumable genetic-programming evolution operators."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np

from .fitness import (
    FitnessContext,
    FitnessResult,
    evaluate_program_fitness,
)
from .tree import (
    DEFAULT_EXPONENTS,
    DEFAULT_TERMINALS,
    DEFAULT_WINDOWS,
    FUNCTION_SPECS,
    ExpressionTree,
    point_mutation,
    random_tree,
    valid_tree,
)

@dataclass(frozen=True)
class EvolutionConfig:
    """Paper defaults plus explicit operational bounds for expression bloat."""

    generations: int = 3
    population_size: int = 1000
    hall_of_fame: int = 100
    n_components: int = 100
    init_depth_min: int = 1
    init_depth_max: int = 4
    tournament_size: int = 20
    parsimony_coefficient: float = 0.0001
    p_crossover: float = 0.40
    p_subtree_mutation: float = 0.01
    p_hoist_mutation: float = 0.0
    p_point_mutation: float = 0.01
    p_point_replace: float = 0.40
    max_depth: int = 8
    max_nodes: int = 127
    elite_size: int = 1
    n_jobs: int = 1
    compute_backend: str = "cpu"
    terminals: tuple[str, ...] = DEFAULT_TERMINALS
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    exponents: tuple[float, ...] = DEFAULT_EXPONENTS
    constant_range: tuple[float, float] | None = (-1.0, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "generations",
            "population_size",
            "hall_of_fame",
            "n_components",
            "init_depth_min",
            "init_depth_max",
            "tournament_size",
            "max_depth",
            "max_nodes",
            "elite_size",
            "n_jobs",
        ):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.init_depth_min > self.init_depth_max:
            raise ValueError("init_depth_min cannot exceed init_depth_max")
        if self.init_depth_max > self.max_depth:
            raise ValueError("initial depth cannot exceed max_depth")
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size cannot exceed population_size")
        if self.elite_size > self.population_size:
            raise ValueError("elite_size cannot exceed population_size")
        if self.n_components > self.hall_of_fame:
            raise ValueError("n_components cannot exceed hall_of_fame")
        if self.hall_of_fame > self.population_size * self.generations:
            raise ValueError("hall_of_fame exceeds all possible generation slots")
        probabilities = (
            self.p_crossover,
            self.p_subtree_mutation,
            self.p_hoist_mutation,
            self.p_point_mutation,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("genetic operation probabilities must be between zero and one")
        if sum(probabilities) > 1:
            raise ValueError("genetic operation probabilities cannot sum above one")
        if not 0 <= self.p_point_replace <= 1:
            raise ValueError("p_point_replace must be between zero and one")
        if self.parsimony_coefficient < 0:
            raise ValueError("parsimony_coefficient cannot be negative")
        if not self.terminals or not self.windows or not self.exponents:
            raise ValueError("terminals, windows, and exponents cannot be empty")
        if self.compute_backend not in {"cpu", "mps"}:
            raise ValueError("compute_backend must be cpu or mps")
        if self.compute_backend == "mps" and self.n_jobs != 1:
            raise ValueError("MPS backend requires n_jobs=1; the GPU supplies parallelism")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoredTree:
    tree: ExpressionTree
    fitness: FitnessResult

    def as_dict(self) -> dict[str, Any]:
        return {"tree": self.tree.to_dict(), "fitness": self.fitness.as_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScoredTree":
        return cls(
            tree=ExpressionTree.from_dict(payload["tree"]),
            fitness=FitnessResult.from_dict(payload["fitness"]),
        )


def scored_sort_key(item: ScoredTree) -> tuple[float, int, str]:
    return (
        -item.fitness.selection_score,
        item.tree.node_count,
        item.fitness.expression,
    )


def initial_population(
    rng: np.random.Generator,
    config: EvolutionConfig,
) -> tuple[ExpressionTree, ...]:
    result: list[ExpressionTree] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = config.population_size * 100
    while len(result) < config.population_size and attempts < max_attempts:
        attempts += 1
        depth = int(rng.integers(config.init_depth_min, config.init_depth_max + 1))
        tree = random_tree(
            rng,
            target_depth=depth,
            grow=bool(len(result) % 2),
            terminals=config.terminals,
            functions=FUNCTION_SPECS,
            windows=config.windows,
            exponents=config.exponents,
            constant_range=config.constant_range,
        )
        expression = tree.to_expression()
        if valid_tree(tree, max_depth=config.max_depth, max_nodes=config.max_nodes) and expression not in seen:
            seen.add(expression)
            result.append(tree)
    if len(result) != config.population_size:
        raise RuntimeError("Could not generate a unique initial GP population")
    return tuple(result)


def evaluate_population(
    population: tuple[ExpressionTree, ...],
    context: FitnessContext,
    config: EvolutionConfig,
    cache: MutableMapping[str, FitnessResult],
    *,
    on_result: Callable[[FitnessResult], None] | None = None,
    backend: "FitnessBackend | None" = None,
) -> tuple[ScoredTree, ...]:
    """Evaluate unique programs, reusing a durable expression-level cache."""

    missing: list[ExpressionTree] = []
    for tree in population:
        if tree.to_expression() not in cache:
            missing.append(tree)

    def calculate(tree: ExpressionTree) -> FitnessResult:
        return evaluate_program_fitness(
            tree,
            context,
            parsimony_coefficient=config.parsimony_coefficient,
        )

    def record(result: FitnessResult) -> None:
        cache[result.expression] = result
        if on_result is not None:
            on_result(result)

    if backend is not None:
        def record_progress(results: tuple[FitnessResult, ...]) -> None:
            for result in results:
                record(result)

        calculated = backend.evaluate_many(
            missing,
            parsimony_coefficient=config.parsimony_coefficient,
            on_progress=record_progress,
        )
        executor = None
    elif config.n_jobs == 1:
        calculated = map(calculate, missing)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=config.n_jobs)
        calculated = executor.map(calculate, missing)
    try:
        for result in calculated:
            # MPS reports completed batches during a long generation.  CPU
            # evaluation and custom backends are recorded here after yielding.
            if result.expression not in cache:
                record(result)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return tuple(
        ScoredTree(tree=tree, fitness=cache[tree.to_expression()])
        for tree in population
    )


def tournament_select(
    scored: tuple[ScoredTree, ...],
    rng: np.random.Generator,
    *,
    tournament_size: int,
) -> ExpressionTree:
    positions = rng.integers(0, len(scored), size=tournament_size)
    competitors = [scored[int(position)] for position in positions]
    return min(competitors, key=scored_sort_key).tree


def _crossover(
    parent: ExpressionTree,
    donor: ExpressionTree,
    rng: np.random.Generator,
) -> ExpressionTree:
    parent_path = parent.paths()[int(rng.integers(0, len(parent.paths())))]
    donor_path = donor.paths()[int(rng.integers(0, len(donor.paths())))]
    return parent.replace(parent_path, donor.subtree(donor_path))


def _subtree_mutation(
    parent: ExpressionTree,
    rng: np.random.Generator,
    config: EvolutionConfig,
) -> ExpressionTree:
    path = parent.paths()[int(rng.integers(0, len(parent.paths())))]
    replacement_depth = int(rng.integers(0, config.init_depth_max + 1))
    replacement = random_tree(
        rng,
        target_depth=replacement_depth,
        grow=True,
        terminals=config.terminals,
        functions=FUNCTION_SPECS,
        windows=config.windows,
        exponents=config.exponents,
        constant_range=config.constant_range,
    )
    return parent.replace(path, replacement)


def _hoist_mutation(
    parent: ExpressionTree,
    rng: np.random.Generator,
) -> ExpressionTree:
    outer_path = parent.paths()[int(rng.integers(0, len(parent.paths())))]
    outer = parent.subtree(outer_path)
    inner_paths = outer.paths()
    inner_path = inner_paths[int(rng.integers(0, len(inner_paths)))]
    return parent.replace(outer_path, outer.subtree(inner_path))


def _valid_offspring(
    candidate: ExpressionTree,
    fallback: ExpressionTree,
    config: EvolutionConfig,
) -> ExpressionTree:
    return (
        candidate
        if valid_tree(candidate, max_depth=config.max_depth, max_nodes=config.max_nodes)
        else fallback
    )


def evolve_population(
    scored: tuple[ScoredTree, ...],
    rng: np.random.Generator,
    config: EvolutionConfig,
) -> tuple[ExpressionTree, ...]:
    """Create one duplicate-free generation using the paper's four operators."""

    ranked = sorted(scored, key=scored_sort_key)
    next_population: list[ExpressionTree] = [
        item.tree for item in ranked[: config.elite_size]
    ]
    seen = {tree.to_expression() for tree in next_population}
    attempts = 0
    max_attempts = config.population_size * 200
    cutoffs = np.cumsum(
        [
            config.p_crossover,
            config.p_subtree_mutation,
            config.p_hoist_mutation,
            config.p_point_mutation,
        ]
    )
    while len(next_population) < config.population_size and attempts < max_attempts:
        attempts += 1
        parent = tournament_select(
            scored,
            rng,
            tournament_size=config.tournament_size,
        )
        draw = float(rng.random())
        if draw < cutoffs[0]:
            donor = tournament_select(
                scored,
                rng,
                tournament_size=config.tournament_size,
            )
            child = _crossover(parent, donor, rng)
        elif draw < cutoffs[1]:
            child = _subtree_mutation(parent, rng, config)
        elif draw < cutoffs[2]:
            child = _hoist_mutation(parent, rng)
        elif draw < cutoffs[3]:
            child = point_mutation(
                parent,
                rng,
                replacement_probability=config.p_point_replace,
                terminals=config.terminals,
                functions=FUNCTION_SPECS,
                windows=config.windows,
                exponents=config.exponents,
                constant_range=config.constant_range,
            )
        else:
            child = parent
        child = _valid_offspring(child, parent, config)
        expression = child.to_expression()
        if expression not in seen:
            seen.add(expression)
            next_population.append(child)

    while len(next_population) < config.population_size and attempts < max_attempts * 2:
        attempts += 1
        depth = int(rng.integers(config.init_depth_min, config.init_depth_max + 1))
        child = random_tree(
            rng,
            target_depth=depth,
            grow=True,
            terminals=config.terminals,
            functions=FUNCTION_SPECS,
            windows=config.windows,
            exponents=config.exponents,
            constant_range=config.constant_range,
        )
        expression = child.to_expression()
        if valid_tree(child, max_depth=config.max_depth, max_nodes=config.max_nodes) and expression not in seen:
            seen.add(expression)
            next_population.append(child)
    if len(next_population) != config.population_size:
        raise RuntimeError("Could not create a duplicate-free GP generation")
    return tuple(next_population)


def update_hall_of_fame(
    hall: MutableMapping[str, ScoredTree],
    scored: tuple[ScoredTree, ...],
    *,
    limit: int,
) -> None:
    for item in scored:
        existing = hall.get(item.fitness.expression)
        if existing is None or scored_sort_key(item) < scored_sort_key(existing):
            hall[item.fitness.expression] = item
    ranked = sorted(hall.values(), key=scored_sort_key)
    hall.clear()
    hall.update((item.fitness.expression, item) for item in ranked[:limit])


def select_top_components(
    hall: Mapping[str, ScoredTree],
    config: EvolutionConfig,
) -> tuple[tuple[ScoredTree, ...], list[dict[str, Any]]]:
    """Freeze the highest-fitness HOF programs for independent test evaluation.

    Diversity is enforced only by formal-library admission.  A training-only
    pairwise HOF filter can discard a candidate merely because it resembles a
    higher-ranked program which later fails the frozen test set or the existing
    library's correlation gate.
    """

    ranked = [
        item
        for item in sorted(hall.values(), key=scored_sort_key)
        if np.isfinite(item.fitness.selection_score)
    ]
    selected = tuple(ranked[: config.n_components])
    audit = [
        {
            "rank": rank,
            "expression": item.fitness.expression,
            "adjusted_fitness": item.fitness.adjusted_fitness,
            "selection_score": item.fitness.selection_score,
            "selected_for_frozen_test": True,
        }
        for rank, item in enumerate(selected, start=1)
    ]
    return selected, audit
