"""Deterministic, resumable genetic-programming evolution operators."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np

from .fitness import (
    FitnessContext,
    FitnessResult,
    apply_direction_multiplier,
    evaluate_program_fitness,
)
from .tree import (
    DEFAULT_EXPONENTS,
    DEFAULT_TERMINALS,
    DEFAULT_WINDOWS,
    FUNCTION_SPECS,
    ExpressionTree,
    canonicalize_tree,
    delete_node,
    insert_node,
    mutate_constant,
    mutate_window,
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
    p_crossover: float = 0.25
    p_subtree_mutation: float = 0.18
    p_delete_mutation: float = 0.12
    p_insert_mutation: float = 0.12
    p_point_mutation: float = 0.10
    p_window_mutation: float = 0.08
    p_constant_mutation: float = 0.05
    p_hoist_mutation: float = 0.05
    p_random_tree: float = 0.05
    p_point_replace: float = 1.0
    max_depth: int = 8
    max_nodes: int = 127
    complexity_warmup: bool = True
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
            self.p_delete_mutation,
            self.p_insert_mutation,
            self.p_point_mutation,
            self.p_window_mutation,
            self.p_constant_mutation,
            self.p_hoist_mutation,
            self.p_random_tree,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("genetic operation probabilities must be between zero and one")
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("genetic operation probabilities must sum to one")
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


def active_limits(config: EvolutionConfig, generation: int) -> EvolutionConfig:
    """Return the complexity limits used to construct/evaluate one generation."""

    if not 0 <= generation < config.generations:
        raise ValueError("generation must be within the configured search")
    if (
        not config.complexity_warmup
        or config.generations == 1
        or generation == config.generations - 1
    ):
        return config
    progress = generation / (config.generations - 1)
    if progress < 0.5:
        max_depth = min(config.max_depth, max(config.init_depth_max, 4))
        max_nodes = min(config.max_nodes, 31)
    else:
        max_depth = min(config.max_depth, max(config.init_depth_max, 6))
        max_nodes = min(config.max_nodes, 63)
    return replace(config, max_depth=max_depth, max_nodes=max_nodes)


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
        item.tree.to_expression(),
    )


def frozen_expression(item: ScoredTree) -> str:
    """Return the training-frozen, direction-aligned expression for a program."""

    return item.fitness.frozen_expression or item.fitness.expression


def _direction_normalized_scored_tree(
    tree: ExpressionTree,
    fitness: FitnessResult,
) -> ScoredTree:
    """Make the post-training, positive-IC tree the generation's genotype."""

    return ScoredTree(
        tree=canonicalize_tree(
            apply_direction_multiplier(tree, fitness.direction_multiplier)
        ),
        fitness=fitness,
    )


def _unique_parent_pool(scored: tuple[ScoredTree, ...]) -> tuple[ScoredTree, ...]:
    """Deduplicate sign-equivalent, direction-normalized candidates before selection."""

    unique: dict[str, ScoredTree] = {}
    for item in scored:
        expression = item.tree.to_expression()
        existing = unique.get(expression)
        if existing is None or scored_sort_key(item) < scored_sort_key(existing):
            unique[expression] = item
    return tuple(sorted(unique.values(), key=scored_sort_key))


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
        tree = canonicalize_tree(random_tree(
            rng,
            target_depth=depth,
            grow=bool(len(result) % 2),
            terminals=config.terminals,
            functions=FUNCTION_SPECS,
            windows=config.windows,
            exponents=config.exponents,
            constant_range=config.constant_range,
        ))
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
        _direction_normalized_scored_tree(
            tree,
            cache[tree.to_expression()],
        )
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
    candidate = canonicalize_tree(candidate)
    return candidate if valid_tree(
        candidate,
        max_depth=config.max_depth,
        max_nodes=config.max_nodes,
    ) else fallback


def evolve_population(
    scored: tuple[ScoredTree, ...],
    rng: np.random.Generator,
    config: EvolutionConfig,
) -> tuple[ExpressionTree, ...]:
    """Create one duplicate-free generation without an implicit copy channel."""

    parent_pool = _unique_parent_pool(scored)
    if not parent_pool:
        raise ValueError("cannot evolve an empty scored population")
    ranked = list(parent_pool)
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
            config.p_delete_mutation,
            config.p_insert_mutation,
            config.p_point_mutation,
            config.p_window_mutation,
            config.p_constant_mutation,
            config.p_hoist_mutation,
            config.p_random_tree,
        ]
    )
    while len(next_population) < config.population_size and attempts < max_attempts:
        attempts += 1
        parent = tournament_select(
            parent_pool,
            rng,
            tournament_size=config.tournament_size,
        )
        draw = float(rng.random())
        if draw < cutoffs[0]:
            donor = tournament_select(
                parent_pool,
                rng,
                tournament_size=config.tournament_size,
            )
            child = _crossover(parent, donor, rng)
        elif draw < cutoffs[1]:
            child = _subtree_mutation(parent, rng, config)
        elif draw < cutoffs[2]:
            child = delete_node(parent, rng)
        elif draw < cutoffs[3]:
            child = insert_node(
                parent,
                rng,
                functions=FUNCTION_SPECS,
                windows=config.windows,
                exponents=config.exponents,
            )
        elif draw < cutoffs[4]:
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
        elif draw < cutoffs[5]:
            child = mutate_window(parent, rng, windows=config.windows)
        elif draw < cutoffs[6]:
            child = mutate_constant(
                parent,
                rng,
                constant_range=config.constant_range,
            )
        elif draw < cutoffs[7]:
            child = _hoist_mutation(parent, rng)
        else:
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
        child = _valid_offspring(child, parent, config)
        expression = child.to_expression()
        if expression not in seen:
            seen.add(expression)
            next_population.append(child)

    while len(next_population) < config.population_size and attempts < max_attempts * 2:
        attempts += 1
        depth = int(rng.integers(config.init_depth_min, config.init_depth_max + 1))
        child = canonicalize_tree(random_tree(
            rng,
            target_depth=depth,
            grow=True,
            terminals=config.terminals,
            functions=FUNCTION_SPECS,
            windows=config.windows,
            exponents=config.exponents,
            constant_range=config.constant_range,
        ))
        expression = child.to_expression()
        if valid_tree(child, max_depth=config.max_depth, max_nodes=config.max_nodes) and expression not in seen:
            seen.add(expression)
            next_population.append(child)
    if len(next_population) != config.population_size:
        raise RuntimeError("Could not create a duplicate-free GP generation")
    return tuple(next_population)


def aligned_ic_mean(item: ScoredTree) -> float:
    value = item.fitness.aligned_ic_mean
    if value is None and item.fitness.ic_mean is not None:
        value = abs(float(item.fitness.ic_mean))
    return float(value) if value is not None and np.isfinite(value) else -np.inf


def dominates(left: ScoredTree, right: ScoredTree) -> bool:
    """Return whether left is no worse in IC/complexity and better in one."""

    left_ic = aligned_ic_mean(left)
    right_ic = aligned_ic_mean(right)
    return (
        left_ic >= right_ic
        and left.tree.node_count <= right.tree.node_count
        and (
            left_ic > right_ic
            or left.tree.node_count < right.tree.node_count
        )
    )


def _pareto_quality_key(item: ScoredTree) -> tuple[float, int, float, str]:
    return (
        -aligned_ic_mean(item),
        item.tree.node_count,
        -item.fitness.selection_score,
        item.tree.to_expression(),
    )


def pareto_front(candidates: list[ScoredTree]) -> list[ScoredTree]:
    """Deduplicate, retain the best candidate per size, then take the frontier."""

    unique: dict[str, ScoredTree] = {}
    for raw_item in candidates:
        if not np.isfinite(aligned_ic_mean(raw_item)):
            continue
        tree = canonicalize_tree(raw_item.tree)
        item = ScoredTree(tree=tree, fitness=raw_item.fitness)
        expression = tree.to_expression()
        existing = unique.get(expression)
        if existing is None or _pareto_quality_key(item) < _pareto_quality_key(existing):
            unique[expression] = item

    best_by_size: dict[int, ScoredTree] = {}
    for item in unique.values():
        existing = best_by_size.get(item.tree.node_count)
        if existing is None or _pareto_quality_key(item) < _pareto_quality_key(existing):
            best_by_size[item.tree.node_count] = item
    candidates_by_size = sorted(
        best_by_size.values(),
        key=lambda item: (item.tree.node_count, -aligned_ic_mean(item), item.tree.to_expression()),
    )
    return [
        item
        for item in candidates_by_size
        if not any(
            other is not item and dominates(other, item)
            for other in candidates_by_size
        )
    ]


def _complexity_coverage_prune(
    candidates: list[ScoredTree],
    limit: int,
) -> list[ScoredTree]:
    """Keep the strongest expression and spread remaining slots across sizes."""

    if limit < 1:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (item.tree.node_count, -aligned_ic_mean(item), item.tree.to_expression()),
    )
    if len(ordered) <= limit:
        return ordered
    strongest = min(ordered, key=_pareto_quality_key)
    selected: dict[str, ScoredTree] = {
        strongest.tree.to_expression(): strongest
    }
    for raw_index in np.linspace(0, len(ordered) - 1, num=limit):
        item = ordered[int(round(float(raw_index)))]
        selected.setdefault(item.tree.to_expression(), item)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        for item in sorted(ordered, key=_pareto_quality_key):
            selected.setdefault(item.tree.to_expression(), item)
            if len(selected) == limit:
                break
    return sorted(
        selected.values(),
        key=lambda item: (item.tree.node_count, -aligned_ic_mean(item), item.tree.to_expression()),
    )


def update_hall_of_fame(
    hall: MutableMapping[str, ScoredTree],
    scored: tuple[ScoredTree, ...],
    *,
    limit: int,
) -> None:
    front = pareto_front([*hall.values(), *scored])
    retained = _complexity_coverage_prune(front, limit)
    hall.clear()
    hall.update((item.tree.to_expression(), item) for item in retained)


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

    front = pareto_front(list(hall.values()))
    selected_items = _complexity_coverage_prune(front, config.n_components)
    selected = tuple(sorted(selected_items, key=_pareto_quality_key))
    audit = [
        {
            "rank": rank,
            "expression": frozen_expression(item),
            "raw_expression": item.fitness.expression,
            "direction_multiplier": item.fitness.direction_multiplier,
            "training_ic_mean": item.fitness.ic_mean,
            "aligned_ic_mean": item.fitness.aligned_ic_mean,
            "node_count": item.tree.node_count,
            "adjusted_fitness": item.fitness.adjusted_fitness,
            "selection_score": item.fitness.selection_score,
            "pareto_front": True,
            "selected_for_frozen_test": True,
        }
        for rank, item in enumerate(selected, start=1)
    ]
    return selected, audit
