"""Composable evaluation-step primitives.

Each evaluation method is a plain function that accepts an ``EvaluationState``.
Methods can add metrics, tabular details, and artifact paths without coupling
the orchestration layer to a fixed metric schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd


EvaluationDetail = pd.Series | pd.DataFrame
EvaluationMethod = Callable[["EvaluationState"], None]


@dataclass(frozen=True)
class EvaluationContext:
    """Immutable inputs shared by every evaluation method."""

    factor_name: str
    artifact_name: str
    factor: pd.DataFrame
    close: pd.DataFrame
    forward_return: pd.DataFrame
    horizon: int
    n_quantiles: int
    output_dir: Path
    expression: str | None = None
    market_data: Mapping[str, pd.DataFrame] | None = None
    # The raw factor evaluated on the complete market-data history. Bounded
    # train/test evaluations keep ``factor`` as the scoring slice while
    # causality diagnostics use this matrix to locate their checkpoints in
    # the original timeline.
    source_factor: pd.DataFrame | None = None


def versioned_key(name: str, version: int) -> str:
    """Storage key for the ``version``-th production of ``name`` (1-based)."""

    return name if version == 1 else f"{name}__{version}"


def latest_versioned_key(name: str, mapping: Mapping[str, Any]) -> str | None:
    """Highest-version key for ``name`` present in ``mapping``, or None."""

    if name not in mapping:
        return None
    latest = name
    version = 2
    while versioned_key(name, version) in mapping:
        latest = versioned_key(name, version)
        version += 1
    return latest


@dataclass
class EvaluationState:
    """Outputs accumulated by an ordered list of evaluation methods.

    A pipeline may execute the same method more than once (e.g. rank IC both
    before and after market-cap neutralization). Later productions of an
    existing metric/detail/artifact/cache name are stored under a versioned
    key (``ic``, ``ic__2``, ``ic__3``, …) so earlier outputs are never lost,
    and reads through ``require_detail``/``require_cache`` resolve to the
    most recent version.
    """

    context: EvaluationContext
    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, EvaluationDetail] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    executed_methods: list[str] = field(default_factory=list)
    factor: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.factor = self.context.factor

    def replace_factor(self, factor: pd.DataFrame) -> None:
        if not factor.index.equals(
            self.context.factor.index
        ) or not factor.columns.equals(self.context.factor.columns):
            raise ValueError("Replacement factor must preserve the original axes")
        self.factor = factor

    @staticmethod
    def _versioned_store(mapping: dict[str, Any], name: str, value: Any) -> None:
        version = 1
        while versioned_key(name, version) in mapping:
            version += 1
        mapping[versioned_key(name, version)] = value

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        for name, value in metrics.items():
            self._versioned_store(self.metrics, name, value)

    def add_detail(self, name: str, detail: EvaluationDetail) -> None:
        self._versioned_store(self.details, name, detail)

    def add_artifact(self, name: str, path: Path) -> None:
        self._versioned_store(self.artifacts, name, path)

    def require_detail(self, name: str) -> EvaluationDetail:
        key = latest_versioned_key(name, self.details)
        if key is None:
            raise ValueError(
                f"Evaluation detail {name!r} is required but has not been produced"
            )
        return self.details[key]

    def add_cache(self, name: str, value: Any) -> None:
        self._versioned_store(self.cache, name, value)

    def require_cache(self, name: str) -> Any:
        key = latest_versioned_key(name, self.cache)
        if key is None:
            raise ValueError(
                f"Evaluation cache {name!r} is required but has not been produced"
            )
        return self.cache[key]


@dataclass(frozen=True)
class EvaluationMethodMetadata:
    name: str
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    required_data_symbols: tuple[str, ...]


REGISTERED_EVALUATION_METHODS: dict[str, EvaluationMethod] = {}


def evaluation_method(
    name: str,
    *,
    requires: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    required_data_symbols: tuple[str, ...] = (),
) -> Callable[[EvaluationMethod], EvaluationMethod]:
    """Register a function as a named, dependency-aware evaluation method."""

    if not name or not name.strip():
        raise ValueError("Evaluation method name cannot be empty")

    def decorator(function: EvaluationMethod) -> EvaluationMethod:
        existing = REGISTERED_EVALUATION_METHODS.get(name)
        if existing is not None and existing is not function:
            raise ValueError(f"Duplicate evaluation method name: {name}")
        setattr(
            function,
            "__evaluation_method__",
            EvaluationMethodMetadata(
                name=name,
                requires=requires,
                provides=(name, *provides),
                required_data_symbols=required_data_symbols,
            ),
        )
        REGISTERED_EVALUATION_METHODS[name] = function
        return function

    return decorator


def method_metadata(function: EvaluationMethod) -> EvaluationMethodMetadata:
    metadata = getattr(function, "__evaluation_method__", None)
    if not isinstance(metadata, EvaluationMethodMetadata):
        raise TypeError(
            f"{getattr(function, '__name__', function)!r} is not a registered "
            "evaluation method; decorate it with @evaluation_method"
        )
    return metadata


def evaluation_method_names(
    methods: Iterable[EvaluationMethod],
) -> tuple[str, ...]:
    return tuple(method_metadata(method).name for method in methods)


def evaluation_required_data_symbols(
    methods: Iterable[EvaluationMethod],
) -> set[str]:
    symbols: set[str] = set()
    for method in methods:
        symbols.update(method_metadata(method).required_data_symbols)
    return symbols


def run_evaluation_methods(
    context: EvaluationContext,
    methods: Iterable[EvaluationMethod],
) -> EvaluationState:
    """Run evaluation functions in order and validate declared dependencies."""

    selected = tuple(methods)
    if not selected:
        raise ValueError("At least one evaluation method must be selected")

    state = EvaluationState(context=context)
    executed: set[str] = set()
    for method in selected:
        metadata = method_metadata(method)
        missing = set(metadata.requires) - executed
        if missing:
            raise ValueError(
                f"Evaluation method {metadata.name!r} requires methods to run first: "
                f"{sorted(missing)}"
            )
        method(state)
        state.executed_methods.append(metadata.name)
        executed.update(metadata.provides)
    return state
