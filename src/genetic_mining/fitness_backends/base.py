"""Backend contract for training-only GP fitness computation."""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

import pandas as pd

from ..fitness import FitnessResult
from ..tree import ExpressionTree


class FitnessBackend(Protocol):
    """Compute frozen-expression fitness without reading the test window."""

    name: str

    @property
    def metadata(self) -> dict[str, Any]: ...

    def evaluate_many(
        self,
        trees: Sequence[ExpressionTree],
        *,
        parsimony_coefficient: float,
        on_progress: Callable[[tuple[FitnessResult, ...]], None] | None = None,
    ) -> tuple[FitnessResult, ...]: ...

    def evaluate_processed_tree(self, tree: ExpressionTree) -> pd.DataFrame: ...

    def close(self) -> None: ...
