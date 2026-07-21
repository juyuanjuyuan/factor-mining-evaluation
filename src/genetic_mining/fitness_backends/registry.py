"""Fitness-backend discovery and construction."""

from __future__ import annotations

from typing import Any

from ..fitness import FitnessContext
from .base import FitnessBackend
from .mps import MPSFitnessBackend, mps_runtime_status


FITNESS_BACKEND_NAMES = ("cpu", "mps")


def fitness_backend_statuses() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "cpu",
            "available": True,
            "reason": None,
            "device_name": "NumPy/Pandas CPU",
            "torch_version": None,
        },
        mps_runtime_status(),
    )


def require_fitness_backend(name: str) -> None:
    if name not in FITNESS_BACKEND_NAMES:
        raise ValueError(
            f"Unknown GP fitness backend {name!r}; expected {FITNESS_BACKEND_NAMES}"
        )
    if name == "mps":
        status = mps_runtime_status()
        if not status["available"]:
            raise RuntimeError(status.get("reason") or "MPS 训练后端不可用")


def create_fitness_backend(
    name: str,
    context: FitnessContext,
) -> FitnessBackend | None:
    require_fitness_backend(name)
    if name == "cpu":
        return None
    return MPSFitnessBackend(context)
