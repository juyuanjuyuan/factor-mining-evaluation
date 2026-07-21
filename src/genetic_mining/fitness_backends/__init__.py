"""Selectable training-fitness compute backends."""

from .base import FitnessBackend
from .mps import MPSFitnessBackend, SUPPORTED_TERMINALS, mps_runtime_status
from .registry import (
    FITNESS_BACKEND_NAMES,
    create_fitness_backend,
    fitness_backend_statuses,
    require_fitness_backend,
)

__all__ = [
    "FITNESS_BACKEND_NAMES",
    "FitnessBackend",
    "MPSFitnessBackend",
    "SUPPORTED_TERMINALS",
    "create_fitness_backend",
    "fitness_backend_statuses",
    "mps_runtime_status",
    "require_fitness_backend",
]
