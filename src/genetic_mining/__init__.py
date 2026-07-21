"""Paper-inspired genetic-programming factor discovery with frozen OOS gates."""

from .admission import AdmissionResult, admit_factor_to_library
from .evolution import EvolutionConfig, ScoredTree
from .fitness import (
    FitnessContext,
    FitnessResult,
    evaluate_program_fitness,
    prepare_fitness_context,
)
from .fitness_backends import FITNESS_BACKEND_NAMES, fitness_backend_statuses
from .runner import GeneticMiningRunner, MiningCampaignConfig
from .tree import ExpressionTree

__all__ = [
    "AdmissionResult",
    "EvolutionConfig",
    "ExpressionTree",
    "FitnessContext",
    "FITNESS_BACKEND_NAMES",
    "FitnessResult",
    "GeneticMiningRunner",
    "MiningCampaignConfig",
    "ScoredTree",
    "admit_factor_to_library",
    "evaluate_program_fitness",
    "fitness_backend_statuses",
    "prepare_fitness_context",
]
