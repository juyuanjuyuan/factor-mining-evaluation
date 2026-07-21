"""Paper-inspired genetic-programming factor discovery with frozen OOS gates."""

from .admission import AdmissionResult, admit_factor_to_library
from .evolution import EvolutionConfig, ScoredTree
from .fitness import (
    FitnessContext,
    FitnessResult,
    evaluate_program_fitness,
    prepare_fitness_context,
)
from .runner import GeneticMiningRunner, MiningCampaignConfig
from .tree import ExpressionTree

__all__ = [
    "AdmissionResult",
    "EvolutionConfig",
    "ExpressionTree",
    "FitnessContext",
    "FitnessResult",
    "GeneticMiningRunner",
    "MiningCampaignConfig",
    "ScoredTree",
    "admit_factor_to_library",
    "evaluate_program_fitness",
    "prepare_fitness_context",
]
