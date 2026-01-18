"""
Inspect-GEPA Bridge: Generic bridge for Inspect AI tasks with GEPA optimization.

This package provides utilities to integrate Inspect AI tasks with the GEPA
(Genetic Evolution for Prompt Adaptation) optimization framework.
"""

from inspect_gepa_bridge.scoring import (
    ScorerResult,
    default_feedback_generator,
    first_scorer_as_float,
    run_inspect_eval,
    score_to_float,
)
from inspect_gepa_bridge.task_adapter import TaskAdapter
from inspect_gepa_bridge.types import (
    FeedbackGenerator,
    InspectOutput,
    InspectTrajectory,
    ScoreAggregator,
)

__all__ = [
    "TaskAdapter",
    "InspectTrajectory",
    "InspectOutput",
    "ScoreAggregator",
    "FeedbackGenerator",
    "first_scorer_as_float",
    "default_feedback_generator",
    "ScorerResult",
    "run_inspect_eval",
    "score_to_float",
]
