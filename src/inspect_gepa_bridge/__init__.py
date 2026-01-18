"""
Inspect-GEPA Bridge: Generic bridge for Inspect AI tasks with GEPA optimization.

This package provides utilities to integrate Inspect AI tasks with the GEPA
(Genetic Evolution for Prompt Adaptation) optimization framework.
"""

from inspect_gepa_bridge.adapter import InspectGEPAAdapter
from inspect_gepa_bridge.dataset import (
    DataInstBase,
    InspectSampleConverter,
    load_inspect_dataset,
)
from inspect_gepa_bridge.scoring import (
    ScorerResult,
    run_inspect_eval,
    score_to_float,
)

__all__ = [
    "InspectGEPAAdapter",
    "DataInstBase",
    "InspectSampleConverter",
    "load_inspect_dataset",
    "ScorerResult",
    "run_inspect_eval",
    "score_to_float",
]
