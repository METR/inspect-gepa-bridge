"""
Type definitions for the TaskAdapter.

This module provides data structures and protocols for the simplified
TaskAdapter that wraps existing Inspect AI tasks.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import inspect_ai.model
import inspect_ai.scorer


@dataclass
class InspectTrajectory:
    """
    Trajectory data for a single sample evaluation.

    Contains the full context of an evaluation including input, output,
    and scoring information for GEPA optimization.
    """

    sample_id: str | int
    input: str | list[inspect_ai.model.ChatMessage]
    target: str | list[str]
    messages: list[inspect_ai.model.ChatMessage]
    completion: str
    scores: dict[str, inspect_ai.scorer.Score]
    score: float
    feedback: str


@dataclass
class InspectOutput:
    """
    Output data from a single sample evaluation.

    Contains the completion and scores from running an Inspect evaluation.
    """

    completion: str
    scores: dict[str, inspect_ai.scorer.Score] = field(default_factory=dict)
    error: str | None = None


class ScoreAggregator(Protocol):
    """Protocol for aggregating multiple scorer results into a single float."""

    def __call__(self, scores: dict[str, inspect_ai.scorer.Score]) -> float:
        """Aggregate scores to a single float value."""
        ...


class FeedbackGenerator(Protocol):
    """Protocol for generating feedback strings from evaluation results."""

    def __call__(
        self,
        input_text: str,
        completion: str,
        target: str | list[str] | None,
        scores: dict[str, inspect_ai.scorer.Score],
        score: float,
    ) -> str:
        """Generate feedback for the reflective dataset."""
        ...


# Type alias for sample IDs
SampleId = str | int


def format_target(target: str | list[str] | Sequence[Any] | None) -> str:
    """Format a target value for display in feedback."""
    if target is None:
        return "N/A"
    if isinstance(target, str):
        return target
    return ", ".join(str(t) for t in target)
