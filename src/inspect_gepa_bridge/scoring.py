"""
Scoring utilities for bridging Inspect AI scorers with GEPA.

This module provides utilities for running Inspect evaluations and
converting scorer results to GEPA-compatible formats.
"""

from dataclasses import dataclass
from typing import Any

import inspect_ai
import inspect_ai.log
import inspect_ai.model
import inspect_ai.scorer
import inspect_ai.solver

from inspect_gepa_bridge.types import format_target


@dataclass
class ScorerResult:
    """
    Container for scorer results with metadata.

    Provides easy access to score value, explanation, and metadata
    from Inspect AI scorer results.
    """

    score: inspect_ai.scorer.Score | None
    scorer_name: str

    @property
    def value(self) -> inspect_ai.scorer.Value | None:
        """Get the score value, or None if no score."""
        return self.score.value if self.score else None

    @property
    def explanation(self) -> str | None:
        """Get the score explanation, or None if no score."""
        return self.score.explanation if self.score else None

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the score metadata, or empty dict if no score."""
        return self.score.metadata if self.score and self.score.metadata else {}

    def is_correct(self) -> bool:
        """Check if the score indicates correct answer."""
        return self.score is not None and self.score.value == inspect_ai.scorer.CORRECT

    def is_incorrect(self) -> bool:
        """Check if the score indicates incorrect answer."""
        return (
            self.score is not None and self.score.value == inspect_ai.scorer.INCORRECT
        )

    def as_float(self) -> float:
        """Convert the score value to a float, returning 0.0 if not convertible."""
        return score_to_float(self.score)


def score_to_float(score: inspect_ai.scorer.Score | None) -> float:
    """
    Convert an Inspect Score to a float value.

    Handles CORRECT/INCORRECT values as well as numeric scores.

    Args:
        score: The Inspect Score object (or None)

    Returns:
        Float value (typically 0.0-1.0)
    """
    if score is None:
        return 0.0

    value = score.value

    # Handle special values
    if value == inspect_ai.scorer.CORRECT:
        return 1.0
    if value == inspect_ai.scorer.INCORRECT:
        return 0.0
    if value == inspect_ai.scorer.NOANSWER:
        return 0.0

    # Try to convert to float (value can be str, int, float, bool, Sequence, or Mapping)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def run_inspect_eval(
    task: inspect_ai.Task,
    solver: inspect_ai.solver.Solver,
    model: str | inspect_ai.model.Model,
    log_dir: str | None = None,
    model_roles: dict[str, inspect_ai.model.Model] | None = None,
    **kwargs: Any,
) -> list[inspect_ai.log.EvalLog]:
    """
    Run an Inspect evaluation with common options.

    Args:
        task: The Inspect Task to evaluate
        model: Model identifier or Model instance
        log_dir: Optional directory for logs
        model_roles: Optional dict of role names to Model instances
        **kwargs: Additional arguments passed to inspect_ai.eval()

    Returns:
        List of EvalLog results
    """
    if isinstance(model, str):
        model = inspect_ai.model.get_model(model)

    eval_kwargs: dict[str, Any] = {"model": model, **kwargs}

    if log_dir:
        eval_kwargs["log_dir"] = log_dir

    if model_roles:
        eval_kwargs["model_roles"] = model_roles

    return inspect_ai.eval(tasks=task, solver=solver, **eval_kwargs)


def first_scorer_as_float(scores: dict[str, inspect_ai.scorer.Score]) -> float:
    """
    Default score aggregator: use the first scorer's result.

    Args:
        scores: Dict mapping scorer names to Score objects

    Returns:
        Float value from the first scorer, or 0.0 if no scores
    """
    if not scores:
        return 0.0
    return score_to_float(next(iter(scores.values())))


def default_feedback_generator(
    input_text: str,
    completion: str,
    target: str | list[str] | None,
    scores: dict[str, inspect_ai.scorer.Score],
    score: float,
) -> str:
    """
    Generate a default feedback string for the reflective dataset.

    Creates a simple feedback format showing the evaluation results
    that can be used by GEPA for prompt refinement.

    Args:
        input_text: The input/question text
        completion: The model's completion/answer
        target: The expected target answer(s)
        scores: Dict of scorer results
        score: The aggregated score

    Returns:
        Formatted feedback string
    """
    # Format target
    target_str = format_target(target)

    # Format scores
    score_strs = [f"{name}: {s.value}" for name, s in scores.items()]
    scores_str = ", ".join(score_strs) if score_strs else "N/A"

    # Build feedback
    parts = [
        f"Input: {input_text[:500]}{'...' if len(input_text) > 500 else ''}",
        f"Target: {target_str}",
        f"Completion: {completion[:500]}{'...' if len(completion) > 500 else ''}",
        f"Scores: {scores_str}",
        f"Aggregated Score: {score:.3f}",
    ]

    return "\n".join(parts)
