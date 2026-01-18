"""
Scoring utilities for bridging Inspect AI scorers with GEPA.

This module provides utilities for running Inspect evaluations and
converting scorer results to GEPA-compatible formats.
"""

from dataclasses import dataclass
from typing import Any

import inspect_ai
import inspect_ai.model
import inspect_ai.scorer


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
    def value(self) -> Any:
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


def extract_scores_from_sample(
    sample: Any,  # inspect_ai.log.EvalSample
    scorer_names: list[str],
) -> dict[str, ScorerResult]:
    """
    Extract scores for specific scorers from an Inspect sample result.

    Args:
        sample: The Inspect EvalSample from evaluation results
        scorer_names: List of scorer names to extract

    Returns:
        Dict mapping scorer name to ScorerResult
    """
    sample_scores: dict[str, inspect_ai.scorer.Score] = sample.scores or {}
    results: dict[str, ScorerResult] = {}

    for name in scorer_names:
        score: inspect_ai.scorer.Score | None = sample_scores.get(name)
        results[name] = ScorerResult(score=score, scorer_name=name)

    return results


def run_inspect_eval(
    task: inspect_ai.Task,
    model: str | inspect_ai.model.Model,
    log_dir: str | None = None,
    model_roles: dict[str, inspect_ai.model.Model] | None = None,
    **kwargs: Any,
) -> list[Any]:  # list[inspect_ai.log.EvalLog]
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

    return inspect_ai.eval(task, **eval_kwargs)


def get_completion_from_sample(sample: Any) -> str:
    """
    Extract the completion text from an Inspect sample result.

    Args:
        sample: The Inspect EvalSample from evaluation results

    Returns:
        The completion text, or empty string if not available
    """
    if sample.output is None:
        return ""
    return sample.output.completion or ""
