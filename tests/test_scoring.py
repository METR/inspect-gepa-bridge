"""Tests for scoring utilities."""

import inspect_ai.scorer

from inspect_gepa_bridge.scoring import (
    ScorerResult,
    default_feedback_generator,
    score_to_float,
)
from inspect_gepa_bridge.types import format_target


def test_score_to_float_correct():
    score = inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT)
    assert score_to_float(score) == 1.0


def test_score_to_float_incorrect():
    score = inspect_ai.scorer.Score(value=inspect_ai.scorer.INCORRECT)
    assert score_to_float(score) == 0.0


def test_score_to_float_numeric():
    score = inspect_ai.scorer.Score(value=0.75)
    assert score_to_float(score) == 0.75


def test_score_to_float_none():
    assert score_to_float(None) == 0.0


def test_scorer_result_is_correct():
    score = inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT)
    result = ScorerResult(score=score, scorer_name="test")
    assert result.is_correct()
    assert not result.is_incorrect()


def test_scorer_result_is_incorrect():
    score = inspect_ai.scorer.Score(value=inspect_ai.scorer.INCORRECT)
    result = ScorerResult(score=score, scorer_name="test")
    assert result.is_incorrect()
    assert not result.is_correct()


def test_scorer_result_as_float():
    score = inspect_ai.scorer.Score(value=0.5)
    result = ScorerResult(score=score, scorer_name="test")
    assert result.as_float() == 0.5


def test_scorer_result_none_score():
    result = ScorerResult(score=None, scorer_name="test")
    assert result.value is None
    assert result.explanation is None
    assert result.metadata == {}
    assert result.as_float() == 0.0


def test_format_target_none():
    assert format_target(None) == "N/A"


def test_format_target_string():
    assert format_target("expected answer") == "expected answer"


def test_format_target_list():
    assert format_target(["answer1", "answer2"]) == "answer1, answer2"


def test_format_target_sequence():
    assert format_target(("a", "b", "c")) == "a, b, c"


def test_default_feedback_generator_basic():
    scores = {"accuracy": inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT)}

    result = default_feedback_generator(
        input_text="What is 2+2?",
        completion="4",
        target="4",
        scores=scores,
        score=1.0,
    )

    assert "Input: What is 2+2?" in result
    assert "Target: 4" in result
    assert "Completion: 4" in result
    assert "Scores: accuracy: C" in result
    assert "Aggregated Score: 1.000" in result


def test_default_feedback_generator_list_target():
    result = default_feedback_generator(
        input_text="Question",
        completion="Answer",
        target=["ans1", "ans2"],
        scores={},
        score=0.5,
    )

    assert "Target: ans1, ans2" in result


def test_default_feedback_generator_none_target():
    result = default_feedback_generator(
        input_text="Question",
        completion="Answer",
        target=None,
        scores={},
        score=0.0,
    )

    assert "Target: N/A" in result


def test_default_feedback_generator_truncates_long_input():
    long_input = "x" * 600

    result = default_feedback_generator(
        input_text=long_input,
        completion="short",
        target="target",
        scores={},
        score=0.5,
    )

    assert "Input: " + "x" * 500 + "..." in result


def test_default_feedback_generator_truncates_long_completion():
    long_completion = "y" * 600

    result = default_feedback_generator(
        input_text="short",
        completion=long_completion,
        target="target",
        scores={},
        score=0.5,
    )

    assert "Completion: " + "y" * 500 + "..." in result


def test_default_feedback_generator_empty_scores():
    result = default_feedback_generator(
        input_text="Question",
        completion="Answer",
        target="target",
        scores={},
        score=0.0,
    )

    assert "Scores: N/A" in result
