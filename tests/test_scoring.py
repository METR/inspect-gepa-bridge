"""Tests for scoring utilities."""

import inspect_ai.scorer

from inspect_gepa_bridge.scoring import ScorerResult, score_to_float


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
