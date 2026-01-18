"""Tests for TaskAdapter."""

from unittest.mock import MagicMock, patch

import inspect_ai
import inspect_ai.dataset
import inspect_ai.model
import inspect_ai.scorer
import inspect_ai.solver
from gepa.core.adapter import EvaluationBatch

from inspect_gepa_bridge import TaskAdapter
from inspect_gepa_bridge.scoring import first_scorer_as_float
from inspect_gepa_bridge.types import InspectOutput, InspectTrajectory


def test_task_adapter_init_builds_sample_index():
    samples = [
        inspect_ai.dataset.Sample(input="2+2=?", target="4", id="s1"),
        inspect_ai.dataset.Sample(input="3+3=?", target="6", id="s2"),
    ]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )

    adapter = TaskAdapter(task=task, model="test-model")

    assert adapter.get_sample_ids() == ["s1", "s2"]


def test_task_adapter_assigns_sequential_ids_when_missing():
    samples = [
        inspect_ai.dataset.Sample(input="2+2=?", target="4"),
        inspect_ai.dataset.Sample(input="3+3=?", target="6"),
    ]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )

    adapter = TaskAdapter(task=task, model="test-model")

    assert adapter.get_sample_ids() == [0, 1]


def test_create_eval_task_prepends_system_message():
    samples = [inspect_ai.dataset.Sample(input="test", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    original_solver = inspect_ai.solver.generate()
    task = inspect_ai.Task(
        dataset=dataset,
        solver=original_solver,
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    eval_task = adapter._create_eval_task(samples, "You are helpful.")

    assert eval_task.solver is not None
    assert eval_task.dataset is not None
    assert len(list(eval_task.dataset)) == 1


def test_create_eval_task_with_solver_list():
    samples = [inspect_ai.dataset.Sample(input="test", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    solver_list = [
        inspect_ai.solver.generate(),
        inspect_ai.solver.generate(),
    ]
    task = inspect_ai.Task(
        dataset=dataset,
        solver=solver_list,
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    eval_task = adapter._create_eval_task(samples, "You are helpful.")

    assert eval_task.solver is not None
    assert eval_task.dataset is not None
    assert len(list(eval_task.dataset)) == 1


@patch("inspect_gepa_bridge.task_adapter.scoring.run_inspect_eval")
def test_evaluate_returns_empty_batch_for_missing_samples(mock_eval: MagicMock) -> None:
    samples = [inspect_ai.dataset.Sample(input="test", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    result = adapter.evaluate(
        batch=["nonexistent"],
        candidate={"system_prompt": "test"},
        capture_traces=False,
    )

    assert len(result.outputs) == 1
    assert result.scores == [0.0]


@patch("inspect_gepa_bridge.task_adapter.scoring.run_inspect_eval")
def test_evaluate_processes_results(mock_eval: MagicMock) -> None:
    samples = [inspect_ai.dataset.Sample(input="2+2=?", target="4", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    mock_sample = MagicMock()
    mock_sample.id = "s1"
    mock_sample.scores = {
        "match": inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT)
    }
    mock_sample.messages = []
    mock_output = MagicMock()
    mock_output.completion = "4"
    mock_sample.output = mock_output

    mock_log = MagicMock()
    mock_log.samples = [mock_sample]
    mock_eval.return_value = [mock_log]

    result = adapter.evaluate(
        batch=["s1"],
        candidate={"system_prompt": "You are a math tutor."},
        capture_traces=True,
    )

    assert len(result.outputs) == 1
    assert isinstance(result.outputs[0], InspectOutput)
    assert result.outputs[0].completion == "4"
    assert result.scores == [1.0]
    assert result.trajectories is not None
    assert len(result.trajectories) == 1
    assert isinstance(result.trajectories[0], InspectTrajectory)


@patch("inspect_gepa_bridge.task_adapter.scoring.run_inspect_eval")
def test_evaluate_handles_failed_results(mock_eval: MagicMock) -> None:
    samples = [inspect_ai.dataset.Sample(input="test", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    mock_log = MagicMock()
    mock_log.samples = None
    mock_eval.return_value = [mock_log]

    result = adapter.evaluate(
        batch=["s1"],
        candidate={"system_prompt": "test"},
        capture_traces=False,
    )

    assert len(result.outputs) == 1
    assert result.outputs[0].error == "Eval failed: no results"
    assert result.scores == [0.0]


def test_make_reflective_dataset() -> None:
    samples = [inspect_ai.dataset.Sample(input="2+2=?", target="4", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    traj = InspectTrajectory(
        sample_id="s1",
        input="2+2=?",
        target="4",
        messages=[],
        completion="4",
        scores={"match": inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT)},
        score=1.0,
        feedback="Correct!",
    )
    eval_batch: EvaluationBatch[InspectTrajectory, InspectOutput] = EvaluationBatch(
        outputs=[InspectOutput(completion="4")],
        scores=[1.0],
        trajectories=[traj],
        objective_scores=None,
    )

    result = adapter.make_reflective_dataset(
        candidate={"system_prompt": "test"},
        eval_batch=eval_batch,
        components_to_update=["system_prompt"],
    )

    assert "system_prompt" in result
    assert len(result["system_prompt"]) == 1
    # GEPA recommended schema keys
    assert result["system_prompt"][0]["Inputs"] == "2+2=?"
    assert result["system_prompt"][0]["Generated Outputs"] == "4"
    assert result["system_prompt"][0]["Feedback"] == "Correct!"
    # Additional keys
    assert result["system_prompt"][0]["target"] == "4"
    assert result["system_prompt"][0]["score"] == 1.0


def test_make_reflective_dataset_no_trajectories() -> None:
    samples = [inspect_ai.dataset.Sample(input="test", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    eval_batch: EvaluationBatch[InspectTrajectory, InspectOutput] = EvaluationBatch(
        outputs=[InspectOutput(completion="result")],
        scores=[1.0],
        trajectories=None,
        objective_scores=None,
    )

    result = adapter.make_reflective_dataset(
        candidate={"system_prompt": "test"},
        eval_batch=eval_batch,
        components_to_update=["system_prompt", "other"],
    )

    assert result == {"system_prompt": [], "other": []}


def test_custom_score_aggregator() -> None:
    samples = [inspect_ai.dataset.Sample(input="test", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )

    def custom_aggregator(scores: dict[str, inspect_ai.scorer.Score]) -> float:
        return 0.5

    adapter = TaskAdapter(
        task=task, model="test-model", score_aggregator=custom_aggregator
    )

    assert adapter.score_aggregator is custom_aggregator


def test_custom_feedback_generator() -> None:
    samples = [inspect_ai.dataset.Sample(input="test", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )

    def custom_feedback(
        input_text: str,
        completion: str,
        target: str | list[str] | None,
        scores: dict[str, inspect_ai.scorer.Score],
        score: float,
    ) -> str:
        return f"Custom: {score}"

    adapter = TaskAdapter(
        task=task, model="test-model", feedback_generator=custom_feedback
    )

    assert adapter.feedback_generator is custom_feedback


def test_first_scorer_as_float_empty_scores():
    assert first_scorer_as_float({}) == 0.0


def test_first_scorer_as_float_with_scores():
    scores = {
        "scorer1": inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT),
        "scorer2": inspect_ai.scorer.Score(value=inspect_ai.scorer.INCORRECT),
    }
    assert first_scorer_as_float(scores) == 1.0


def test_format_input_string():
    samples = [inspect_ai.dataset.Sample(input="test input", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    result = adapter._format_input("test input")

    assert result == "test input"


def test_format_input_chat_messages():
    samples = [inspect_ai.dataset.Sample(input="test", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    messages: list[inspect_ai.model.ChatMessage] = [
        inspect_ai.model.ChatMessageUser(content="Hello"),
        inspect_ai.model.ChatMessageAssistant(content="Hi there"),
    ]

    result = adapter._format_input(messages)

    assert result == "Hello\nHi there"
