from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import inspect_ai
import inspect_ai.dataset
import inspect_ai.log
import inspect_ai.model
import inspect_ai.scorer
import inspect_ai.solver
import pytest
from gepa.core.adapter import EvaluationBatch

from inspect_gepa_bridge import TaskAdapter
from inspect_gepa_bridge.scoring import first_scorer_as_float
from inspect_gepa_bridge.task_adapter import set_system_message
from inspect_gepa_bridge.types import InspectOutput, InspectTrajectory

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


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


@patch("inspect_gepa_bridge.task_adapter.scoring.run_inspect_eval")
def test_evaluate_all_same_id_uses_epochs(mock_eval: MagicMock) -> None:
    """Batch [s1, s1, s1] runs 1 eval with epochs=3."""
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

    def make_mock_sample(sid: str, epoch: int) -> MagicMock:
        mock_sample = MagicMock()
        mock_sample.id = sid
        mock_sample.epoch = epoch
        mock_sample.scores = {
            "match": inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT)
        }
        mock_sample.messages = []
        mock_output = MagicMock()
        mock_output.completion = f"4-epoch{epoch}"
        mock_sample.output = mock_output
        return mock_sample

    mock_log = MagicMock()
    mock_log.samples = [
        make_mock_sample("s1", 1),
        make_mock_sample("s1", 2),
        make_mock_sample("s1", 3),
    ]
    mock_eval.return_value = [mock_log]

    result = adapter.evaluate(
        batch=["s1", "s1", "s1"],
        candidate={"system_prompt": "test"},
        capture_traces=False,
    )

    assert len(result.outputs) == 3
    assert result.scores == [1.0, 1.0, 1.0]
    assert result.outputs[0].completion == "4-epoch1"
    assert result.outputs[1].completion == "4-epoch2"
    assert result.outputs[2].completion == "4-epoch3"
    mock_eval.assert_called_once()
    call_args = mock_eval.call_args
    assert call_args.kwargs["epochs"] == 3
    eval_task = call_args.kwargs["task"]
    assert len(list(eval_task.dataset)) == 1


def test_evaluate_mixed_duplicates_uses_multiple_evals(mocker: MockerFixture) -> None:
    """Batch [s1, s2, s1] runs 2 evals."""
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

    def make_mock_sample(sid: str, completion: str) -> MagicMock:
        mock_sample = MagicMock()
        mock_sample.id = sid
        mock_sample.scores = {
            "match": inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT)
        }
        mock_sample.messages = []
        mock_output = MagicMock()
        mock_output.completion = completion
        mock_sample.output = mock_output
        return mock_sample

    mock_log1 = MagicMock()
    mock_log1.samples = [
        make_mock_sample("s1", "4-first"),
        make_mock_sample("s2", "6-first"),
    ]

    mock_log2 = MagicMock()
    mock_log2.samples = [
        make_mock_sample("s1", "4-second"),
    ]

    # This is more complicated than checking mock_eval.call_args, but necessary because
    # evaluate mutates task.dataset.
    expected_dataset_lengths = iter([2, 1])
    mock_eval_return_values = iter([[mock_log1], [mock_log2]])

    def mock_eval_side_effect(
        task: inspect_ai.Task, *args: Any, **kwargs: Any
    ) -> Sequence[inspect_ai.log.EvalLog]:
        assert len(list(task.dataset)) == next(expected_dataset_lengths)

        return next(mock_eval_return_values)

    mock_eval = mocker.patch(
        "inspect_gepa_bridge.task_adapter.scoring.run_inspect_eval",
        side_effect=mock_eval_side_effect,
    )

    result = adapter.evaluate(
        batch=["s1", "s2", "s1"],
        candidate={"system_prompt": "test"},
        capture_traces=False,
    )

    assert len(result.outputs) == 3
    assert result.scores == [1.0, 1.0, 1.0]
    assert result.outputs[0].completion == "4-first"
    assert result.outputs[1].completion == "6-first"
    assert result.outputs[2].completion == "4-second"
    assert mock_eval.call_count == 2
    for call in mock_eval.call_args_list:
        assert call.kwargs["epochs"] == 1


@patch("inspect_gepa_bridge.task_adapter.scoring.run_inspect_eval")
def test_evaluate_complex_duplicates(mock_eval: MagicMock) -> None:
    """Batch [s1, s2, s1, s2, s1] runs 2 evals with different epochs."""
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

    def make_mock_sample(sid: str, completion: str) -> MagicMock:
        mock_sample = MagicMock()
        mock_sample.id = sid
        mock_sample.scores = {
            "match": inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT)
        }
        mock_sample.messages = []
        mock_output = MagicMock()
        mock_output.completion = completion
        mock_sample.output = mock_output
        return mock_sample

    mock_log1 = MagicMock()
    mock_log1.samples = [
        make_mock_sample("s1", "4-e1"),
        make_mock_sample("s2", "6-e1"),
        make_mock_sample("s1", "4-e2"),
        make_mock_sample("s2", "6-e2"),
    ]

    mock_log2 = MagicMock()
    mock_log2.samples = [
        make_mock_sample("s1", "4-e3"),
    ]

    mock_eval.side_effect = [[mock_log1], [mock_log2]]

    result = adapter.evaluate(
        batch=["s1", "s2", "s1", "s2", "s1"],
        candidate={"system_prompt": "test"},
        capture_traces=False,
    )

    assert len(result.outputs) == 5
    assert result.scores == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert result.outputs[0].completion == "4-e1"
    assert result.outputs[1].completion == "6-e1"
    assert result.outputs[2].completion == "4-e2"
    assert result.outputs[3].completion == "6-e2"
    assert result.outputs[4].completion == "4-e3"
    assert mock_eval.call_count == 2
    first_call = mock_eval.call_args_list[0]
    assert first_call.kwargs["epochs"] == 2
    second_call = mock_eval.call_args_list[1]
    assert second_call.kwargs["epochs"] == 1


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
    assert result["system_prompt"][0]["Inputs"] == "2+2=?"
    assert result["system_prompt"][0]["Generated Outputs"] == "4"
    assert result["system_prompt"][0]["Feedback"] == "Correct!"


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


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({}, 0.0),
        (
            {
                "scorer1": inspect_ai.scorer.Score(value=inspect_ai.scorer.CORRECT),
                "scorer2": inspect_ai.scorer.Score(value=inspect_ai.scorer.INCORRECT),
            },
            1.0,
        ),
    ],
)
def test_first_scorer_as_float(
    scores: dict[str, inspect_ai.scorer.Score], expected: float
):
    assert first_scorer_as_float(scores) == expected


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        ("test input", "test input"),
        (
            [
                inspect_ai.model.ChatMessageUser(content="Hello"),
                inspect_ai.model.ChatMessageAssistant(content="Hi there"),
            ],
            "Hello\nHi there",
        ),
    ],
)
def test_format_input(
    input_data: str | list[inspect_ai.model.ChatMessage], expected: str
):
    samples = [inspect_ai.dataset.Sample(input="test", target="result", id="s1")]
    dataset = inspect_ai.dataset.MemoryDataset(samples)
    task = inspect_ai.Task(
        dataset=dataset,
        solver=inspect_ai.solver.generate(),
        scorer=inspect_ai.scorer.match(),
    )
    adapter = TaskAdapter(task=task, model="test-model")

    assert adapter._format_input(input_data) == expected


@pytest.mark.parametrize(
    ("solvers"),
    [
        pytest.param(
            [inspect_ai.solver.system_message("Test 1")], id="single_system_message"
        ),
        pytest.param(
            [
                inspect_ai.solver.system_message(template)
                for template in ["Test 1", "Test 2"]
            ],
            id="multiple_system_messages",
        ),
        pytest.param(
            [
                inspect_ai.solver.system_message("Test 1"),
                inspect_ai.solver.user_message("Test 2"),
            ],
            id="system_and_user_messages",
        ),
    ],
)
def test_set_system_message(solvers: list[inspect_ai.solver.Solver]) -> None:
    mock_generate = AsyncMock(spec=inspect_ai.solver.Generate)

    wrapped_solver = set_system_message(
        inspect_ai.solver.chain(*solvers, inspect_ai.solver.generate()),
        "Test 2",
    )

    state = inspect_ai.solver.TaskState(
        model=inspect_ai.model.ModelName("mockllm/model"),
        sample_id="s1",
        epoch=0,
        input="2+2=?",
        messages=[],
        target=inspect_ai.scorer.Target("4"),
    )

    asyncio.run(wrapped_solver(state, mock_generate))

    mock_generate.assert_called_once()
    generate_state = mock_generate.call_args[0][0]
    assert isinstance(generate_state, inspect_ai.solver.TaskState)
    assert isinstance(generate_state.messages[0], inspect_ai.model.ChatMessageSystem)
    assert generate_state.messages[0].content == "Test 2"
