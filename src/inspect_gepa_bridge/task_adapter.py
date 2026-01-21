"""TaskAdapter for wrapping existing Inspect AI tasks with GEPA optimization."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gepa.core.adapter
import inspect_ai
import inspect_ai.dataset
import inspect_ai.log
import inspect_ai.model
import inspect_ai.scorer
import inspect_ai.solver
from gepa.core.adapter import EvaluationBatch
from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, solver

from inspect_gepa_bridge import scoring
from inspect_gepa_bridge.types import (
    FeedbackGenerator,
    InspectOutput,
    InspectTrajectory,
    SampleId,
    ScoreAggregator,
)


@solver
def set_system_message(template: str) -> Solver:
    """Replace (not prepend) all system messages with a single new one."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages = [
            msg for msg in state.messages if not isinstance(msg, ChatMessageSystem)
        ]
        state.messages.insert(0, ChatMessageSystem(content=template))
        return state

    return solve


@dataclass
class _SampleResult:
    output: InspectOutput
    score: float
    trajectory: InspectTrajectory | None


class TaskAdapter(
    gepa.core.adapter.GEPAAdapter[SampleId, InspectTrajectory, InspectOutput]
):
    """Wraps an Inspect AI Task for GEPA optimization."""

    def __init__(
        self,
        task: inspect_ai.Task,
        model: str,
        *,
        score_aggregator: ScoreAggregator | None = None,
        feedback_generator: FeedbackGenerator | None = None,
        log_dir: str | None = None,
        model_roles: dict[str, str] | None = None,
    ):
        self.task = task
        self.model = model
        self.score_aggregator = score_aggregator or scoring.first_scorer_as_float
        self.feedback_generator = (
            feedback_generator or scoring.default_feedback_generator
        )
        self.log_dir = log_dir
        self.model_roles = model_roles or {}

        self._sample_index: dict[SampleId, inspect_ai.dataset.Sample] = {}
        self._sample_ids: list[SampleId] = []
        self._build_sample_index()

    def _build_sample_index(self) -> None:
        for i, sample in enumerate(self.task.dataset):
            sample_id: SampleId = sample.id if sample.id is not None else i
            self._sample_index[sample_id] = sample
            self._sample_ids.append(sample_id)

    def get_sample_ids(self) -> list[SampleId]:
        return list(self._sample_ids)

    def evaluate(
        self,
        batch: list[SampleId],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[InspectTrajectory, InspectOutput]:
        """Handles duplicate sample IDs by running multiple evals with epochs."""
        system_prompt = candidate.get("system_prompt", "")

        remaining: dict[SampleId, int] = {}
        for sid in batch:
            if sid in self._sample_index:
                remaining[sid] = remaining.get(sid, 0) + 1

        if not remaining:
            return self._create_empty_batch(batch, capture_traces)

        all_results: dict[tuple[SampleId, int], _SampleResult] = {}
        epoch_counters: dict[SampleId, int] = {sid: 0 for sid in remaining}
        model_roles_resolved = self._resolve_model_roles()

        while any(r > 0 for r in remaining.values()):
            sample_ids_to_eval = [sid for sid, count in remaining.items() if count > 0]
            samples_to_eval = [self._sample_index[sid] for sid in sample_ids_to_eval]
            epochs = min(remaining[sid] for sid in sample_ids_to_eval)

            eval_task = self._create_eval_task(samples_to_eval, system_prompt)
            results = scoring.run_inspect_eval(
                task=eval_task,
                model=self.model,
                log_dir=self.log_dir,
                model_roles=model_roles_resolved,
                epochs=epochs,
            )

            self._collect_epoch_results(
                results,
                samples_to_eval,
                epochs,
                all_results,
                epoch_counters,
                capture_traces,
            )

            for sid in sample_ids_to_eval:
                remaining[sid] -= epochs

        return self._build_batch_from_results(batch, all_results, capture_traces)

    def _create_eval_task(
        self,
        samples: list[inspect_ai.dataset.Sample],
        system_prompt: str,
    ) -> inspect_ai.Task:
        original_solver = self.task.solver
        if isinstance(original_solver, list):
            solver_chain: list[inspect_ai.solver.Solver] = [
                *original_solver,
                set_system_message(template=system_prompt),
            ]
        else:
            solver_chain = [
                original_solver,
                set_system_message(template=system_prompt),
            ]

        return inspect_ai.Task(
            dataset=inspect_ai.dataset.MemoryDataset(samples),
            solver=solver_chain,
            scorer=self.task.scorer,
            sandbox=self.task.sandbox,
            message_limit=self.task.message_limit,
        )

    def _resolve_model_roles(self) -> dict[str, inspect_ai.model.Model] | None:
        if not self.model_roles:
            return None
        return {
            role: inspect_ai.model.get_model(mid)
            for role, mid in self.model_roles.items()
        }

    def _collect_epoch_results(
        self,
        eval_results: list[inspect_ai.log.EvalLog],
        samples: list[inspect_ai.dataset.Sample],
        epochs: int,
        all_results: dict[tuple[SampleId, int], _SampleResult],
        epoch_counters: dict[SampleId, int],
        capture_traces: bool,
    ) -> None:
        if not eval_results or eval_results[0].samples is None:
            for sample in samples:
                sid: SampleId = sample.id  # pyright: ignore[reportAssignmentType]
                for _ in range(epochs):
                    epoch_idx = epoch_counters[sid]
                    epoch_counters[sid] += 1
                    all_results[(sid, epoch_idx)] = _SampleResult(
                        output=InspectOutput(
                            completion="", scores={}, error="Eval failed: no results"
                        ),
                        score=0.0,
                        trajectory=(
                            self._create_failed_trajectory(
                                sid, "Eval failed: no results"
                            )
                            if capture_traces
                            else None
                        ),
                    )
            return

        for eval_log in eval_results:
            assert eval_log.samples is not None

            for eval_sample in eval_log.samples:
                sample_id: SampleId = eval_sample.id  # pyright: ignore[reportAssignmentType]
                epoch_idx = epoch_counters[sample_id]
                epoch_counters[sample_id] += 1

                original_sample = self._sample_index.get(sample_id)
                if original_sample is None:
                    all_results[(sample_id, epoch_idx)] = _SampleResult(
                        output=InspectOutput(
                            completion="",
                            scores={},
                            error=f"Sample {sample_id} not found in index",
                        ),
                        score=0.0,
                        trajectory=(
                            self._create_failed_trajectory(
                                sample_id, f"Sample {sample_id} not found"
                            )
                            if capture_traces
                            else None
                        ),
                    )
                    continue

                completion = eval_sample.output.completion
                sample_scores: dict[str, inspect_ai.scorer.Score] = (
                    eval_sample.scores or {}
                )
                score = self.score_aggregator(sample_scores)

                output = InspectOutput(completion=completion, scores=sample_scores)

                trajectory: InspectTrajectory | None = None
                if capture_traces:
                    target = original_sample.target
                    input_text = self._format_input(original_sample.input)
                    feedback = self.feedback_generator(
                        input_text, completion, target, sample_scores, score
                    )
                    trajectory = InspectTrajectory(
                        sample_id=sample_id,
                        input=original_sample.input,
                        target=target,
                        messages=(
                            list(eval_sample.messages) if eval_sample.messages else []
                        ),
                        completion=completion,
                        scores=sample_scores,
                        score=score,
                        feedback=feedback,
                    )

                all_results[(sample_id, epoch_idx)] = _SampleResult(
                    output=output, score=score, trajectory=trajectory
                )

    def _build_batch_from_results(
        self,
        batch: list[SampleId],
        all_results: dict[tuple[SampleId, int], _SampleResult],
        capture_traces: bool,
    ) -> EvaluationBatch[InspectTrajectory, InspectOutput]:
        outputs: list[InspectOutput] = []
        scores: list[float] = []
        trajectories: list[InspectTrajectory] | None = [] if capture_traces else None
        batch_epoch_counters: dict[SampleId, int] = {}

        for sample_id in batch:
            epoch_idx = batch_epoch_counters.get(sample_id, 0)
            batch_epoch_counters[sample_id] = epoch_idx + 1

            result = all_results.get((sample_id, epoch_idx))
            if result is None:
                outputs.append(
                    InspectOutput(
                        completion="",
                        scores={},
                        error=f"Sample {sample_id} not found in results",
                    )
                )
                scores.append(0.0)
                if trajectories is not None:
                    trajectories.append(
                        self._create_failed_trajectory(
                            sample_id, f"Sample {sample_id} not found"
                        )
                    )
            else:
                outputs.append(result.output)
                scores.append(result.score)
                if trajectories is not None and result.trajectory is not None:
                    trajectories.append(result.trajectory)
                elif trajectories is not None:
                    trajectories.append(
                        self._create_failed_trajectory(sample_id, "No trajectory")
                    )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None,
        )

    def _create_empty_batch(
        self,
        batch: list[SampleId],
        capture_traces: bool,
    ) -> EvaluationBatch[InspectTrajectory, InspectOutput]:
        outputs: list[InspectOutput] = []
        scores: list[float] = []
        trajectories: list[InspectTrajectory] | None = [] if capture_traces else None

        for sample_id in batch:
            output = InspectOutput(
                completion="", scores={}, error="Eval failed: no results"
            )
            outputs.append(output)
            scores.append(0.0)
            if trajectories is not None:
                trajectories.append(
                    self._create_failed_trajectory(sample_id, "Eval failed: no results")
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None,
        )

    def _create_failed_trajectory(
        self,
        sample_id: SampleId,
        error: str,
    ) -> InspectTrajectory:
        return InspectTrajectory(
            sample_id=sample_id,
            input="",
            target="",
            messages=[],
            completion="",
            scores={},
            score=0.0,
            feedback=error,
        )

    def _format_input(
        self,
        input_data: str | list[inspect_ai.model.ChatMessage],
    ) -> str:
        if isinstance(input_data, str):
            return input_data
        return "\n".join(
            msg.content if isinstance(msg.content, str) else str(msg.content)
            for msg in input_data
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[InspectTrajectory, InspectOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            return {comp: [] for comp in components_to_update}

        examples: list[Mapping[str, Any]] = []
        for traj in eval_batch.trajectories:
            input_text = (
                traj.input
                if isinstance(traj.input, str)
                else self._format_input(traj.input)
            )

            example: dict[str, Any] = {
                "Inputs": input_text,
                "Generated Outputs": traj.completion,
                "Feedback": traj.feedback,
            }
            examples.append(example)

        return {comp: examples for comp in components_to_update}
