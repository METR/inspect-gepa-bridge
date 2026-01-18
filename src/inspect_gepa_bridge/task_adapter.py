"""
TaskAdapter for wrapping existing Inspect AI tasks with GEPA optimization.

This module provides a simpler adapter that wraps an existing Inspect Task
directly, without requiring users to implement abstract methods.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import inspect_ai
import inspect_ai.dataset
import inspect_ai.model
import inspect_ai.scorer
import inspect_ai.solver
from gepa.core.adapter import EvaluationBatch

from inspect_gepa_bridge import scoring
from inspect_gepa_bridge.types import (
    FeedbackGenerator,
    InspectOutput,
    InspectTrajectory,
    SampleId,
    ScoreAggregator,
    format_target,
)


class TaskAdapter:
    """
    Adapter that wraps an existing Inspect AI Task for GEPA optimization.

    This adapter takes a complete Inspect Task (with dataset, solver, scorer)
    and makes it compatible with GEPA's optimization interface. The key
    simplification is that users don't need to implement any abstract methods;
    instead, customization is done through optional constructor parameters.

    Usage:
        task = Task(
            dataset=hf_dataset("gsm8k", split="train[:100]"),
            solver=generate(),
            scorer=model_graded_fact(),
        )
        adapter = TaskAdapter(task=task, model="anthropic/claude-sonnet-4-20250514")
        dataset = adapter.get_sample_ids()
    """

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
        """
        Initialize the adapter with an existing Inspect Task.

        Args:
            task: The Inspect Task to wrap (must have dataset, solver, scorer)
            model: Model identifier for evaluation (e.g., "anthropic/claude-sonnet-4-20250514")
            score_aggregator: Optional function to aggregate multiple scores to float.
                Defaults to using the first scorer's result.
            feedback_generator: Optional function to generate feedback strings.
                Defaults to a simple format showing input, completion, target, and score.
            log_dir: Optional directory for Inspect logs
            model_roles: Optional mapping of role names to model identifiers
        """
        self.task = task
        self.model = model
        self.score_aggregator = score_aggregator or scoring.first_scorer_as_float
        self.feedback_generator = (
            feedback_generator or scoring.default_feedback_generator
        )
        self.log_dir = log_dir
        self.model_roles = model_roles or {}

        # Build sample index from task dataset
        self._sample_index: dict[SampleId, inspect_ai.dataset.Sample] = {}
        self._sample_ids: list[SampleId] = []
        self._build_sample_index()

    def _build_sample_index(self) -> None:
        """Build an index of samples by their IDs for efficient lookup."""
        for i, sample in enumerate(self.task.dataset):
            sample_id: SampleId = sample.id if sample.id is not None else i
            self._sample_index[sample_id] = sample
            self._sample_ids.append(sample_id)

    def get_sample_ids(self) -> list[SampleId]:
        """Return all sample IDs for GEPA to sample from."""
        return list(self._sample_ids)

    def evaluate(
        self,
        batch: list[SampleId],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[InspectTrajectory, InspectOutput]:
        """
        Run evaluation on a batch of sample IDs using Inspect AI.

        Args:
            batch: List of sample IDs to evaluate
            candidate: Dict containing "system_prompt" key with the prompt to test
            capture_traces: Whether to capture full trajectories

        Returns:
            EvaluationBatch with outputs, scores, and optionally trajectories
        """
        system_prompt = candidate.get("system_prompt", "")

        # Get samples for the batch
        samples = [
            self._sample_index[sid] for sid in batch if sid in self._sample_index
        ]
        if not samples:
            return self._create_empty_batch(batch, capture_traces)

        # Create a new task with the batch samples and prepended system_message
        eval_task = self._create_eval_task(samples, system_prompt)

        # Run evaluation
        model_roles_resolved = (
            {
                role: inspect_ai.model.get_model(mid)
                for role, mid in self.model_roles.items()
            }
            if self.model_roles
            else None
        )
        results = scoring.run_inspect_eval(
            task=eval_task,
            model=self.model,
            log_dir=self.log_dir,
            model_roles=model_roles_resolved,
        )

        return self._process_results(results, batch, capture_traces)

    def _create_eval_task(
        self,
        samples: list[inspect_ai.dataset.Sample],
        system_prompt: str,
    ) -> inspect_ai.Task:
        """
        Create a new Task for evaluation with the system prompt prepended.

        This creates a new task that doesn't mutate the original task.
        The system_message solver is chained before the original solver.
        """
        # Get the original solver(s) and build new chain with system_message prepended
        original_solver = self.task.solver
        if isinstance(original_solver, list):
            solver_chain: list[inspect_ai.solver.Solver] = [
                inspect_ai.solver.system_message(template=system_prompt),
                *original_solver,
            ]
        else:
            solver_chain = [
                inspect_ai.solver.system_message(template=system_prompt),
                original_solver,
            ]

        return inspect_ai.Task(
            dataset=inspect_ai.dataset.MemoryDataset(samples),
            solver=solver_chain,
            scorer=self.task.scorer,
            sandbox=self.task.sandbox,
            message_limit=self.task.message_limit,
        )

    def _process_results(
        self,
        eval_results: list[Any],  # list[inspect_ai.log.EvalLog]
        batch: list[SampleId],
        capture_traces: bool,
    ) -> EvaluationBatch[InspectTrajectory, InspectOutput]:
        """Process Inspect evaluation results into GEPA's EvaluationBatch format."""
        outputs: list[InspectOutput] = []
        scores: list[float] = []
        trajectories: list[InspectTrajectory] | None = [] if capture_traces else None

        # Handle empty or failed results
        if not eval_results or eval_results[0].samples is None:
            return self._create_empty_batch(batch, capture_traces)

        eval_log = eval_results[0]
        samples_by_id: dict[Any, Any] = {s.id: s for s in eval_log.samples}

        for sample_id in batch:
            eval_sample = samples_by_id.get(sample_id)
            original_sample = self._sample_index.get(sample_id)

            if eval_sample is None or original_sample is None:
                output = InspectOutput(
                    completion="",
                    scores={},
                    error=f"Sample {sample_id} not found in results",
                )
                outputs.append(output)
                scores.append(0.0)
                if trajectories is not None:
                    trajectories.append(
                        self._create_failed_trajectory(
                            sample_id, f"Sample {sample_id} not found"
                        )
                    )
                continue

            # Extract completion and scores
            completion = scoring.get_completion_from_sample(eval_sample)
            sample_scores: dict[str, inspect_ai.scorer.Score] = eval_sample.scores or {}

            output = InspectOutput(completion=completion, scores=sample_scores)
            outputs.append(output)

            # Compute aggregated score
            score = self.score_aggregator(sample_scores)
            scores.append(score)

            # Create trajectory if capturing traces
            if trajectories is not None:
                # Get target from original sample
                target = original_sample.target

                # Generate feedback
                input_text = self._format_input(original_sample.input)
                feedback = self.feedback_generator(
                    input_text, completion, target, sample_scores, score
                )

                trajectory = InspectTrajectory(
                    sample_id=sample_id,
                    input=original_sample.input,
                    target=target,
                    messages=list(eval_sample.messages) if eval_sample.messages else [],
                    completion=completion,
                    scores=sample_scores,
                    score=score,
                    feedback=feedback,
                )
                trajectories.append(trajectory)

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
        """Create an empty/failed batch result."""
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
        """Create a trajectory representing a failed evaluation."""
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
        """Format input data as a string for feedback generation."""
        if isinstance(input_data, str):
            return input_data
        # For chat messages, concatenate content
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
        """
        Build the reflective dataset for GEPA instruction refinement.

        This creates a dataset that can be used by the teacher LLM to
        generate improved prompts based on evaluation results.

        Args:
            candidate: The candidate prompt configuration
            eval_batch: The evaluation batch results
            components_to_update: List of components being updated

        Returns:
            Mapping from component name to sequence of examples
        """
        if eval_batch.trajectories is None:
            return {comp: [] for comp in components_to_update}

        examples: list[Mapping[str, Any]] = []
        for traj in eval_batch.trajectories:
            input_text = (
                traj.input
                if isinstance(traj.input, str)
                else self._format_input(traj.input)
            )
            target_text = format_target(traj.target)

            # Use GEPA's recommended schema (see gepa.core.adapter.GEPAAdapter)
            example: dict[str, Any] = {
                "Inputs": input_text,
                "Generated Outputs": traj.completion,
                "Feedback": traj.feedback,
                # Additional keys (allowed per GEPA docs)
                "target": target_text,
                "score": traj.score,
                "scores": {k: str(v.value) for k, v in traj.scores.items()},
            }
            examples.append(example)

        # Map the same examples to all components being updated
        return {comp: examples for comp in components_to_update}
