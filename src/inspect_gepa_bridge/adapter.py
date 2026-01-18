"""
Generic GEPA adapter base class for Inspect AI tasks.

This module provides a base class that handles the common patterns for
integrating Inspect AI tasks with GEPA optimization. Task-specific adapters
should extend this class and implement the abstract methods.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

import inspect_ai
import inspect_ai.dataset
import inspect_ai.model
import inspect_ai.scorer
import inspect_ai.solver
from gepa.core.adapter import EvaluationBatch

# Generic type variables for task-specific types
DataInst = TypeVar("DataInst")
Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")


class InspectGEPAAdapter(ABC, Generic[DataInst, Trajectory, RolloutOutput]):
    """
    Abstract base class for GEPA adapters that use Inspect AI for evaluation.

    This adapter provides the common pattern:
    1. Convert DataInst instances to Inspect samples
    2. Create an Inspect Task with the candidate's system prompt
    3. Run inspect_ai.eval() programmatically
    4. Process results into GEPA's EvaluationBatch format

    Subclasses must implement:
    - inst_to_sample(): Convert task-specific DataInst to Inspect Sample
    - get_scorers(): Return the list of Inspect scorers to use
    - get_solver_steps(): Return the solver steps (after system_message)
    - process_sample_result(): Convert Inspect sample result to RolloutOutput
    - compute_score(): Compute the GEPA score from the RolloutOutput
    - create_trajectory(): Create a Trajectory from the results
    - make_reflective_dataset(): Build the reflective dataset for GEPA
    """

    def __init__(
        self,
        agent_model: str,
        sandbox_type: str = "docker",
        log_dir: str | None = None,
        model_roles: dict[str, str] | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            agent_model: Model identifier for the agent (e.g., "anthropic/claude-sonnet-4-20250514")
            sandbox_type: Sandbox type for Inspect (e.g., "docker", "local")
            log_dir: Optional directory for Inspect logs
            model_roles: Optional mapping of role names to model identifiers
        """
        self.agent_model = agent_model
        self.sandbox_type = sandbox_type
        self.log_dir = log_dir
        self.model_roles = model_roles or {}

    @abstractmethod
    def inst_to_sample(self, inst: DataInst) -> inspect_ai.dataset.Sample:
        """Convert a task-specific DataInst to an Inspect Sample."""
        ...

    @abstractmethod
    def get_scorers(self) -> list[inspect_ai.scorer.Scorer]:
        """Return the list of Inspect scorers to use for evaluation."""
        ...

    @abstractmethod
    def get_solver_steps(
        self, candidate: dict[str, str]
    ) -> list[inspect_ai.solver.Solver]:
        """
        Return the solver steps to use after the system message.

        The system_message solver is added automatically from candidate["system_prompt"].
        This method should return any additional solvers (prompt_template, generate, etc.)
        """
        ...

    @abstractmethod
    def process_sample_result(
        self,
        inst: DataInst,
        sample: Any,  # inspect_ai.log.EvalSample
    ) -> RolloutOutput:
        """Convert an Inspect sample result to the task-specific RolloutOutput."""
        ...

    @abstractmethod
    def compute_score(self, output: RolloutOutput) -> tuple[float, str]:
        """
        Compute the GEPA score and feedback from a RolloutOutput.

        Returns:
            Tuple of (score, feedback_string)
        """
        ...

    @abstractmethod
    def create_trajectory(
        self,
        inst: DataInst,
        output: RolloutOutput,
        score: float,
        feedback: str,
    ) -> Trajectory:
        """Create a Trajectory object from the evaluation results."""
        ...

    @abstractmethod
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build the reflective dataset for GEPA instruction refinement.

        This is called by GEPA to get examples for the teacher LLM.
        """
        ...

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """
        Run evaluation on a batch of instances using Inspect AI.

        This is the main entry point called by GEPA.
        """
        system_prompt = candidate.get("system_prompt", "")

        # Convert instances to Inspect samples
        samples = [self.inst_to_sample(inst) for inst in batch]
        dataset = inspect_ai.dataset.MemoryDataset(samples)

        # Build solver chain
        solvers: list[inspect_ai.solver.Solver] = [
            inspect_ai.solver.system_message(template=system_prompt),
            *self.get_solver_steps(candidate),
        ]

        # Create the task
        task = inspect_ai.Task(
            dataset=dataset,
            solver=solvers,
            scorer=self.get_scorers(),
            sandbox=self.sandbox_type,
        )

        # Set up eval kwargs
        model = inspect_ai.model.get_model(self.agent_model)
        eval_kwargs: dict[str, Any] = {"model": model}

        if self.log_dir:
            eval_kwargs["log_dir"] = self.log_dir

        # Add model roles if configured
        if self.model_roles:
            eval_kwargs["model_roles"] = {
                role: inspect_ai.model.get_model(model_id)
                for role, model_id in self.model_roles.items()
            }

        # Run evaluation
        results = inspect_ai.eval(task, **eval_kwargs)

        return self._process_results(results, batch, capture_traces)

    def _process_results(
        self,
        eval_results: list[Any],  # list[inspect_ai.log.EvalLog]
        batch: list[DataInst],
        capture_traces: bool,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """Process Inspect evaluation results into GEPA's EvaluationBatch format."""
        outputs: list[RolloutOutput] = []
        scores: list[float] = []
        trajectories: list[Trajectory] | None = [] if capture_traces else None

        # Handle empty or failed results
        if not eval_results or eval_results[0].samples is None:
            return self._create_empty_batch(batch, capture_traces)

        eval_log = eval_results[0]
        samples_by_id = {s.id: s for s in eval_log.samples}

        for inst in batch:
            sample_id = self._get_sample_id(inst)
            sample = samples_by_id.get(sample_id)

            if sample is None:
                output = self._create_failed_output(
                    inst, f"Sample {sample_id} not found"
                )
                outputs.append(output)
                scores.append(0.0)
                if trajectories is not None:
                    trajectories.append(
                        self.create_trajectory(
                            inst, output, 0.0, f"Sample {sample_id} not found"
                        )
                    )
                continue

            # Process the sample result
            output = self.process_sample_result(inst, sample)
            outputs.append(output)

            # Compute score and feedback
            score, feedback = self.compute_score(output)
            scores.append(score)

            # Create trajectory if capturing traces
            if trajectories is not None:
                trajectories.append(
                    self.create_trajectory(inst, output, score, feedback)
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None,
        )

    def _get_sample_id(self, inst: DataInst) -> Any:
        """Get the sample ID from a DataInst. Override if needed."""
        return getattr(inst, "sample_id", None)

    def _create_empty_batch(
        self,
        batch: list[DataInst],
        capture_traces: bool,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """Create an empty/failed batch result."""
        outputs: list[RolloutOutput] = []
        scores: list[float] = []
        trajectories: list[Trajectory] | None = [] if capture_traces else None

        for inst in batch:
            output = self._create_failed_output(inst, "Eval failed: no results")
            outputs.append(output)
            scores.append(0.0)
            if trajectories is not None:
                trajectories.append(
                    self.create_trajectory(inst, output, 0.0, "Eval failed: no results")
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None,
        )

    @abstractmethod
    def _create_failed_output(self, inst: DataInst, error: str) -> RolloutOutput:
        """Create a RolloutOutput representing a failed evaluation."""
        ...
