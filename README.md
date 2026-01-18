# inspect-gepa-bridge

Generic bridge for integrating [Inspect AI](https://inspect.ai-safety-institute.org.uk/) tasks with [GEPA](https://github.com/gepa-ai/gepa) (Genetic Evolution for Prompt Adaptation) optimization.

## Overview

This package provides utilities and base classes for creating GEPA adapters that use Inspect AI for evaluation. It handles the common patterns:

- Converting task-specific data instances to Inspect samples
- Running Inspect evaluations programmatically
- Processing scorer results into GEPA's EvaluationBatch format
- Utilities for building reflective datasets

## Installation

```bash
pip install inspect-gepa-bridge
```

Or with uv:

```bash
uv add inspect-gepa-bridge
```

## Usage

Create a task-specific adapter by extending `InspectGEPAAdapter`:

```python
from dataclasses import dataclass
from inspect_gepa_bridge import InspectGEPAAdapter, DataInstBase
import inspect_ai.dataset
import inspect_ai.scorer
import inspect_ai.solver

@dataclass
class MyDataInst(DataInstBase):
    question: str
    expected_answer: str

@dataclass
class MyTrajectory:
    sample_id: int | str
    completion: str
    score: float
    feedback: str

@dataclass
class MyRolloutOutput:
    completion: str
    score: inspect_ai.scorer.Score | None

class MyGEPAAdapter(InspectGEPAAdapter[MyDataInst, MyTrajectory, MyRolloutOutput]):
    def inst_to_sample(self, inst: MyDataInst) -> inspect_ai.dataset.Sample:
        return inspect_ai.dataset.Sample(
            input=inst.question,
            id=inst.sample_id,
            target=inst.expected_answer,
        )

    def get_scorers(self) -> list[inspect_ai.scorer.Scorer]:
        return [inspect_ai.scorer.match()]

    def get_solver_steps(self, candidate: dict[str, str]) -> list[inspect_ai.solver.Solver]:
        return [inspect_ai.solver.generate()]

    # ... implement remaining abstract methods
```

## Components

### `InspectGEPAAdapter`

Abstract base class that handles the common evaluation pattern. Subclasses implement task-specific logic.

### `InspectSampleConverter`

Abstract converter for transforming Inspect Samples to task-specific data instances.

### `load_inspect_dataset`

Utility function for loading Inspect datasets and converting them to task-specific formats.

### Scoring Utilities

- `ScorerResult`: Container for scorer results with convenient accessors
- `score_to_float`: Convert Inspect Score to float value
- `extract_scores_from_sample`: Extract specific scorer results from evaluation
- `get_completion_from_sample`: Extract completion text from sample results

## License

MIT
