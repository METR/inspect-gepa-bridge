# inspect-gepa-bridge

Generic bridge for integrating [Inspect AI](https://inspect.ai-safety-institute.org.uk/) tasks with [GEPA](https://github.com/gepa-ai/gepa) (Genetic Evolution for Prompt Adaptation) optimization.

## Overview

This package provides utilities for wrapping existing Inspect AI tasks for GEPA optimization. The primary API is `TaskAdapter`, which takes an existing Inspect Task and makes it compatible with GEPA's optimization interface.

## Installation

```bash
pip install inspect-gepa-bridge
```

Or with uv:

```bash
uv add inspect-gepa-bridge
```

## Quick Start

Wrap an existing Inspect Task with `TaskAdapter`:

```python
from inspect_ai import Task
from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import model_graded_fact

from inspect_gepa_bridge import TaskAdapter

# Your existing Inspect task
task = Task(
    dataset=hf_dataset("gsm8k", split="train[:100]"),
    solver=generate(),
    scorer=model_graded_fact(),
)

# Wrap for GEPA - that's it!
adapter = TaskAdapter(task=task, model="anthropic/claude-sonnet-4-20250514")

# Get sample IDs for GEPA to sample from
dataset = adapter.get_sample_ids()

# Evaluate a batch of samples with a candidate prompt
result = adapter.evaluate(
    batch=dataset[:10],
    candidate={"system_prompt": "You are a helpful math tutor."},
    capture_traces=True,
)
```

## Components

### `TaskAdapter`

The primary adapter class that wraps an existing Inspect Task. It:
- Builds a sample index from the task's dataset
- Replaces any existing system messages with the GEPA-optimized prompt
- Runs evaluation using Inspect AI
- Returns results in GEPA's EvaluationBatch format

**System Message Handling:** When evaluating with a candidate prompt, `TaskAdapter` removes any existing system messages from the task's solver chain and inserts the GEPA-optimized system prompt. This ensures the model receives exactly one system message - the one being optimized by GEPA.

```python
adapter = TaskAdapter(
    task=my_task,
    model="anthropic/claude-sonnet-4-20250514",
    score_aggregator=my_aggregator,  # Optional: custom score aggregation
    feedback_generator=my_feedback,  # Optional: custom feedback generation
    log_dir="/path/to/logs",         # Optional: Inspect log directory
    model_roles={"grader": "..."},   # Optional: model roles for scoring
)
```

### Types

- `InspectTrajectory`: Full trajectory data including input, output, messages, and scores
- `InspectOutput`: Simple output container with completion and scores
- `ScoreAggregator`: Protocol for aggregating multiple scores to a single float
- `FeedbackGenerator`: Protocol for generating feedback strings

### Scoring Utilities

- `first_scorer_as_float`: Default score aggregator (uses first scorer)
- `default_feedback_generator`: Default feedback generator
- `score_to_float`: Convert Inspect Score to float value
- `ScorerResult`: Container for scorer results with convenient accessors

## Advanced Usage

### Custom Score Aggregation

```python
from inspect_ai.scorer import Score

from inspect_gepa_bridge import TaskAdapter, first_scorer_as_float, score_to_float

def my_aggregator(scores: dict[str, Score]) -> float:
    # Weight different scorers
    if "accuracy" in scores and "style" in scores:
        return 0.8 * score_to_float(scores["accuracy"]) + 0.2 * score_to_float(scores["style"])
    return first_scorer_as_float(scores)

adapter = TaskAdapter(task=task, model=model, score_aggregator=my_aggregator)
```

### Custom Feedback Generation

```python
def my_feedback(input_text, completion, target, scores, score) -> str:
    return f"Score: {score:.2f}\nExpected: {target}\nGot: {completion}"

adapter = TaskAdapter(task=task, model=model, feedback_generator=my_feedback)
```

## License

MIT
