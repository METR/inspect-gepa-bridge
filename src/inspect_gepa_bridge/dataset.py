"""
Dataset utilities for converting Inspect AI datasets to GEPA format.

This module provides utilities for loading and converting Inspect AI datasets
into formats suitable for GEPA optimization.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import inspect_ai.dataset

DataInst = TypeVar("DataInst")


@dataclass
class DataInstBase:
    """
    Base class for task-specific data instances.

    Subclasses should add task-specific fields while keeping sample_id
    for tracking purposes.
    """

    sample_id: int | str


class InspectSampleConverter(ABC, Generic[DataInst]):
    """
    Abstract converter from Inspect Samples to task-specific DataInst objects.

    Subclasses implement the conversion logic for their specific task format.
    """

    @abstractmethod
    def convert(self, sample: inspect_ai.dataset.Sample) -> DataInst:
        """Convert an Inspect Sample to a task-specific DataInst."""
        ...


def load_inspect_dataset(
    dataset_source: str | inspect_ai.dataset.Dataset,
    converter: InspectSampleConverter[DataInst],
    split: str = "train",
    limit: int | None = None,
    sample_fields: Callable[..., inspect_ai.dataset.Sample] | None = None,
    features: Any = None,
) -> list[DataInst]:
    """
    Load an Inspect dataset and convert it to a list of task-specific DataInst objects.

    Args:
        dataset_source: Either a HuggingFace dataset slug (str) or an Inspect Dataset
        converter: Converter to transform Inspect Samples to DataInst objects
        split: Dataset split to load (default: "train")
        limit: Maximum number of samples to load (default: None for all)
        sample_fields: Optional function to convert raw records to Samples
        features: Optional HuggingFace dataset features specification

    Returns:
        List of task-specific DataInst objects
    """
    # Load the Inspect dataset
    if isinstance(dataset_source, str):
        kwargs: dict[str, Any] = {"split": split}
        if sample_fields is not None:
            kwargs["sample_fields"] = sample_fields
        if features is not None:
            kwargs["features"] = features
        inspect_dataset = inspect_ai.dataset.hf_dataset(dataset_source, **kwargs)
    else:
        inspect_dataset = dataset_source

    # Convert samples
    instances: list[DataInst] = []
    for i, sample in enumerate(inspect_dataset):
        if limit is not None and i >= limit:
            break
        instances.append(converter.convert(sample))

    return instances


def extract_sample_metadata(
    sample: inspect_ai.dataset.Sample,
    key: str,
    default: Any = None,
) -> Any:
    """
    Safely extract a value from sample metadata.

    Args:
        sample: The Inspect Sample
        key: Metadata key to extract
        default: Default value if key is not found

    Returns:
        The metadata value or default
    """
    metadata = sample.metadata or {}
    return metadata.get(key, default)
