"""Base evaluator interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Standardized benchmark evaluation result."""
    benchmark: str
    metrics: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    num_samples: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    def evaluate(self, model_path: str, seed: int = 42) -> BenchmarkResult:
        """Evaluate a model checkpoint and return standardized results."""
        ...

    @abstractmethod
    def get_benchmark_name(self) -> str:
        """Return the canonical benchmark name."""
        ...
