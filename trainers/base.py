"""Base trainer interface for all training backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainResult:
    """Result of a training run."""
    recipe_id: str
    trainer_type: str
    backend: str
    status: str  # "success", "failed", "timeout"
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    error: str | None = None


@dataclass
class EvalResult:
    """Result of an evaluation run."""
    recipe_id: str
    benchmark: str
    metrics: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    details: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    Subclasses must implement prepare_data(), train(), and evaluate().
    A trainer is initialized with a compiled recipe config and output directory.
    """

    def __init__(self, config: dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = output_dir

    @abstractmethod
    def prepare_data(self) -> Any:
        """Load and preprocess training data according to recipe config."""
        ...

    @abstractmethod
    def train(self) -> TrainResult:
        """Execute the training run. Returns TrainResult."""
        ...

    @abstractmethod
    def evaluate(self, checkpoint_path: str, benchmark: str, seed: int = 42) -> EvalResult:
        """Evaluate a checkpoint on a benchmark. Returns EvalResult."""
        ...

    def run(self) -> tuple[TrainResult, list[EvalResult]]:
        """Full pipeline: prepare → train → evaluate on all benchmarks."""
        self.prepare_data()
        train_result = self.train()

        eval_results = []
        if train_result.status == "success" and train_result.checkpoint_path:
            for benchmark in self.config.get("eval_config", {}).get("benchmarks", []):
                for seed in self.config.get("eval_config", {}).get("seeds", [42]):
                    eval_result = self.evaluate(train_result.checkpoint_path, benchmark, seed)
                    eval_results.append(eval_result)

        return train_result, eval_results
