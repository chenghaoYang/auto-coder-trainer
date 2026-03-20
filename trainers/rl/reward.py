"""Reward function registry for RL training.

Supports composable reward functions that can be specified in Recipe IR.
Each reward function scores a trajectory (a dict containing at minimum
``response``, ``tests_passed``, ``tests_total``, and optionally ``logprobs``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any


class BaseReward(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def compute(self, trajectory: dict[str, Any]) -> float:
        """Compute reward for a single trajectory.

        Args:
            trajectory: Must contain at least:
                - tests_passed (int): number of tests that passed
                - tests_total (int): total number of tests
                Optionally:
                - logprobs (list[float]): per-token log-probabilities
                - response (str): the generated text
        """
        ...


# ---------------------------------------------------------------------------
# Concrete reward functions
# ---------------------------------------------------------------------------

class BinaryPassReward(BaseReward):
    """Binary reward: 1.0 if all tests pass, 0.0 otherwise."""

    def compute(self, trajectory: dict[str, Any]) -> float:
        passed = trajectory.get("tests_passed", 0)
        total = trajectory.get("tests_total", 0)
        if total == 0:
            return 0.0
        return 1.0 if passed >= total else 0.0


class WeightedPassReward(BaseReward):
    """Proportional reward based on fraction of tests passed."""

    def compute(self, trajectory: dict[str, Any]) -> float:
        passed = trajectory.get("tests_passed", 0)
        total = trajectory.get("tests_total", 0)
        if total == 0:
            return 0.0
        return passed / total


class EntropyBonusReward(BaseReward):
    """Entropy bonus that encourages diverse (high-entropy) token distributions.

    Computes the average per-token entropy from ``logprobs`` and scales it by
    ``entropy_coeff``.  If ``logprobs`` are not available, returns 0.
    """

    def __init__(self, entropy_coeff: float = 0.01):
        self.entropy_coeff = entropy_coeff

    def compute(self, trajectory: dict[str, Any]) -> float:
        logprobs: list[float] | None = trajectory.get("logprobs")
        if not logprobs:
            return 0.0
        # Approximate per-token entropy: H ≈ -mean(logp)
        avg_neg_logp = -sum(logprobs) / len(logprobs)
        return self.entropy_coeff * avg_neg_logp


class EntropyAwareReward(BaseReward):
    """Task reward with an entropy-based penalty for degenerate (low-entropy) solutions.

    Combines a weighted-pass score with an entropy bonus:
        reward = task_score + entropy_coeff * avg_entropy
    """

    def __init__(self, entropy_coeff: float = 0.01):
        self.entropy_coeff = entropy_coeff
        self._task = WeightedPassReward()
        self._entropy = EntropyBonusReward(entropy_coeff)

    def compute(self, trajectory: dict[str, Any]) -> float:
        return self._task.compute(trajectory) + self._entropy.compute(trajectory)


class LengthPenaltyReward(BaseReward):
    """Penalises overly long responses.

    penalty = -coeff * max(0, len(response) - max_length) / max_length
    """

    def __init__(self, max_length: int = 16384, coeff: float = 0.1):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        self.max_length = max_length
        self.coeff = coeff

    def compute(self, trajectory: dict[str, Any]) -> float:
        response = trajectory.get("response", "")
        excess = max(0, len(response) - self.max_length)
        return -self.coeff * (excess / self.max_length)


class CompositeReward(BaseReward):
    """Weighted combination of multiple reward components.

    Final score = sum(weight_i * component_i.compute(trajectory))
    """

    def __init__(self, components: list[tuple[BaseReward, float]]):
        if not components:
            raise ValueError("CompositeReward requires at least one component")
        self.components = components

    def compute(self, trajectory: dict[str, Any]) -> float:
        return sum(weight * comp.compute(trajectory) for comp, weight in self.components)


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

REWARD_REGISTRY: dict[str, type[BaseReward]] = {
    "binary_pass": BinaryPassReward,
    "weighted_pass": WeightedPassReward,
    "entropy_bonus": EntropyBonusReward,
    "entropy_aware": EntropyAwareReward,
    "length_penalty": LengthPenaltyReward,
    "composite": CompositeReward,
}


def build_reward(reward_config: dict[str, Any]) -> BaseReward:
    """Build a reward function from a recipe reward config block.

    Config format (from recipe JSON)::

        {
            "type": "entropy_aware",            # or "composite", etc.
            "entropy_coeff": 0.01,              # type-specific kwargs
            "components": [                     # only for composite / component list
                {"type": "binary_pass", "weight": 0.7},
                {"type": "entropy_bonus", "weight": 0.3, "entropy_coeff": 0.01}
            ]
        }

    When ``components`` are present and ``type`` is **not** ``"composite"``,
    the components are automatically wrapped in a ``CompositeReward``.
    """
    reward_type = reward_config.get("type", "binary_pass")
    components_cfg = reward_config.get("components")

    # If explicit components are provided, build a CompositeReward regardless
    # of the top-level type (matches recipe convention).
    if components_cfg:
        parts: list[tuple[BaseReward, float]] = []
        for comp_cfg in components_cfg:
            comp_type = comp_cfg["type"]
            weight = comp_cfg.get("weight", 1.0)
            comp_cls = REWARD_REGISTRY.get(comp_type)
            if comp_cls is None:
                raise ValueError(f"Unknown reward component type: {comp_type!r}")
            # Extract kwargs for the component (exclude meta keys)
            comp_kwargs = {k: v for k, v in comp_cfg.items() if k not in ("type", "weight")}
            parts.append((comp_cls(**comp_kwargs), weight))
        return CompositeReward(parts)

    # Simple (non-composite) reward
    cls = REWARD_REGISTRY.get(reward_type)
    if cls is None:
        raise ValueError(f"Unknown reward type: {reward_type!r}")

    kwargs = {k: v for k, v in reward_config.items() if k not in ("type", "components")}
    return cls(**kwargs)
