"""Custom reward functions for TinyZero experiments 5 and 21.

Experiment 5: Reward function design comparison
  - binary:  0/1
  - partial: 0/0.3/1.0  (format bonus)
  - process: step-level rewards

Experiment 21: Reward-shaped thinking
  - result_only:   baseline, only final answer
  - step_bonus:    reward detailed thinking
  - step_penalty:  reward concise thinking
  - clever:        reward efficient/clever solutions

Usage in verl:
  These are monkey-patched into verl.utils.reward_score.gsm8k at runtime.
  See run_experiments.sh for how REWARD_TYPE env var selects the function.
"""

from __future__ import annotations

import re
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Experiment 5: Reward function design
# ---------------------------------------------------------------------------

def compute_score_binary(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """Binary reward: 0 wrong, 1 correct."""
    answer = _extract_answer(solution_str)
    if answer is None:
        return 0.0
    return 1.0 if answer == ground_truth else 0.0


def compute_score_partial(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """Partial reward: 0 bad format, 0.3 correct format wrong answer, 1.0 correct."""
    answer = _extract_answer(solution_str)
    if answer is None:
        return 0.0
    return 1.0 if answer == ground_truth else 0.3


def compute_score_process(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """Process reward: bonus for intermediate reasoning steps."""
    answer = _extract_answer(solution_str)
    steps = re.findall(r"\d+\s*[+\-*/=]\s*\d+", solution_str)
    step_reward = min(len(steps) * 0.05, 0.3)

    if answer is None:
        return step_reward * 0.5
    if answer == ground_truth:
        return 1.0 + step_reward
    return 0.1 + step_reward


# ---------------------------------------------------------------------------
# Experiment 21: Reward-shaped thinking style
# ---------------------------------------------------------------------------

def compute_score_result_only(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """Baseline: only care about final answer (same as binary)."""
    return compute_score_binary(solution_str, ground_truth)


def compute_score_step_bonus(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """Slow thinking: reward detailed step-by-step reasoning."""
    answer = _extract_answer(solution_str)
    if answer is None:
        return 0.0

    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    n_steps = 0
    if think_match:
        lines = [l for l in think_match.group(1).strip().split("\n") if l.strip()]
        n_steps = len(lines)

    bonus = min(n_steps * 0.1, 0.5)
    if answer == ground_truth:
        return 1.0 + bonus
    return bonus * 0.3


def compute_score_step_penalty(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """Fast thinking: penalize verbose reasoning."""
    answer = _extract_answer(solution_str)
    if answer is None:
        return 0.0

    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    n_steps = 0
    if think_match:
        lines = [l for l in think_match.group(1).strip().split("\n") if l.strip()]
        n_steps = len(lines)

    penalty = n_steps * 0.05
    if answer == ground_truth:
        return max(0.1, 1.0 - penalty)
    return 0.0


def compute_score_clever(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """Clever thinking: reward efficient mathematical tricks."""
    answer = _extract_answer(solution_str)
    if answer is None:
        return 0.0

    tricks = [
        r"distributive|分配律",
        r"associative|结合律",
        r"factor|因式",
        r"\d+\s*[×x]\s*\(\s*\d+\s*[+\-]\s*\d+\s*\)",
    ]
    trick_count = sum(
        1 for t in tricks if re.search(t, solution_str, re.IGNORECASE)
    )
    trick_bonus = min(trick_count * 0.15, 0.3)

    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    n_tokens = len(think_match.group(1).split()) if think_match else 0
    efficiency_bonus = max(0, 0.2 - n_tokens * 0.002)

    if answer == ground_truth:
        return 1.0 + trick_bonus + efficiency_bonus
    return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_answer(solution_str: str) -> str | None:
    match = re.search(r"####\s*(\-?[0-9\.\,]+)", solution_str)
    if match is None:
        return None
    return match.group(1).replace(",", "").replace("$", "")


REWARD_REGISTRY: dict[str, Callable[..., float]] = {
    # Experiment 5
    "binary": compute_score_binary,
    "partial": compute_score_partial,
    "process": compute_score_process,
    # Experiment 21
    "result_only": compute_score_result_only,
    "step_bonus": compute_score_step_bonus,
    "step_penalty": compute_score_step_penalty,
    "clever": compute_score_clever,
}


def get_reward_fn(name: str = "binary") -> Callable[..., float]:
    return REWARD_REGISTRY.get(name, compute_score_binary)


def patch_verl_reward(reward_type: str = "binary") -> None:
    """Monkey-patch verl's GSM8K reward function at runtime."""
    import verl.utils.reward_score.gsm8k as gsm8k_mod

    fn = get_reward_fn(reward_type)
    gsm8k_mod.compute_score = fn
    print(f"[reward] Patched verl GSM8K reward -> {reward_type} ({fn.__name__})")
