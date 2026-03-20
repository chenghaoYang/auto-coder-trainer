"""pass@k metric computation."""

import math
from typing import Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric.

    Args:
        n: Total number of samples generated per problem.
        c: Number of correct samples per problem.
        k: k in pass@k.

    Returns:
        pass@k probability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def compute_pass_at_k(results: Sequence[dict], k_values: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """Compute pass@k for multiple k values across all problems.

    Args:
        results: List of dicts, each with keys ``problem_id``,
            ``n_samples`` (total generated), and ``n_correct``.
        k_values: k values to compute.

    Returns:
        Dict mapping ``"pass@{k}"`` to the mean pass@k across problems.

    Raises:
        ValueError: If *results* is empty or a k value exceeds the
            number of samples for any problem.
    """
    if not results:
        raise ValueError("results must be a non-empty sequence")

    aggregated: dict[str, float] = {}
    for k in k_values:
        scores: list[float] = []
        for entry in results:
            n = entry["n_samples"]
            c = entry["n_correct"]
            if k > n:
                raise ValueError(
                    f"k={k} exceeds n_samples={n} for problem "
                    f"'{entry.get('problem_id', '?')}'"
                )
            scores.append(pass_at_k(n, c, k))
        aggregated[f"pass@{k}"] = sum(scores) / len(scores)
    return aggregated
