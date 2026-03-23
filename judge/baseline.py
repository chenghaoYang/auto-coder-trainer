"""Baseline alignment utilities for experiment judge."""

import json
from typing import Any


def find_baseline(recipe: dict[str, Any], result_db: Any) -> dict[str, Any] | None:
    """Find the matching baseline experiment for a given recipe.

    Queries the result DB for experiments matching the same recipe_id with
    status 'success', then returns the one with the best primary metric
    (resolve_rate by default).
    """
    recipe_id = recipe.get("id", recipe.get("recipe_id", ""))
    experiments = result_db.find_by_recipe(recipe_id)
    if not experiments:
        return None

    # Filter to successful experiments only
    successful = [e for e in experiments if e.get("status") == "success"]
    if not successful:
        return None

    # Pick the experiment with the best primary metric (resolve_rate)
    def _primary_metric(exp: dict[str, Any]) -> float:
        metrics = exp.get("metrics_json", {})
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        return metrics.get("resolve_rate", 0.0)

    best = max(successful, key=_primary_metric)
    return best


def compute_delta(
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute metric deltas between current and baseline.

    Returns a dict with keys like:
      - "<metric>_delta": absolute difference (current - baseline)
      - "<metric>_relative_pct": relative % change vs baseline

    Only metrics present in *both* dicts are compared.
    """
    deltas: dict[str, float] = {}
    common_keys = set(current_metrics) & set(baseline_metrics)

    for key in sorted(common_keys):
        cur = current_metrics[key]
        base = baseline_metrics[key]
        abs_delta = cur - base
        deltas[f"{key}_delta"] = abs_delta
        if base != 0.0:
            deltas[f"{key}_relative_pct"] = (abs_delta / abs(base)) * 100.0
        else:
            deltas[f"{key}_relative_pct"] = float("inf") if abs_delta != 0 else 0.0

    return deltas
