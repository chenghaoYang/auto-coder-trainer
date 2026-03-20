"""Ablation registry and validation for experiment judge."""

import json
from typing import Any


def validate_ablation_coverage(recipe: dict[str, Any], result_db: Any) -> list[str]:
    """Check which ablation experiments are missing.

    Reads the ablation spec from recipe["ablation"] (a list of dicts with
    "variable" and "values" keys) and cross-references against ablation
    records stored in the result DB for matching experiments.

    Returns a list of missing ablation descriptions (e.g. "lr=0.001").
    An empty list means full coverage.
    """
    ablation_spec = recipe.get("ablation", [])
    if not ablation_spec:
        return []

    recipe_id = recipe.get("id", recipe.get("recipe_id", ""))
    experiments = result_db.find_by_recipe(recipe_id)
    experiment_ids = {e["id"] for e in experiments}

    # Collect all (variable, value) pairs already recorded
    recorded: set[tuple[str, str]] = set()
    for exp_id in experiment_ids:
        exp = result_db.get_experiment(exp_id)
        if exp is None:
            continue
        # Look up ablations for this experiment via the DB
        # We query ablations table directly through the connection
        if hasattr(result_db, '_conn') and result_db._conn is not None:
            cursor = result_db._conn.execute(
                "SELECT variable, value FROM ablations WHERE experiment_id = ?",
                (exp_id,),
            )
            for row in cursor.fetchall():
                recorded.add((row[0], row[1]))

    # Determine which ablations are missing
    missing: list[str] = []
    for spec in ablation_spec:
        variable = spec.get("variable", "")
        values = spec.get("values", [])
        for val in values:
            val_str = json.dumps(val) if not isinstance(val, str) else val
            if (variable, val_str) not in recorded:
                missing.append(f"{variable}={val_str}")

    return missing


def register_ablation(
    experiment_id: str,
    variable: str,
    value: Any,
    result_db: Any,
    metrics: dict[str, float] | None = None,
) -> None:
    """Register an ablation result in the result DB.

    Inserts a row into the ablations table linking the ablation to its
    parent experiment.
    """
    value_str = json.dumps(value) if not isinstance(value, str) else value
    metrics_json = json.dumps(metrics) if metrics else None

    result_db.insert_ablation({
        "experiment_id": experiment_id,
        "variable": variable,
        "value": value_str,
        "metrics_json": metrics_json,
    })
