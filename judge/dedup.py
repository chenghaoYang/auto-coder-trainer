"""Result deduplication for experiment judge."""

import hashlib
import json
from typing import Any


def compute_config_hash(recipe: dict[str, Any]) -> str:
    """Compute a deterministic hash of the experiment configuration.

    Serialises the recipe to JSON with sorted keys (excluding
    volatile / non-reproducibility-relevant fields like timestamps
    and output paths) and returns the hex SHA-256 digest.
    """
    # Strip fields that don't affect experiment identity
    excluded_keys = {"output_dir", "timestamp", "id", "recipe_id", "run_id"}
    canonical = {k: v for k, v in recipe.items() if k not in excluded_keys}
    serialised = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def find_duplicates(config_hash: str, result_db: Any) -> list[dict[str, Any]]:
    """Find duplicate experiments in the result DB by config hash.

    Returns a (possibly empty) list of experiment dicts that share the
    same configuration hash.
    """
    return result_db.find_by_config_hash(config_hash)
