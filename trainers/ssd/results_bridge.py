"""Results bridge for SSD launcher bundles.

Parses launcher bundle artifacts written by ``trainers.ssd.launcher``
and converts them into the canonical import payload expected by ``cli.train``.

Follows the same pattern as ``trainers.tinyzero.results_bridge``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _coerce_numeric_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _find_checkpoint_path(bundle_dir: Path) -> str | None:
    """Find the most recent model checkpoint in the bundle."""
    model_dir = bundle_dir / "model"
    if model_dir.exists() and (model_dir / "config.json").exists():
        return str(model_dir)
    return None


def _parse_train_result(bundle_dir: Path) -> dict[str, Any]:
    """Parse training status from bundle artifacts."""
    model_path = _find_checkpoint_path(bundle_dir)
    if model_path is not None:
        return {
            "status": "success",
            "metrics": {},
            "checkpoint_path": model_path,
            "error": None,
        }

    sample_path = bundle_dir / "sample_data.jsonl"
    if sample_path.exists():
        return {
            "status": "failed",
            "metrics": {},
            "checkpoint_path": None,
            "error": "Sampling completed but training did not produce a checkpoint",
        }

    return {
        "status": "failed",
        "metrics": {},
        "checkpoint_path": None,
        "error": "No training artifacts found",
    }


def _parse_eval_results(
    bundle_dir: Path,
    *,
    recipe_id: str,
) -> list[dict[str, Any]]:
    """Parse evaluation results from eval_results.json."""
    payload = _load_json_if_exists(bundle_dir / "eval_results.json")
    if not payload:
        return []

    metrics = _coerce_numeric_metrics(payload)
    benchmark_metrics = {
        k: v for k, v in metrics.items()
        if k.startswith("pass@") and not k.startswith("pass@_")
    }

    details = {
        "num_total": payload.get("num_total"),
        "num_repeat": payload.get("num_repeat"),
    }

    return [
        {
            "recipe_id": recipe_id,
            "benchmark": "livecodebench-v6",
            "metrics": benchmark_metrics,
            "seed": 42,
            "details": details,
        }
    ]


def import_results(
    bundle_dir: str | Path,
    recipe_id: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Import SSD launcher artifacts into a canonical result payload."""
    bundle_dir = Path(bundle_dir)
    train_result = _parse_train_result(bundle_dir)
    eval_results = _parse_eval_results(
        bundle_dir,
        recipe_id=recipe_id,
    )

    return {
        "experiment_id": experiment_id,
        "recipe_id": recipe_id,
        "train_result": {
            "recipe_id": recipe_id,
            "trainer_type": "ssd",
            "backend": "ssd",
            "status": train_result["status"],
            "metrics": train_result["metrics"],
            "checkpoint_path": train_result["checkpoint_path"],
            "error": train_result["error"],
        },
        "eval_results": eval_results,
    }
