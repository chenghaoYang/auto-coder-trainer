"""Tests for SSD results bridge."""
import json
from pathlib import Path

import pytest


def test_import_results_parses_eval_json(tmp_path: Path):
    """import_results should parse eval_results.json into canonical payload."""
    from trainers.ssd.results_bridge import import_results

    eval_results = {
        "pass@1": 0.45,
        "pass@5": 0.72,
        "pass@1_easy": 0.55,
        "pass@1_medium": 0.40,
        "pass@1_hard": 0.25,
        "num_total": 100,
        "num_repeat": 20,
    }
    eval_path = tmp_path / "eval_results.json"
    eval_path.write_text(json.dumps(eval_results))

    # Create model dir so train_result is "success"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    payload = import_results(
        bundle_dir=tmp_path,
        recipe_id="recipe-ssd-001",
        experiment_id="exp-001",
    )
    assert payload["experiment_id"] == "exp-001"
    assert payload["recipe_id"] == "recipe-ssd-001"
    assert payload["train_result"]["status"] == "success"
    assert "pass@1" in payload["eval_results"][0]["metrics"]


def test_import_results_handles_missing_eval(tmp_path: Path):
    """import_results should return failed status when eval_results.json missing."""
    from trainers.ssd.results_bridge import import_results

    payload = import_results(
        bundle_dir=tmp_path,
        recipe_id="recipe-ssd-001",
        experiment_id="exp-001",
    )
    assert payload["train_result"]["status"] == "failed"
    assert len(payload["eval_results"]) == 0


def test_import_results_handles_empty_metrics(tmp_path: Path):
    """import_results should handle eval_results.json with no pass@k keys."""
    from trainers.ssd.results_bridge import import_results

    eval_path = tmp_path / "eval_results.json"
    eval_path.write_text(json.dumps({"num_total": 0}))

    payload = import_results(
        bundle_dir=tmp_path,
        recipe_id="recipe-ssd-002",
        experiment_id="exp-002",
    )
    assert payload["eval_results"][0]["metrics"] == {}
