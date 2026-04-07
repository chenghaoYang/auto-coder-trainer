"""Tests for SSD launcher bundle generation."""
import json
from pathlib import Path

import pytest


def _ssd_recipe_config(tmp_path: Path) -> dict:
    """Return a minimal compiled recipe config for SSD."""
    return {
        "recipe_id": "recipe-ssd-test-001",
        "trainer_type": "ssd",
        "backend": "ssd",
        "model_config": {
            "base": "Qwen/Qwen3-Coder-8B",
            "adapter": "full",
        },
        "data_config": {
            "sources": [],
        },
        "training_params": {
            "sample_temperature": 0.9,
            "sample_top_p": 0.8,
            "sample_top_k": 20,
            "n_samples_per_problem": 10,
            "max_tokens": 65536,
            "epochs": 1,
            "lr": 2e-5,
            "batch_size": 1,
            "decode_temperature": 0.6,
            "n_repeat": 20,
            "eval_max_tokens": 32768,
            "tensor_parallel_size": 4,
        },
        "eval_config": {
            "benchmarks": ["livecodebench-v6"],
            "seeds": [42],
        },
        "budget": {},
    }


def test_build_ssd_bundle_creates_all_scripts(tmp_path: Path):
    """build_ssd_launcher_bundle should produce all required bundle files."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    # Verify all expected files exist
    assert Path(paths["sample_script"]).exists()
    assert Path(paths["train_script"]).exists()
    assert Path(paths["eval_script"]).exists()
    assert Path(paths["import_results_script"]).exists()
    assert Path(paths["env_script"]).exists()
    assert Path(paths["launcher_json"]).exists()

    # Verify scripts are executable
    for key in ("sample_script", "train_script", "eval_script", "import_results_script"):
        assert Path(paths[key]).stat().st_mode & 0o111  # executable bit set


def test_build_ssd_bundle_scripts_are_valid_bash(tmp_path: Path):
    """Generated scripts should start with shebang and have error handling."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    for key in ("sample_script", "train_script", "eval_script", "import_results_script"):
        content = Path(paths[key]).read_text()
        assert content.startswith("#!/usr/bin/env bash"), f"{key} missing shebang"


def test_build_ssd_bundle_launcher_json(tmp_path: Path):
    """launcher.json should contain correct metadata."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    launcher = json.loads(Path(paths["launcher_json"]).read_text())
    assert launcher["backend"] == "ssd"
    assert launcher["recipe_id"] == "recipe-ssd-test-001"
    assert launcher["trainer_type"] == "ssd"


def test_build_ssd_bundle_contains_config_files(tmp_path: Path):
    """Bundle should contain JSON configs for each stage."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    assert Path(paths["sample_config"]).exists()
    assert Path(paths["eval_config"]).exists()

    sample_cfg = json.loads(Path(paths["sample_config"]).read_text())
    assert sample_cfg["model_name"] == "Qwen/Qwen3-Coder-8B"
    assert sample_cfg["temperature"] == 0.9


def test_ssd_bundle_end_to_end(tmp_path: Path):
    """Full bundle generation → write → verify all artifacts → results import."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle
    from trainers.ssd.results_bridge import import_results

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    # Verify bundle directory structure
    bundle_dir = Path(paths["bundle_dir"])
    assert bundle_dir.exists()

    # Verify sample config has correct model
    sample_cfg = json.loads(Path(paths["sample_config"]).read_text())
    assert sample_cfg["model_name"] == "Qwen/Qwen3-Coder-8B"
    assert sample_cfg["temperature"] == 0.9
    assert sample_cfg["n_samples"] == 10

    # Verify eval config
    eval_cfg = json.loads(Path(paths["eval_config"]).read_text())
    assert eval_cfg["n_repeat"] == 20
    assert eval_cfg["sampling_params"]["temperature"] == 0.6

    # Verify launcher.json has no internal keys
    launcher = json.loads(Path(paths["launcher_json"]).read_text())
    assert "_sample_config" not in launcher
    assert "_train_config" not in launcher

    # Simulate results import (no eval results yet → failed)
    payload = import_results(
        bundle_dir=bundle_dir,
        recipe_id="recipe-ssd-test-001",
        experiment_id="exp-integration",
    )
    assert payload["train_result"]["status"] == "failed"
    assert payload["recipe_id"] == "recipe-ssd-test-001"

    # Simulate successful eval results
    eval_results = {
        "pass@1": 0.45,
        "pass@5": 0.72,
        "num_total": 100,
        "num_repeat": 20,
    }
    (bundle_dir / "eval_results.json").write_text(json.dumps(eval_results))

    payload = import_results(
        bundle_dir=bundle_dir,
        recipe_id="recipe-ssd-test-001",
        experiment_id="exp-integration-2",
    )
    assert payload["eval_results"][0]["metrics"]["pass@1"] == 0.45


def test_ssd_launcher_registered():
    """SSDLauncher should be discoverable via the trainer registry."""
    from trainers.registry import get_trainer_class

    cls = get_trainer_class("ssd", "ssd")
    assert cls is not None

    from trainers.ssd.launcher import SSDLauncher
    assert cls is SSDLauncher


def test_ssd_launcher_in_module_all():
    """SSDLauncher should be exported from trainers.__init__."""
    import trainers

    assert "SSDLauncher" in trainers.__all__
    assert hasattr(trainers, "SSDLauncher")
