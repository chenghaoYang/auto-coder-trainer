"""Regression tests for the dialogue-driven closed-loop workflow."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import cli.compose as compose_mod
import trainers.swe_lego.results_bridge as swe_bridge
from cli.compose import run_compose
from cli.pipeline import _get_latest_verdict
from cli.rerun import _dispatch_run_ablation, run_rerun
from cli.train import run_train
from results.db import ResultDB
from trainers.base import EvalResult, TrainResult


def _write_trainer_state(log_dir: Path, loss: float = 0.5) -> None:
    state = {
        "epoch": 4.0,
        "global_step": 100,
        "log_history": [
            {"loss": 1.0, "learning_rate": 1e-4, "epoch": 1.0, "step": 25},
            {"loss": 0.8, "learning_rate": 8e-5, "epoch": 2.0, "step": 50},
            {"loss": 0.6, "learning_rate": 5e-5, "epoch": 3.0, "step": 75},
            {"loss": loss, "learning_rate": 2e-5, "epoch": 4.0, "step": 100},
        ],
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "trainer_state.json").write_text(json.dumps(state))


def _seed_experiment(db: ResultDB, recipe_id: str) -> None:
    db.insert_experiment(
        {
            "id": "exp-workflow-1",
            "recipe_id": recipe_id,
            "config_hash": "hash-workflow-1",
            "status": "success",
            "trainer_type": "sft",
            "backend": "swe_lego",
            "model_base": "Qwen/Qwen3.5-9B",
            "metrics_json": {},
            "train_metrics_json": {},
            "recipe_json": {"id": recipe_id},
            "budget_json": {},
            "checkpoint_path": None,
            "error": None,
        }
    )


def test_compose_keeps_reward_atoms_as_augmentations_and_applies_swe_lego_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_file = tmp_path / "method_atoms.json"
    registry_file.write_text(
        json.dumps(
            {
                "atoms": [
                    {
                        "name": "swe-fuse",
                        "category": "training",
                        "source_papers": ["2410.01021"],
                        "dataset": {
                            "sources": [
                                {
                                    "name": "swe-bench-trajectories",
                                    "path": "bigcode/swe-bench-trajectories",
                                    "mix_weight": 1.0,
                                }
                            ]
                        },
                        "trainer": {
                            "type": "sft",
                            "backend": "trl",
                            "params": {"lr": 2e-5},
                        },
                        "eval": {
                            "benchmarks": ["swe-bench-lite"],
                            "metrics": ["resolve_rate"],
                        },
                    },
                    {
                        "name": "entropy-rl",
                        "category": "reward",
                        "trainer": {
                            "type": "grpo",
                            "backend": "verl",
                            "reward": {"type": "entropy_aware"},
                        },
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(compose_mod, "REGISTRY_PATH", registry_file)

    output_file = tmp_path / "composed.recipe.json"
    run_compose(
        Namespace(
            atoms="swe-fuse,entropy-rl",
            model="Qwen/Qwen3.5-9B",
            trainer_type=None,
            backend=None,
            output=str(output_file),
        )
    )

    recipe = json.loads(output_file.read_text())
    assert recipe["trainer"]["type"] == "sft"
    assert recipe["trainer"]["backend"] == "swe_lego"
    assert recipe["trainer"]["reward"]["type"] == "entropy_aware"
    assert recipe["eval"]["benchmarks"] == ["swe-bench-verified"]
    assert recipe["budget"]["slurm"]["partition"] == "gpu"
    assert any(spec["variable"] == "trainer.params.lr" for spec in recipe["ablation"])


def test_pipeline_latest_verdict_exposes_research_suggestions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    db = ResultDB(db_path)
    db.connect()
    try:
        _seed_experiment(db, "recipe-workflow")
        db.insert_verdict(
            {
                "experiment_id": "exp-workflow-1",
                "verdict": "needs_rerun",
                "reasoning": "Need more evidence",
                "checks_json": {"seeds": False},
                "suggestions_json": ["Re-run evaluation with all required seeds"],
                "research_suggestions_json": [
                    {
                        "type": "research_queries",
                        "queries": [{"query": "coding agent distillation qwen3.5", "priority": 1}],
                        "trigger_collection": True,
                    }
                ],
            }
        )
    finally:
        db.close()

    verdict = _get_latest_verdict("recipe-workflow")
    assert verdict is not None
    assert verdict["research_suggestions"][0]["type"] == "research_queries"
    assert verdict["research_suggestions"][0]["trigger_collection"] is True


def test_pipeline_latest_verdict_prefers_newest_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    db = ResultDB(db_path)
    db.connect()
    try:
        db.insert_experiment(
            {
                "id": "exp-workflow-old",
                "recipe_id": "recipe-workflow",
                "config_hash": "hash-workflow-old",
                "status": "success",
                "trainer_type": "sft",
                "backend": "swe_lego",
                "model_base": "Qwen/Qwen3.5-9B",
                "metrics_json": {"resolve_rate": 24.0},
                "train_metrics_json": {},
                "recipe_json": {"id": "recipe-workflow"},
                "budget_json": {},
                "checkpoint_path": None,
                "error": None,
            }
        )
        db.insert_verdict(
            {
                "experiment_id": "exp-workflow-old",
                "verdict": "reject",
                "reasoning": "Older run regressed",
                "checks_json": {"baseline": False},
                "suggestions_json": ["Do not use the old run"],
                "research_suggestions_json": [],
            }
        )
        db.insert_experiment(
            {
                "id": "exp-workflow-new",
                "recipe_id": "recipe-workflow",
                "config_hash": "hash-workflow-new",
                "status": "success",
                "trainer_type": "sft",
                "backend": "swe_lego",
                "model_base": "Qwen/Qwen3.5-9B",
                "metrics_json": {"resolve_rate": 31.2},
                "train_metrics_json": {},
                "recipe_json": {"id": "recipe-workflow"},
                "budget_json": {},
                "checkpoint_path": None,
                "error": None,
            }
        )
        db.insert_verdict(
            {
                "experiment_id": "exp-workflow-new",
                "verdict": "accept",
                "reasoning": "Newest run passes all checks",
                "checks_json": {"baseline": True},
                "suggestions_json": [],
                "research_suggestions_json": [],
            }
        )
    finally:
        db.close()

    verdict = _get_latest_verdict("recipe-workflow")
    assert verdict is not None
    assert verdict["experiment_id"] == "exp-workflow-new"
    assert verdict["verdict"] == "accept"


def test_rerun_ablation_updates_nested_recipe_and_tags_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_recipes: list[dict] = []

    import cli.train as train_mod

    def _fake_run_train(args) -> None:
        captured_recipes.append(json.loads(Path(args.recipe).read_text()))

    monkeypatch.setattr(train_mod, "run_train", _fake_run_train)

    recipe = {
        "id": "recipe-ablation",
        "name": "Ablation Demo",
        "trainer": {
            "type": "sft",
            "backend": "swe_lego",
            "params": {"lr": 2e-5},
        },
        "ablation": [
            {
                "name": "lr_sweep",
                "variable": "trainer.params.lr",
                "values": [1e-5, 2e-5, 5e-5],
            }
        ],
    }

    success = _dispatch_run_ablation(
        {"kind": "run_ablation", "payload_json": {}},
        "recipe-ablation",
        recipe,
        db=None,
        dry_run=False,
    )

    assert success is True
    assert [item["trainer"]["params"]["lr"] for item in captured_recipes] == [1e-5, 2e-5, 5e-5]
    assert all(item["ablation_run"]["variable"] == "trainer.params.lr" for item in captured_recipes)


def test_rerun_ablation_targets_only_missing_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_recipes: list[dict] = []

    import cli.train as train_mod

    def _fake_run_train(args) -> None:
        captured_recipes.append(json.loads(Path(args.recipe).read_text()))

    monkeypatch.setattr(train_mod, "run_train", _fake_run_train)

    recipe = {
        "id": "recipe-ablation-targeted",
        "name": "Ablation Targeted Demo",
        "trainer": {
            "type": "sft",
            "backend": "swe_lego",
            "params": {"lr": 2e-5},
        },
        "ablation": [
            {
                "name": "lr_sweep",
                "variable": "trainer.params.lr",
                "values": [1e-5, 2e-5, 5e-5],
            }
        ],
    }

    success = _dispatch_run_ablation(
        {"kind": "run_ablation", "payload_json": {"targets": ["trainer.params.lr=2e-05"]}},
        "recipe-ablation-targeted",
        recipe,
        db=None,
        dry_run=False,
    )

    assert success is True
    assert len(captured_recipes) == 1
    assert captured_recipes[0]["trainer"]["params"]["lr"] == 2e-5


def test_rerun_leaves_execution_steps_blocked_for_manual_follow_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    db = ResultDB(db_path)
    db.connect()
    try:
        db.insert_experiment(
            {
                "id": "exp-external-1",
                "recipe_id": "recipe-external",
                "config_hash": "hash-external-1",
                "status": "prepared",
                "trainer_type": "sft",
                "backend": "swe_lego",
                "model_base": "Qwen/Qwen3.5-9B",
                "metrics_json": {},
                "train_metrics_json": {},
                "recipe_json": {"id": "recipe-external"},
                "budget_json": {},
                "checkpoint_path": None,
                "error": None,
            }
        )
        db.upsert_task(
            {
                "id": "task-external-1",
                "recipe_id": "recipe-external",
                "experiment_id": "exp-external-1",
                "kind": "execution_step",
                "title": "Submit SWE-Lego SLURM bundle",
                "status": "pending",
                "priority": "high",
                "payload_json": {"mode": "prepared"},
                "notes": None,
            }
        )
    finally:
        db.close()

    run_rerun(Namespace(recipe_id="recipe-external", dry_run=False))

    db = ResultDB(db_path)
    db.connect()
    try:
        task = db.get_tasks(recipe_id="recipe-external")[0]
        assert task["status"] == "blocked"
        assert "Awaiting manual/external execution." in task["notes"]
    finally:
        db.close()


def test_train_persists_ablation_rows_for_ablation_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trainers.registry
    import trainers.sft

    recipe = {
        "id": "recipe-ablation-train",
        "name": "Ablation Variant",
        "model": {"base": "Qwen/Qwen2.5-Coder-7B-Instruct", "adapter": "lora"},
        "dataset": {
            "sources": [{"name": "demo", "path": "demo.jsonl"}],
        },
        "trainer": {
            "type": "sft",
            "backend": "trl",
            "params": {"lr": 1e-5, "epochs": 1, "batch_size": 1},
        },
        "eval": {"benchmarks": ["swe-bench-lite"], "seeds": [42]},
        "ablation_run": {
            "parent_recipe_id": "recipe-ablation-train",
            "name": "lr_sweep",
            "variable": "trainer.params.lr",
            "value": 1e-5,
        },
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_text("stub")

    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    class FakeSFTTrainer:
        def __init__(self, config, output_dir):
            self.config = config
            self.output_dir = output_dir

        def run(self):
            return (
                TrainResult(
                    recipe_id=self.config["recipe_id"],
                    trainer_type="sft",
                    backend="trl",
                    status="success",
                    metrics={"train/loss": 0.2},
                    checkpoint_path=str(checkpoint_dir),
                ),
                [
                    EvalResult(
                        recipe_id=self.config["recipe_id"],
                        benchmark="swe-bench-lite",
                        seed=42,
                        metrics={"resolve_rate": 0.4},
                        details={"source": "fake"},
                    )
                ],
            )

    monkeypatch.setattr(trainers.sft, "SFTTrainer", FakeSFTTrainer)
    monkeypatch.setitem(trainers.registry._REGISTRY, ("sft", None), FakeSFTTrainer)
    monkeypatch.setitem(trainers.registry._REGISTRY, ("sft", "trl"), FakeSFTTrainer)

    run_train(Namespace(recipe=str(recipe_path), output_dir=str(tmp_path / "outputs"), dry_run=False))

    db = ResultDB(db_path)
    db.connect()
    try:
        experiment = db.find_by_recipe(recipe["id"])[0]
        ablations = db.get_ablations_for_experiment(experiment["id"])
        assert len(ablations) == 1
        assert ablations[0]["variable"] == "trainer.params.lr"
        assert ablations[0]["value"] == "1e-05"
    finally:
        db.close()


def test_swe_lego_import_results_collects_multiple_seed_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    saves_dir = bundle_dir / "saves" / "SWE-Lego-demo"
    _write_trainer_state(saves_dir, loss=0.33)
    (saves_dir / "config.json").write_text("{}")

    swe_lego_root = tmp_path / "mock-swe-lego"
    results_dir = swe_lego_root / "SWE-bench-4.0.4" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for seed, resolved in ((42, 80), (123, 82), (456, 78)):
        (results_dir / f"openhands_seed_{seed}.swe_bench.json").write_text(
            json.dumps(
                {
                    "resolved_ids": [f"instance_{idx}" for idx in range(resolved)],
                    "total_instances": 500,
                }
            )
        )

    monkeypatch.setattr(swe_bridge, "SWE_LEGO_ROOT", swe_lego_root)

    imported = swe_bridge.import_results(
        bundle_dir,
        recipe_id="recipe-demo",
        experiment_id="exp-demo",
        expected_seeds=[42, 123, 456],
    )

    seeds = [item["seed"] for item in imported["eval_results"]]
    assert seeds == [42, 123, 456]
    assert imported["eval_results"][1]["benchmark"] == "swe-bench-verified"
    assert imported["eval_results"][1]["metrics"]["resolve_rate"] == pytest.approx(82 / 500 * 100.0)
