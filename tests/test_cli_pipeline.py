import json
from argparse import Namespace
from pathlib import Path

import pytest

import cli.compose as compose_mod
import cli.collect as collect_mod
from cli.collect import run_collect
from cli.compose import run_compose
from cli.status import run_status
from cli.train import run_train
from recipes.compiler import load_schema, validate_recipe
from results.db import ResultDB
from trainers.base import EvalResult, TrainResult


def _baseline_recipe() -> dict:
    return {
        "id": "recipe-baseline-sft-001",
        "name": "Baseline SFT on SWE-bench Trajectories",
        "version": "1.0",
        "source_papers": ["2410.01021"],
        "model": {
            "base": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "size": "7B",
            "adapter": "lora",
        },
        "dataset": {
            "sources": [
                {
                    "name": "swe-bench-trajectories",
                    "path": "bigcode/swe-bench-trajectories",
                    "mix_weight": 1.0,
                }
            ],
            "filters": [
                {"type": "quality_score", "params": {"min_score": 0.7}},
            ],
            "total_samples": 10000,
        },
        "trainer": {
            "type": "sft",
            "backend": "trl",
            "params": {
                "lr": 2e-5,
                "epochs": 3,
                "batch_size": 4,
            },
        },
        "eval": {
            "benchmarks": ["swe-bench-lite"],
            "metrics": ["resolve_rate", "pass@1"],
            "seeds": [42, 123, 456],
        },
        "ablation": [],
        "budget": {
            "max_gpu_hours": 24,
            "gpu_type": "A100-80GB",
        },
    }


def test_collect_imports_atoms_from_json_file(tmp_path: Path) -> None:
    source = tmp_path / "atoms.json"
    source.write_text(
        json.dumps(
            {
                "atoms": [
                    {
                        "name": "swe-fuse",
                        "source_papers": ["2410.01021"],
                        "dataset": {
                            "sources": [
                                {"name": "swe-bench", "path": "bigcode/swe-bench"}
                            ]
                        },
                        "trainer": {"params": {"lr": 1e-5}},
                    }
                ]
            }
        )
    )

    output_dir = tmp_path / "registry_out"
    run_collect(
        Namespace(query=str(source), max_papers=5, output=str(output_dir)),
    )

    registry_file = output_dir / "method_atoms.json"
    payload = json.loads(registry_file.read_text())
    assert [atom["name"] for atom in payload["atoms"]] == ["swe-fuse"]


def test_collect_discovers_online_atoms_and_records_collection_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        collect_mod,
        "_search_arxiv_papers",
        lambda query, max_papers: [
            {
                "id": "2501.12345",
                "title": "Entropy Guided SWE-Bench Training",
                "abstract": "Entropy-aware reward shaping for coding agents on SWE-Bench.",
            }
        ],
    )
    monkeypatch.setattr(
        collect_mod,
        "_search_github_repos",
        lambda query, max_repos: [
            {
                "name": "agent-lab",
                "full_name": "demo/agent-lab",
                "html_url": "https://github.com/demo/agent-lab",
                "description": "SWE-Bench training harness with sandboxed evaluation",
                "stargazers_count": 128,
                "license": "MIT",
            }
        ],
    )

    output_dir = tmp_path / "registry_out"
    run_collect(
        Namespace(query="coding agent training", max_papers=5, max_repos=5, output=str(output_dir)),
    )

    registry_file = output_dir / "method_atoms.json"
    payload = json.loads(registry_file.read_text())
    assert len(payload["atoms"]) == 2
    assert payload["last_collection"]["query"] == "coding agent training"
    assert payload["collections"][0]["papers_found"] == 1
    assert payload["collections"][0]["repos_found"] == 1


def test_compose_outputs_schema_clean_recipe(tmp_path: Path, monkeypatch) -> None:
    registry_file = tmp_path / "method_atoms.json"
    registry_file.write_text(json.dumps({"atoms": []}))
    monkeypatch.setattr(compose_mod, "REGISTRY_PATH", registry_file)

    output_file = tmp_path / "composed.recipe.json"
    run_compose(
        Namespace(atoms="missing-atom", model="Qwen/Qwen2.5-Coder-7B-Instruct", output=str(output_file)),
    )

    recipe = json.loads(output_file.read_text())
    schema = load_schema()
    assert validate_recipe(recipe, schema) == []
    assert "size" not in recipe["model"]
    assert "total_samples" not in recipe["dataset"]
    assert recipe["trainer"]["type"] == "sft"


def test_train_dry_run_writes_execution_plan_and_task_ledger(tmp_path: Path, monkeypatch, capsys) -> None:
    recipe = _baseline_recipe()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    output_dir = tmp_path / "outputs"
    run_train(
        Namespace(recipe=str(recipe_path), output_dir=str(output_dir), dry_run=True),
    )

    captured = capsys.readouterr().out
    assert "Execution plan written" in captured

    plan_dir = output_dir / recipe["id"]
    plan_json = json.loads((plan_dir / "execution-plan.json").read_text())
    assert plan_json["recipe_id"] == recipe["id"]
    assert plan_json["mode"] == "dry_run"
    assert plan_json["eval"]["benchmarks"] == ["swe-bench-lite"]
    assert (plan_dir / "task-ledger.json").exists()

    db = ResultDB(db_path)
    db.connect()
    try:
        experiments = db.find_by_recipe(recipe["id"])
        assert len(experiments) == 1
        assert experiments[0]["status"] == "planned"
        tasks = db.get_tasks(recipe_id=recipe["id"])
        assert any(task["kind"] == "compile_recipe" for task in tasks)
        artifacts = db.get_artifacts_for_recipe(recipe["id"])
        assert any(artifact["kind"] == "execution_plan_json" for artifact in artifacts)
    finally:
        db.close()


def test_train_persists_successful_run_eval_results_and_tasks(tmp_path: Path, monkeypatch, capsys) -> None:
    import trainers.sft

    recipe = _baseline_recipe()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))
    db_path = tmp_path / "results.db"
    checkpoint_dir = tmp_path / "fake-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_text("stub")
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
                    metrics={"train/loss": 0.25},
                    checkpoint_path=str(checkpoint_dir),
                ),
                [
                    EvalResult(
                        recipe_id=self.config["recipe_id"],
                        benchmark="swe-bench-lite",
                        seed=seed,
                        metrics={"resolve_rate": 0.8, "pass@1": 0.5},
                        details={"source": "fake"},
                    )
                    for seed in (42, 123, 456)
                ],
            )

    monkeypatch.setattr(trainers.sft, "SFTTrainer", FakeSFTTrainer)

    output_dir = tmp_path / "outputs"
    run_train(
        Namespace(recipe=str(recipe_path), output_dir=str(output_dir), dry_run=False),
    )

    captured = capsys.readouterr().out
    assert "Judge verdict: accept" in captured
    assert "Status     : success" in captured

    db = ResultDB(db_path)
    db.connect()
    try:
        experiments = db.find_by_recipe(recipe["id"])
        assert len(experiments) == 1
        experiment = experiments[0]
        assert experiment["status"] == "success"
        assert experiment["metrics_json"]["resolve_rate"] == pytest.approx(0.8)
        eval_runs = db.get_eval_runs_for_experiment(experiment["id"])
        assert len(eval_runs) == 3
        assert {row["seed"] for row in eval_runs} == {42, 123, 456}
        tasks = db.get_tasks(recipe_id=recipe["id"], experiment_id=experiment["id"])
        assert any(task["kind"] == "record_judge_verdict" and task["status"] == "completed" for task in tasks)
        assert any(task["kind"] == "generate_report" and task["status"] == "pending" for task in tasks)
        artifacts = db.get_artifacts_for_experiment(experiment["id"])
        assert any(artifact["kind"] == "checkpoint" for artifact in artifacts)
        assert any(artifact["kind"] == "task_ledger_json" for artifact in artifacts)
    finally:
        db.close()

    plan_dir = output_dir / recipe["id"]
    ledger = json.loads((plan_dir / "task-ledger.json").read_text())
    assert ledger["status"] == "success"
    assert ledger["latest_verdict"] == "accept"


def test_status_reports_open_tasks(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    db = ResultDB(db_path)
    db.connect()
    try:
        db.insert_experiment(
            {
                "id": "exp-1",
                "recipe_id": "recipe-demo",
                "config_hash": "hash-1",
                "status": "prepared",
                "trainer_type": "sft",
                "backend": "tinyzero",
                "model_base": "demo-model",
                "metrics_json": {},
                "train_metrics_json": {},
                "recipe_json": {"id": "recipe-demo"},
                "budget_json": {},
                "checkpoint_path": None,
                "error": None,
            }
        )
        db.upsert_task(
            {
                "id": "task-1",
                "recipe_id": "recipe-demo",
                "experiment_id": "exp-1",
                "kind": "execution_step",
                "title": "Launch external TinyZero bundle",
                "status": "pending",
                "priority": "high",
                "payload_json": {},
                "notes": None,
            }
        )
    finally:
        db.close()

    output_path = tmp_path / "status.md"
    run_status(Namespace(recipe_id="recipe-demo", open_only=True, output=str(output_path)))

    captured = capsys.readouterr().out
    assert "Launch external TinyZero bundle" in captured
    assert output_path.exists()
    assert "Tracked Experiments" in output_path.read_text()
    monkeypatch.delenv("ACT_RESULTS_DB", raising=False)
