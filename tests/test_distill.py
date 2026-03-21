import json
from argparse import Namespace
from pathlib import Path

import pytest

from cli.train import run_train
from recipes.compiler import load_schema, validate_recipe
from results.db import ResultDB
from trainers.base import EvalResult, TrainResult
from trainers.distill.data import load_distillation_data


def _distill_recipe(dataset_path: str) -> dict:
    return {
        "id": "recipe-trajectory-distill-001",
        "name": "Trajectory Distillation for Coding Agents",
        "version": "1.0",
        "source_papers": ["2505.17612", "2505.24850"],
        "model": {
            "base": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "adapter": "lora",
        },
        "dataset": {
            "sources": [
                {
                    "name": "teacher-agent-trajectories",
                    "path": dataset_path,
                    "mix_weight": 1.0,
                }
            ],
            "filters": [
                {"type": "quality_score", "params": {"min_score": 0.6}},
            ],
        },
        "trainer": {
            "type": "distill",
            "backend": "trl",
            "params": {
                "lr": 2e-5,
                "epochs": 1,
                "batch_size": 2,
                "pairwise_epochs": 1,
                "pairwise_lr": 5e-6,
            },
        },
        "distill": {
            "strategy": "trajectory",
            "teacher_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "teacher_mode": "offline_dataset",
            "stages": ["positive_sft", "pairwise_refine"],
            "pairwise_beta": 0.1,
            "condense": {
                "strategy": "edge_preserving",
                "max_chars": 120,
                "head_chars": 40,
                "tail_chars": 40,
            },
        },
        "eval": {
            "benchmarks": ["swe-bench-lite"],
            "metrics": ["resolve_rate", "pass@1"],
            "seeds": [42, 123, 456],
        },
        "ablation": [],
        "budget": {
            "max_gpu_hours": 12,
            "gpu_type": "A100-80GB",
        },
    }


def test_distill_example_recipe_is_schema_valid() -> None:
    recipe_path = Path("recipes/examples/trajectory-distill.recipe.json")
    schema = load_schema()
    recipe = json.loads(recipe_path.read_text())

    assert validate_recipe(recipe, schema) == []


def test_load_distillation_data_builds_positive_and_pair_examples(tmp_path: Path) -> None:
    dataset_path = tmp_path / "distill.jsonl"
    long_rejected = "retry " * 80
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "Fix the failing parser test",
                        "chosen": "Apply the parser patch and run pytest -q.",
                        "rejected": long_rejected,
                        "quality_score": 0.9,
                    }
                ),
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "You are a coding assistant."},
                            {"role": "user", "content": "Write a regression test."},
                            {"role": "assistant", "content": "Add a focused pytest covering the bug."},
                        ],
                        "quality_score": 0.8,
                    }
                ),
            ]
        )
    )

    payload = load_distillation_data(
        [{"name": "teacher", "path": str(dataset_path)}],
        filters=[{"type": "quality_score", "params": {"min_score": 0.7}}],
        distill_config={
            "trace_template": "chatml",
            "condense": {
                "strategy": "edge_preserving",
                "max_chars": 120,
                "head_chars": 40,
                "tail_chars": 40,
            },
        },
    )

    assert len(payload["positive_examples"]) == 2
    assert len(payload["pair_examples"]) == 1
    pair = payload["pair_examples"][0]
    assert "Assistant:" in pair["prompt_text"]
    assert "[... condensed trajectory middle ...]" in pair["rejected_output"]


def test_train_dispatches_distill_trainer_and_persists_results(tmp_path: Path, monkeypatch, capsys) -> None:
    import trainers.distill

    dataset_path = tmp_path / "distill.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "prompt": "Fix the parser bug",
                "chosen": "Patch the parser and run pytest -q.",
                "rejected": "Guess at a fix without tests.",
                "quality_score": 0.9,
            }
        )
    )
    recipe = _distill_recipe(str(dataset_path))
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))
    db_path = tmp_path / "results.db"
    checkpoint_dir = tmp_path / "fake-distill-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.safetensors").write_text("stub")
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    class FakeDistillTrainer:
        def __init__(self, config, output_dir):
            self.config = config
            self.output_dir = output_dir

        def run(self):
            return (
                TrainResult(
                    recipe_id=self.config["recipe_id"],
                    trainer_type="distill",
                    backend="trl",
                    status="success",
                    metrics={"positive/train_loss": 0.4, "pairwise/train_accuracy": 0.75},
                    checkpoint_path=str(checkpoint_dir),
                ),
                [
                    EvalResult(
                        recipe_id=self.config["recipe_id"],
                        benchmark="swe-bench-lite",
                        seed=seed,
                        metrics={"resolve_rate": 0.61, "pass@1": 0.33},
                        details={"source": "fake"},
                    )
                    for seed in (42, 123, 456)
                ],
            )

    monkeypatch.setattr(trainers.distill, "DistillTrainer", FakeDistillTrainer)

    output_dir = tmp_path / "outputs"
    run_train(Namespace(recipe=str(recipe_path), output_dir=str(output_dir), dry_run=False))

    captured = capsys.readouterr().out
    assert "Using distillation trainer" in captured
    assert "Judge verdict: accept" in captured
    assert "Status     : success" in captured

    db = ResultDB(db_path)
    db.connect()
    try:
        experiments = db.find_by_recipe(recipe["id"])
        assert len(experiments) == 1
        experiment = experiments[0]
        assert experiment["trainer_type"] == "distill"
        assert experiment["status"] == "success"
        assert experiment["metrics_json"]["resolve_rate"] == pytest.approx(0.61)
    finally:
        db.close()
