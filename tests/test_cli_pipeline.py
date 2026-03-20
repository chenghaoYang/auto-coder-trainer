import json
from argparse import Namespace
from pathlib import Path

import cli.compose as compose_mod
from cli.collect import run_collect
from cli.compose import run_compose
from cli.train import run_train
from recipes.compiler import load_schema, validate_recipe


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


def test_train_writes_execution_plan_when_trainer_unimplemented(tmp_path: Path, capsys) -> None:
    recipe = {
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
            "benchmarks": ["swe-bench-lite", "humaneval"],
            "metrics": ["resolve_rate", "pass@1"],
            "seeds": [42, 123, 456],
        },
        "ablation": [],
        "budget": {
            "max_gpu_hours": 24,
            "gpu_type": "A100-80GB",
        },
    }

    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))

    output_dir = tmp_path / "outputs"
    run_train(
        Namespace(recipe=str(recipe_path), output_dir=str(output_dir), dry_run=False),
    )

    captured = capsys.readouterr().out
    assert "Execution plan written" in captured
    assert "Status     : blocked" in captured

    plan_dir = output_dir / recipe["id"]
    plan_json = json.loads((plan_dir / "execution-plan.json").read_text())
    assert plan_json["recipe_id"] == recipe["id"]
    assert plan_json["mode"] == "blocked"
    assert plan_json["eval"]["benchmarks"] == ["swe-bench-lite", "humaneval"]
    assert "training backend is not ready" in plan_json["reason"].lower()
