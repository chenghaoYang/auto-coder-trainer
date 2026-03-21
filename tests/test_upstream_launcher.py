import json
from argparse import Namespace
from pathlib import Path

from cli.train import run_train
from recipes.compiler import compile_recipe, load_schema, validate_recipe
from trainers.upstream import build_upstream_launcher_bundle, write_upstream_launcher_bundle


def _redi_recipe() -> dict:
    return {
        "id": "recipe-redi-distill-001",
        "name": "REDI refinement for coding agents",
        "model": {
            "base": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "adapter": "lora",
        },
        "dataset": {
            "sources": [
                {
                    "name": "teacher-agent-trajectories",
                    "path": "data/distill/pairs.jsonl",
                }
            ]
        },
        "trainer": {
            "type": "distill",
            "backend": "redi",
            "params": {
                "lr": 2e-5,
                "epochs": 1,
                "batch_size": 2,
            },
        },
        "distill": {
            "strategy": "trajectory",
            "teacher_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "teacher_mode": "offline_dataset",
            "stages": ["positive_sft", "pairwise_refine"],
            "refine_algorithm": "redi",
            "pairwise_beta": 0.1,
        },
        "eval": {
            "benchmarks": ["swe-bench-lite"],
            "seeds": [42],
        },
    }


def test_redi_backend_is_schema_valid_and_writes_upstream_bundle(tmp_path: Path) -> None:
    recipe = _redi_recipe()
    schema = load_schema()
    assert validate_recipe(recipe, schema) == []

    config = compile_recipe(recipe)
    bundle = build_upstream_launcher_bundle(config.__dict__, tmp_path)
    paths = write_upstream_launcher_bundle(bundle)

    launcher = json.loads(Path(paths["launcher_json"]).read_text())
    run_script = Path(paths["run_script"]).read_text()
    env_template = Path(paths["env"]).read_text()
    notes = Path(paths["notes"]).read_text()

    assert launcher["backend"] == "redi"
    assert "ACT_REDI_STAGE2_SCRIPT" in env_template
    assert "git clone" in run_script
    assert "REDI" in notes


def test_train_prepares_redi_bundle_and_execution_plan(tmp_path: Path, capsys) -> None:
    recipe = _redi_recipe()
    recipe_path = tmp_path / "redi.recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))

    output_dir = tmp_path / "outputs"
    run_train(
        Namespace(recipe=str(recipe_path), output_dir=str(output_dir), dry_run=False),
    )

    captured = capsys.readouterr().out
    assert "REDI backend selected" in captured
    assert "Status     : prepared" in captured

    plan_dir = output_dir / recipe["id"]
    plan = json.loads((plan_dir / "execution-plan.json").read_text())
    assert plan["mode"] == "prepared"
    assert plan["launcher"]["backend"] == "redi"
    assert Path(plan["launcher"]["files"]["run_script"]).exists()
    assert Path(plan["launcher"]["files"]["notes"]).exists()
