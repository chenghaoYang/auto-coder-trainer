import json
from argparse import Namespace
from pathlib import Path

from cli.train import run_train
from recipes.compiler import compile_recipe, load_schema, validate_recipe
from trainers.tinyzero import build_tinyzero_launcher_bundle, write_tinyzero_launcher_bundle


def _tinyzero_sft_recipe() -> dict:
    return {
        "id": "recipe-tinyzero-sft-001",
        "name": "TinyZero SFT baseline",
        "model": {
            "base": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "adapter": "full",
        },
        "dataset": {
            "sources": [
                {
                    "name": "swe-traj",
                    "path": "bigcode/swe-bench-trajectories",
                }
            ]
        },
        "trainer": {
            "type": "sft",
            "backend": "tinyzero",
            "params": {
                "lr": 2e-5,
                "epochs": 2,
                "batch_size": 4,
            },
        },
        "eval": {
            "benchmarks": ["humaneval"],
            "seeds": [42],
        },
    }


def test_tinyzero_backend_is_schema_valid_and_writes_bundle(tmp_path: Path) -> None:
    recipe = _tinyzero_sft_recipe()
    schema = load_schema()
    assert validate_recipe(recipe, schema) == []

    config = compile_recipe(recipe)
    bundle = build_tinyzero_launcher_bundle(config.__dict__, tmp_path)
    paths = write_tinyzero_launcher_bundle(bundle)

    launcher = json.loads(Path(paths["launcher_json"]).read_text())
    overrides = Path(paths["hydra_overrides"]).read_text()
    run_script = Path(paths["run_script"]).read_text()
    env_template = Path(paths["env"]).read_text()

    assert launcher["entrypoint"]["module"] == "verl.trainer.fsdp_sft_trainer"
    assert "data.train_files=${ACT_TRAIN_FILE}" in overrides
    assert "torchrun --standalone" in run_script
    assert "ACT_TRAIN_FILE" in env_template


def test_train_prepares_tinyzero_bundle_and_execution_plan(tmp_path: Path, capsys) -> None:
    recipe = _tinyzero_sft_recipe()
    recipe_path = tmp_path / "tinyzero.recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))

    output_dir = tmp_path / "outputs"
    run_train(
        Namespace(recipe=str(recipe_path), output_dir=str(output_dir), dry_run=False),
    )

    captured = capsys.readouterr().out
    assert "TinyZero backend selected" in captured
    assert "Status     : prepared" in captured

    plan_dir = output_dir / recipe["id"]
    plan = json.loads((plan_dir / "execution-plan.json").read_text())
    assert plan["mode"] == "prepared"
    assert plan["launcher"]["backend"] == "tinyzero"
    assert Path(plan["launcher"]["files"]["run_script"]).exists()
