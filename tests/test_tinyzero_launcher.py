import json
from argparse import Namespace
from pathlib import Path

from cli.train import run_train
from recipes.compiler import compile_recipe, load_schema, validate_recipe
from results.db import ResultDB
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


def test_train_tinyzero_with_slurm_submits_and_tracks_jobs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import trainers.slurm.submitter as submitter

    recipe = _tinyzero_sft_recipe()
    recipe["budget"] = {
        "slurm": {
            "partition": "gpu",
            "nodes": 1,
            "gpus_per_node": 1,
            "cpus_per_task": 8,
            "mem": "64G",
            "time": "01:00:00",
        }
    }
    recipe_path = tmp_path / "tinyzero-slurm.recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))

    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    monkeypatch.setattr(
        submitter,
        "run_single_script_pipeline",
        lambda bundle_dir, slurm_cfg, backend, script_name="run.sh", stage="train": {
            "pipeline_id": "pipe-tinyzero-1",
            "job_ids": {"train": "90001"},
            "bundle_dir": str(bundle_dir),
        },
    )

    output_dir = tmp_path / "outputs"
    run_train(
        Namespace(
            recipe=str(recipe_path),
            output_dir=str(output_dir),
            dry_run=False,
            no_submit=False,
            import_results=None,
        ),
    )

    captured = capsys.readouterr().out
    assert "TinyZero SLURM pipeline submitted" in captured
    assert "Status     : running" in captured

    db = ResultDB(db_path)
    db.connect()
    try:
        experiments = db.find_by_recipe(recipe["id"])
        assert len(experiments) == 1
        assert experiments[0]["status"] == "running"
        jobs = db.get_slurm_jobs(recipe_id=recipe["id"])
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "90001"
        assert jobs[0]["stage"] == "train"
    finally:
        db.close()
