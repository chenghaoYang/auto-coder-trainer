import json
from argparse import Namespace
from pathlib import Path

from recipes.compiler import compile_recipe, load_schema, validate_recipe
from trainers.swe_lego import build_swe_lego_launcher_bundle, write_swe_lego_launcher_bundle


def _swe_lego_8b_recipe() -> dict:
    return {
        "id": "recipe-swe-lego-8b-test",
        "name": "SWE-Lego 8B test",
        "model": {
            "base": "Qwen/Qwen3-8B",
            "size": "8B",
            "adapter": "full",
        },
        "dataset": {
            "sources": [
                {"name": "swe-lego-real", "path": "SWE-Lego/SWE-Lego-Real-Data"},
                {"name": "swe-lego-synthetic", "path": "SWE-Lego/SWE-Lego-Synthetic-Data"},
            ]
        },
        "trainer": {
            "type": "sft",
            "backend": "swe_lego",
            "params": {
                "lr": 1e-4,
                "epochs": 4,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_length": 131072,
                "warmup_ratio": 0.1,
                "lr_scheduler": "cosine",
                "turn_mask": True,
                "rope_scaling": "yarn",
                "flash_attn": "fa2",
                "liger_kernel": True,
                "template": "qwen3_nothink",
                "deepspeed": "z2_offload",
                "gradient_checkpointing": True,
            },
        },
        "eval": {
            "benchmarks": ["swe-bench-verified"],
            "seeds": [42],
        },
        "budget": {
            "max_gpu_hours": 96,
            "gpu_type": "1xH200-141GB",
            "slurm": {
                "partition": "gpu",
                "nodes": 1,
                "gpus_per_node": 1,
                "cpus_per_task": 16,
                "mem": "256G",
                "time": "72:00:00",
            },
        },
    }


def test_swe_lego_recipe_passes_schema_validation() -> None:
    recipe = _swe_lego_8b_recipe()
    schema = load_schema()
    errors = validate_recipe(recipe, schema)
    assert errors == [], f"Schema validation errors: {errors}"


def test_swe_lego_recipe_compiles_correctly() -> None:
    recipe = _swe_lego_8b_recipe()
    config = compile_recipe(recipe)
    assert config.backend == "swe_lego"
    assert config.trainer_type == "sft"
    assert config.model_config["base"] == "Qwen/Qwen3-8B"
    assert config.training_params["turn_mask"] is True
    assert config.training_params["lr"] == 1e-4


def test_swe_lego_bundle_generation(tmp_path: Path) -> None:
    recipe = _swe_lego_8b_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)

    assert bundle["backend"] == "swe_lego"
    assert bundle["recipe_id"] == "recipe-swe-lego-8b-test"
    assert bundle["trainer_type"] == "sft"
    assert "_train_config_dict" in bundle
    assert "_dataset_info_dict" in bundle
    assert "train_config" in bundle.get("files", {})


def test_swe_lego_bundle_write(tmp_path: Path) -> None:
    recipe = _swe_lego_8b_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    assert Path(paths["run_script"]).exists()
    assert Path(paths["env"]).exists()
    assert Path(paths["launcher_json"]).exists()
    assert Path(paths["train_config"]).exists()

    # Verify run.sh content
    run_script = Path(paths["run_script"]).read_text()
    assert "llamafactory-cli train" in run_script
    assert "env.sh" in run_script

    # Verify env.sh content
    env_content = Path(paths["env"]).read_text()
    assert "SWE_LEGO_ROOT" in env_content
    assert "LLAMA_FACTORY_DIR" in env_content

    # Verify launcher.json
    launcher = json.loads(Path(paths["launcher_json"]).read_text())
    assert launcher["backend"] == "swe_lego"

    # Verify train config YAML
    train_config = Path(paths["train_config"]).read_text()
    assert "Qwen/Qwen3-8B" in train_config
    assert "turn_mask" in train_config
    assert "rope_scaling" in train_config


def test_swe_lego_train_prepares_bundle(tmp_path: Path, capsys) -> None:
    from cli.train import run_train

    recipe = _swe_lego_8b_recipe()
    recipe_path = tmp_path / "swe-lego.recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))

    output_dir = tmp_path / "outputs"
    run_train(
        Namespace(
            recipe=str(recipe_path),
            output_dir=str(output_dir),
            dry_run=False,
            no_submit=True,
            import_results=None,
        ),
    )

    captured = capsys.readouterr().out
    assert "SWE-Lego" in captured
    assert "Status     : prepared" in captured

    plan_dir = output_dir / recipe["id"]
    plan = json.loads((plan_dir / "execution-plan.json").read_text())
    assert plan["mode"] == "prepared"
    assert plan["launcher"]["backend"] == "swe_lego"


def test_swe_lego_verifier_recipe_passes_schema() -> None:
    recipe = {
        "id": "recipe-swe-lego-verifier-8b-test",
        "name": "SWE-Lego Verifier 8B test",
        "model": {"base": "Qwen/Qwen3-8B", "size": "8B", "adapter": "full"},
        "dataset": {
            "sources": [
                {"name": "swe-lego-verifier", "path": "SWE-Lego/SWE_Lego_real_data_Verifier"},
            ]
        },
        "trainer": {
            "type": "sft",
            "backend": "swe_lego",
            "params": {"lr": 2e-5, "epochs": 5, "batch_size": 1, "turn_mask": False},
        },
        "eval": {"benchmarks": ["swe-bench-verified"], "seeds": [42]},
    }
    schema = load_schema()
    errors = validate_recipe(recipe, schema)
    assert errors == [], f"Schema validation errors: {errors}"


def test_slurm_config_in_schema() -> None:
    recipe = _swe_lego_8b_recipe()
    schema = load_schema()
    errors = validate_recipe(recipe, schema)
    assert errors == []
    # Verify slurm config passes through compilation
    config = compile_recipe(recipe)
    assert "slurm" in config.budget
    assert config.budget["slurm"]["partition"] == "gpu"
    assert config.budget["slurm"]["gpus_per_node"] == 1
