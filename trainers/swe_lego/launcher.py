"""Build SWE-Lego-compatible launch bundles from compiled recipe configs.

Bridges auto-coder-trainer's Recipe IR to SWE-Lego's LLaMA-Factory SFT
training pipeline.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Paths relative to the trainer root
# ---------------------------------------------------------------------------

_SWE_LEGO_SUBDIR = Path("trainers/swe_lego/SWE-Lego")
_LLAMA_FACTORY_SUBDIR = _SWE_LEGO_SUBDIR / "LLaMA-Factory-0.9.4.dev0"
_DEEPSPEED_DIR = "examples/deepspeed"

_DS_CONFIGS: dict[str, str] = {
    "z0": f"{_DEEPSPEED_DIR}/ds_z0_config.json",
    "z2": f"{_DEEPSPEED_DIR}/ds_z2_config.json",
    "z2_offload": f"{_DEEPSPEED_DIR}/ds_z2_offload_config.json",
    "z3": f"{_DEEPSPEED_DIR}/ds_z3_config.json",
    "z3_offload": f"{_DEEPSPEED_DIR}/ds_z3_offload_config.json",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_swe_lego_launcher_bundle(
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compile a training config into a SWE-Lego / LLaMA-Factory launch bundle."""
    recipe_id: str = config.get("recipe_id", "unknown")
    trainer_type: str = config.get("trainer_type", "sft")
    bundle_dir = Path(output_dir) / recipe_id / "swe_lego"

    model_cfg: dict[str, Any] = config.get("model_config", {})
    data_cfg: dict[str, Any] = config.get("data_config", {})
    training_params: dict[str, Any] = config.get("training_params", {})
    budget: dict[str, Any] = config.get("budget", {})

    if trainer_type != "sft":
        raise ValueError(
            f"SWE-Lego launcher only supports trainer_type='sft'; got {trainer_type!r}"
        )

    # Resolve absolute SWE-Lego root (assume repo root is two levels above this file)
    repo_root = Path(__file__).resolve().parent.parent.parent
    swe_lego_root = repo_root / _SWE_LEGO_SUBDIR
    llama_factory_dir = repo_root / _LLAMA_FACTORY_SUBDIR

    train_config = _build_train_config(
        recipe_id=recipe_id,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        training_params=training_params,
        bundle_dir=bundle_dir,
    )
    dataset_info = _build_dataset_info(data_cfg)
    ds_path = _resolve_deepspeed_config(training_params)

    gpu_count = _default_gpu_count(budget)

    env: dict[str, str] = {
        "SWE_LEGO_ROOT": str(swe_lego_root),
        "LLAMA_FACTORY_DIR": str(llama_factory_dir),
        "ACT_RECIPE_ID": recipe_id,
        "ACT_GPU_COUNT": gpu_count,
    }

    warnings: list[str] = []
    if not data_cfg.get("sources"):
        warnings.append(
            "Recipe has no dataset sources. "
            "Add dataset entries to dataset_info_patch.json before launch."
        )
    if model_cfg.get("adapter", "full") != "full":
        warnings.append(
            "SWE-Lego training configs are designed for full fine-tuning. "
            "LoRA / QLoRA would require a different YAML template."
        )
    if int(gpu_count) > 1 and ds_path == _DS_CONFIGS.get("z2_offload"):
        warnings.append(
            "ds_z2_offload_config.json is intended for single-GPU. "
            "Consider z3 for multi-GPU setups."
        )

    yaml_file = bundle_dir / "train_config.yaml"
    dataset_info_file = bundle_dir / "dataset_info_patch.json"
    env_file = bundle_dir / "env.sh"
    run_file = bundle_dir / "run.sh"
    launcher_json = bundle_dir / "launcher.json"

    entrypoint = {
        "kind": "llamafactory-cli",
        "command_prefix": [
            "llamafactory-cli",
            "train",
            str(yaml_file),
        ],
    }
    command_preview = " ".join(entrypoint["command_prefix"])

    return {
        "backend": "swe_lego",
        "recipe_id": recipe_id,
        "trainer_type": trainer_type,
        "artifact_dir": str(bundle_dir),
        "entrypoint": entrypoint,
        "command_preview": command_preview,
        "env": env,
        "warnings": warnings,
        "requirements": [
            "Install LLaMA-Factory (pip install -e LLaMA-Factory-0.9.4.dev0/) before launch.",
            "Merge dataset_info_patch.json into LLaMA-Factory's data/dataset_info.json.",
            "Run the generated run.sh script to start training.",
        ],
        "source_dataset_refs": [
            {
                "name": source.get("name", f"source-{idx}"),
                "path": source.get("path", ""),
                "mix_weight": source.get("mix_weight", 1.0),
            }
            for idx, source in enumerate(data_cfg.get("sources", []))
        ],
        "files": {
            "bundle_dir": str(bundle_dir),
            "train_config": str(yaml_file),
            "dataset_info_patch": str(dataset_info_file),
            "env": str(env_file),
            "run_script": str(run_file),
            "launcher_json": str(launcher_json),
        },
        # Internal data carried for write step
        "_train_config_dict": train_config,
        "_dataset_info_dict": dataset_info,
    }


def write_swe_lego_launcher_bundle(bundle: dict[str, Any]) -> dict[str, str]:
    """Persist a SWE-Lego launch bundle to disk."""
    bundle_dir = Path(bundle["artifact_dir"])
    bundle_dir.mkdir(parents=True, exist_ok=True)

    train_config_path = Path(bundle["files"]["train_config"])
    dataset_info_path = Path(bundle["files"]["dataset_info_patch"])
    env_path = Path(bundle["files"]["env"])
    run_path = Path(bundle["files"]["run_script"])
    launcher_path = Path(bundle["files"]["launcher_json"])

    train_config_path.write_text(
        _render_train_yaml(bundle["_train_config_dict"])
    )
    dataset_info_path.write_text(
        json.dumps(bundle["_dataset_info_dict"], indent=2) + "\n"
    )
    env_path.write_text(_render_env(bundle))
    run_path.write_text(_render_run_script(bundle))

    # Strip internal keys before persisting the JSON manifest
    serializable = {k: v for k, v in bundle.items() if not k.startswith("_")}
    launcher_path.write_text(json.dumps(serializable, indent=2) + "\n")

    run_path.chmod(0o755)

    return {
        "bundle_dir": str(bundle_dir),
        "train_config": str(train_config_path),
        "dataset_info_patch": str(dataset_info_path),
        "env": str(env_path),
        "run_script": str(run_path),
        "launcher_json": str(launcher_path),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_train_config(
    recipe_id: str,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    training_params: dict[str, Any],
    bundle_dir: Path,
) -> dict[str, Any]:
    """Return a dict that mirrors the LLaMA-Factory training YAML structure."""
    sources = data_cfg.get("sources", [])
    dataset_names = [_source_to_dataset_key(s) for s in sources]
    dataset_str = ",".join(dataset_names) if dataset_names else "swe_lego_data"

    ds_path = _resolve_deepspeed_config(training_params)

    config: dict[str, Any] = {
        # --- model ---
        "model_name_or_path": model_cfg.get("base", "Qwen/Qwen3-8B"),
        "trust_remote_code": True,
        # --- method ---
        "stage": "sft",
        "do_train": True,
        "finetuning_type": model_cfg.get("adapter", "full"),
        "deepspeed": ds_path,
        # --- dataset ---
        "dataset": dataset_str,
        "template": training_params.get("template", "qwen3_nothink"),
        "cutoff_len": training_params.get("max_length", 131072),
        "max_samples": training_params.get("max_samples", 1000000),
        "overwrite_cache": True,
        "preprocessing_num_workers": training_params.get(
            "preprocessing_num_workers", 16
        ),
        "dataloader_num_workers": training_params.get("dataloader_num_workers", 4),
        "turn_mask": training_params.get("turn_mask", True),
        # --- output ---
        "output_dir": str(bundle_dir / "saves" / f"SWE-Lego-{recipe_id}"),
        "logging_steps": training_params.get("logging_steps", 1),
        "save_strategy": training_params.get("save_strategy", "epoch"),
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": True,
        "report_to": training_params.get("report_to", "wandb"),
        # --- train ---
        "per_device_train_batch_size": training_params.get("batch_size", 1),
        "gradient_accumulation_steps": training_params.get(
            "gradient_accumulation_steps", 8
        ),
        "learning_rate": float(training_params.get("lr", 1e-4)),
        "weight_decay": float(training_params.get("weight_decay", 0.01)),
        "max_grad_norm": float(training_params.get("max_grad_norm", 1.0)),
        "num_train_epochs": float(training_params.get("epochs", 4)),
        "lr_scheduler_type": training_params.get("lr_scheduler", "cosine"),
        "warmup_ratio": float(training_params.get("warmup_ratio", 0.1)),
        "bf16": True,
        "ddp_timeout": training_params.get("ddp_timeout", 180000000),
        "resume_from_checkpoint": None,
    }

    # Optional toggles
    rope_scaling = training_params.get("rope_scaling")
    if rope_scaling:
        config["rope_scaling"] = rope_scaling

    flash_attn = training_params.get("flash_attn")
    if flash_attn:
        config["flash_attn"] = flash_attn

    if training_params.get("liger_kernel", False):
        config["enable_liger_kernel"] = True

    if training_params.get("gradient_checkpointing", False):
        config["use_unsloth_gc"] = True

    return config


def _build_dataset_info(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return LLaMA-Factory dataset_info.json entries for each data source."""
    entries: dict[str, Any] = {}
    for source in data_cfg.get("sources", []):
        key = _source_to_dataset_key(source)
        source_path = source.get("path", "")
        entries[key] = {
            "file_name": source_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    return entries


def _resolve_deepspeed_config(training_params: dict[str, Any]) -> str:
    """Return the relative path (within LLaMA-Factory) to the DeepSpeed config."""
    ds_key = str(training_params.get("deepspeed", "z2_offload")).lower()

    if ds_key in _DS_CONFIGS:
        return _DS_CONFIGS[ds_key]

    # Allow passing a raw path through
    if "/" in ds_key or ds_key.endswith(".json"):
        return ds_key

    # Fall back to z2_offload for single-GPU default
    return _DS_CONFIGS["z2_offload"]


def _render_env(bundle: dict[str, Any]) -> str:
    """Render the env.sh file contents."""
    lines: list[str] = [
        "#!/usr/bin/env bash",
        "# Generated by auto-coder-trainer (SWE-Lego launcher).",
        "",
    ]
    for source in bundle.get("source_dataset_refs", []):
        lines.append(
            f"# Dataset source: {source.get('name', 'unknown')} -> "
            f"{source.get('path', '')}"
        )
    lines.append("")
    for key, value in bundle["env"].items():
        lines.append(f'export {key}="${{{key}:-{shlex.quote(value)}}}"')
    lines.append("")
    return "\n".join(lines)


def _render_run_script(bundle: dict[str, Any]) -> str:
    """Render the run.sh launcher script."""
    train_config_path = bundle["files"]["train_config"]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'source "$SCRIPT_DIR/env.sh"',
            "",
            '# Change into LLaMA-Factory directory so relative paths resolve',
            'cd "$LLAMA_FACTORY_DIR"',
            "",
            f"llamafactory-cli train {shlex.quote(train_config_path)} \"$@\"",
            "",
        ]
    )


def _render_train_yaml(config_dict: dict[str, Any]) -> str:
    """Render a LLaMA-Factory training YAML from a config dict."""
    return yaml.dump(
        config_dict,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def _source_to_dataset_key(source: dict[str, Any]) -> str:
    """Convert a data source dict to a LLaMA-Factory dataset_info key."""
    name = source.get("name", "unknown")
    # Normalise to underscore-separated lowercase
    return name.replace("-", "_").replace(" ", "_").lower()


def _default_gpu_count(budget: dict[str, Any]) -> str:
    """Extract GPU count from a budget gpu_type string like '1xH200-141GB'."""
    gpu_type = str(budget.get("gpu_type", "")).strip()
    if "x" in gpu_type.lower():
        maybe_count = gpu_type.lower().split("x", 1)[0].strip()
        if maybe_count.isdigit():
            return maybe_count
    return "1"
