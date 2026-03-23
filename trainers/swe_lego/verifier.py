"""Verifier training and TTS (Test-Time Selection) pipeline for SWE-Lego.

Builds LLaMA-Factory configs for verifier training, generates inference scripts,
and assembles the full TTS pipeline.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

SWE_LEGO_ROOT = Path(__file__).parent / "SWE-Lego"


def build_verifier_train_config(
    recipe_id: str,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    training_params: dict[str, Any],
    bundle_dir: str | Path,
) -> dict[str, Any]:
    """Generate LLaMA-Factory YAML config for verifier training.

    Key differences from policy model training:
    - Learning rate: 2e-5 (vs 1e-4 for policy)
    - Epochs: 5 (vs 4 for policy)
    - Uses verifier dataset
    - No turn_mask

    Returns YAML-serialisable dict.
    """
    bundle_dir = Path(bundle_dir)
    base_model = model_cfg.get("base", "Qwen/Qwen3-8B")
    dataset_name = data_cfg.get("dataset", "swe_lego_real_data_trajectories_verifier")
    output_dir = str(bundle_dir / "saves" / f"{recipe_id}-verifier")

    return {
        # model
        "model_name_or_path": base_model,
        "trust_remote_code": True,
        # method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": training_params.get("finetuning_type", "full"),
        "deepspeed": training_params.get("deepspeed", "examples/deepspeed/ds_z3_config.json"),
        # dataset
        "dataset": dataset_name,
        "template": training_params.get("template", "qwen3_nothink"),
        "cutoff_len": training_params.get("cutoff_len", 131072),
        "rope_scaling": training_params.get("rope_scaling", "yarn"),
        "max_samples": training_params.get("max_samples", 1000000),
        "overwrite_cache": True,
        "preprocessing_num_workers": training_params.get("preprocessing_num_workers", 16),
        "dataloader_num_workers": training_params.get("dataloader_num_workers", 4),
        # output
        "output_dir": output_dir,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": True,
        "report_to": training_params.get("report_to", "wandb"),
        # train — verifier-specific defaults
        "per_device_train_batch_size": training_params.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": training_params.get("gradient_accumulation_steps", 8),
        "learning_rate": training_params.get("learning_rate", 2.0e-5),
        "weight_decay": training_params.get("weight_decay", 0.01),
        "max_grad_norm": training_params.get("max_grad_norm", 1.0),
        "num_train_epochs": training_params.get("num_train_epochs", 5.0),
        "lr_scheduler_type": training_params.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": training_params.get("warmup_ratio", 0.1),
        "bf16": True,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None,
        "enable_liger_kernel": True,
        "use_unsloth_gc": True,
        "flash_attn": "fa2",
    }


def build_verifier_train_bundle(
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build complete verifier training bundle.

    Similar to policy launcher but with verifier-specific config.

    Returns bundle dict with config, paths, and launch script content.
    """
    import yaml

    output_dir = Path(output_dir)
    bundle_dir = output_dir / "verifier"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = bundle_dir / "verifier_train.yaml"
    run_script_path = bundle_dir / "run_train.sh"

    swe_lego_root = "${ACT_SWE_LEGO_ROOT:-" + shlex.quote(str(SWE_LEGO_ROOT)) + "}"
    run_script = f"""\
#!/usr/bin/env bash
set -euo pipefail

SWE_LEGO_ROOT={swe_lego_root}
cd "$SWE_LEGO_ROOT/LLaMA-Factory-0.9.4.dev0"

export WANDB_API_KEY="${{WANDB_API_KEY:-}}"

FORCE_TORCHRUN=1 llamafactory-cli train {shlex.quote(str(yaml_path))}
"""

    return {
        "config": config,
        "bundle_dir": str(bundle_dir),
        "files": {
            "yaml_config": str(yaml_path),
            "run_script": str(run_script_path),
        },
        "yaml_content": yaml.dump(config, default_flow_style=False, sort_keys=False),
        "run_script_content": run_script,
    }


def write_verifier_train_bundle(bundle: dict[str, Any]) -> dict[str, str]:
    """Write verifier training bundle to disk.

    Returns paths dict.
    """
    bundle_dir = Path(bundle["bundle_dir"])
    bundle_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = Path(bundle["files"]["yaml_config"])
    yaml_path.write_text(bundle["yaml_content"])

    run_script_path = Path(bundle["files"]["run_script"])
    run_script_path.write_text(bundle["run_script_content"])
    run_script_path.chmod(0o755)

    launcher_path = bundle_dir / "launcher.json"
    launcher_path.write_text(json.dumps(bundle, indent=2))

    return {
        "bundle_dir": str(bundle_dir),
        "yaml_config": str(yaml_path),
        "run_script": str(run_script_path),
        "launcher_json": str(launcher_path),
    }


def build_verifier_infer_script(
    model_path: str,
    input_path: str,
    output_path: str,
    bundle_dir: str | Path,
) -> str:
    """Generate verifier inference script for TTS.

    Uses LLaMA-Factory's tts/inference_verifier_tts.py.

    Returns script content string.
    """
    bundle_dir = Path(bundle_dir)
    swe_lego_root = "${ACT_SWE_LEGO_ROOT:-" + shlex.quote(str(SWE_LEGO_ROOT)) + "}"
    return f"""\
#!/usr/bin/env bash
set -euo pipefail

# Verifier inference script — generated by auto-coder-trainer

SWE_LEGO_ROOT={swe_lego_root}
cd "$SWE_LEGO_ROOT/LLaMA-Factory-0.9.4.dev0"

python tts/inference_verifier_tts.py \\
    --model_path {shlex.quote(model_path)} \\
    --input_path {shlex.quote(input_path)} \\
    --output_path {shlex.quote(output_path)} \\
    --bf16
"""


def build_tts_pipeline_script(
    verifier_model_path: str,
    policy_output_dir: str,
    bundle_dir: str | Path,
) -> str:
    """Generate full TTS (Test-Time Selection) pipeline script.

    Steps:
    1. Convert trajectories to verifier format
    2. Run verifier inference to score each trajectory
    3. Select best predictions per instance
    4. Re-evaluate with SWE-bench harness

    Returns script content string.
    """
    bundle_dir = Path(bundle_dir)
    swe_lego_root = "${ACT_SWE_LEGO_ROOT:-" + shlex.quote(str(SWE_LEGO_ROOT)) + "}"
    return f"""\
#!/usr/bin/env bash
set -euo pipefail

# Full TTS pipeline script — generated by auto-coder-trainer

SWE_LEGO_ROOT={swe_lego_root}
VERIFIER_MODEL={shlex.quote(verifier_model_path)}
POLICY_OUTPUT_DIR={shlex.quote(policy_output_dir)}

WORK_DIR={shlex.quote(str(bundle_dir / "tts_work"))}
mkdir -p "$WORK_DIR"

VERIFIER_INPUT="$WORK_DIR/verifier_input.jsonl"
VERIFIER_OUTPUT="$WORK_DIR/verifier_predictions.jsonl"
BEST_PREDS="$WORK_DIR/best_predictions.swebench.jsonl"

# Step 1: Convert trajectories to verifier format
echo "Step 1: Converting trajectories to verifier format..."
cd "$SWE_LEGO_ROOT/LLaMA-Factory-0.9.4.dev0"
python tts/convert_trajectories_to_verifier.py \\
    --input "$POLICY_OUTPUT_DIR/output.jsonl" \\
    --output "$VERIFIER_INPUT"

# Step 2: Run verifier inference
echo "Step 2: Running verifier inference..."
python tts/inference_verifier_tts.py \\
    --model_path "$VERIFIER_MODEL" \\
    --input_path "$VERIFIER_INPUT" \\
    --output_path "$VERIFIER_OUTPUT" \\
    --bf16

# Step 3: Select best predictions per instance
echo "Step 3: Selecting best predictions..."
python -c "
import json, sys
from collections import defaultdict

preds = defaultdict(list)
with open('$VERIFIER_OUTPUT') as f:
    for line in f:
        rec = json.loads(line)
        for run in rec.get('runs', [rec]):
            iid = rec.get('instance_id', run.get('instance_id', 'unknown'))
            preds[iid].append(run)

with open('$BEST_PREDS', 'w') as out:
    for iid, runs in sorted(preds.items()):
        best = max(runs, key=lambda r: r.get('predicted_score', 0))
        out.write(json.dumps({{'instance_id': iid, 'run': best.get('run', 'run_1')}}) + '\\n')
print(f'Selected best from {{len(preds)}} instances.')
"

# Step 4: Re-evaluate with SWE-bench
echo "Step 4: Running SWE-bench evaluation on best predictions..."
cd "$SWE_LEGO_ROOT/SWE-bench-4.0.4"
python -m swebench.harness.run_evaluation \\
    --max_workers 10 \\
    --dataset_name princeton-nlp/SWE-bench_Verified \\
    --report_dir ./results/ \\
    --cache_level instance \\
    --predictions_path "$BEST_PREDS" \\
    --run_id tts \\
    --timeout 500

echo "TTS pipeline complete. Results in $SWE_LEGO_ROOT/SWE-bench-4.0.4/results/"
"""
