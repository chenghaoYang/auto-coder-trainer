"""Build SSD (Simple Self-Distillation) launch bundles from compiled recipe configs.

Generates a 4-stage SLURM pipeline:
    sample.sh → train.sh → eval.sh → import_results.sh

Follows the same launcher bundle pattern as trainers/swe_lego/ and trainers/tinyzero/.

Reference:
    Agarwal, R., et al. "Embarrassingly Simple Self-Distillation Improves Code Generation."
    arXiv:2604.01193 (2026). https://arxiv.org/abs/2604.01193
"""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

from trainers.base import BaseTrainer, EvalResult, TrainResult
from trainers.ssd.data import generate_sampling_script as gen_sampling_script


# ---------------------------------------------------------------------------
# Public API: bundle builder
# ---------------------------------------------------------------------------


def build_ssd_launcher_bundle(
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compile a training config into an SSD launch bundle.

    Returns a dict describing the bundle (paths, metadata, warnings).
    Internal keys prefixed with ``_`` carry data for the write step.
    """
    recipe_id: str = config.get("recipe_id", "unknown")
    bundle_dir = Path(output_dir) / recipe_id / "ssd"

    model_cfg = config.get("model_config", {})
    training_params = config.get("training_params", {})
    eval_cfg = config.get("eval_config", {})
    budget = config.get("budget", {})

    model_name = model_cfg.get("base", "")
    tensor_parallel_size = int(training_params.get("tensor_parallel_size", 4))

    # Sample config
    sample_config = {
        "model_name": model_name,
        "output_path": str(bundle_dir / "sample_data.jsonl"),
        "dataset": "livecodebench/code_generation_lite",
        "dataset_split": "test",
        "dataset_filter": {
            "contest_date": training_params.get(
                "dataset_filter",
                ["2025-02", "2025-03", "2025-04", "2025-05"],
            ),
        },
        "temperature": float(training_params.get("sample_temperature", 0.9)),
        "top_p": float(training_params.get("sample_top_p", 0.8)),
        "top_k": int(training_params.get("sample_top_k", 20)),
        "max_tokens": int(training_params.get("max_tokens", 65536)),
        "n_samples": int(training_params.get("n_samples_per_problem", 10)),
        "seed": 0,
        "tensor_parallel_size": tensor_parallel_size,
    }

    # Eval config
    eval_config = {
        "model_path": str(bundle_dir / "model"),
        "output_path": str(bundle_dir / "eval_results.json"),
        "tensor_parallel_size": tensor_parallel_size,
        "max_tokens": int(training_params.get("eval_max_tokens", 32768)),
        "n_repeat": int(training_params.get("n_repeat", 20)),
        "sampling_params": {
            "temperature": float(training_params.get("decode_temperature", 0.6)),
            "top_p": 0.95,
            "top_k": 20,
        },
        "seed": [0, 1234, 1234, 1234],
    }

    # Train config (for downstream SFTTrainer)
    train_config = {
        "recipe_id": recipe_id,
        "model_config": model_cfg,
        "training_params": {
            "epochs": float(training_params.get("epochs", 1)),
            "lr": float(training_params.get("lr", 2e-5)),
            "batch_size": int(training_params.get("batch_size", 1)),
            "max_length": int(training_params.get("max_tokens", 65536)),
        },
        "data_config": {
            "sources": [
                {
                    "name": "ssd-sampled",
                    "path": str(bundle_dir / "sample_data.jsonl"),
                }
            ],
        },
    }

    warnings: list[str] = []
    if not model_name:
        warnings.append("No model specified in model_config.base")
    if int(training_params.get("n_samples_per_problem", 10)) < 5:
        warnings.append(
            "n_samples_per_problem < 5 may produce insufficient training data"
        )

    return {
        "backend": "ssd",
        "recipe_id": recipe_id,
        "trainer_type": "ssd",
        "artifact_dir": str(bundle_dir),
        "warnings": warnings,
        "requirements": [
            "Install vLLM >= 0.11.0 before launch.",
            "Ensure sufficient GPU resources for sampling and training stages.",
            "Install with: pip install -e '.[ssd]'",
        ],
        "_sample_config": sample_config,
        "_train_config": train_config,
        "_eval_config": eval_config,
        "_model_config": model_cfg,
        "_training_params": training_params,
        "_budget": budget,
    }


def write_ssd_launcher_bundle(bundle: dict[str, Any]) -> dict[str, str]:
    """Persist an SSD launch bundle to disk.

    Generates four shell scripts (sample, train, eval, import_results),
    JSON configs, and a launcher.json manifest.
    """
    from trainers.ssd.lcb_evaluator import generate_eval_script as gen_eval_script

    bundle_dir = Path(bundle["artifact_dir"])
    logs_dir = bundle_dir / "logs"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    sample_script_path = bundle_dir / "sample.sh"
    train_script_path = bundle_dir / "train.sh"
    eval_script_path = bundle_dir / "eval.sh"
    import_results_path = bundle_dir / "import_results.sh"
    env_path = bundle_dir / "env.sh"
    launcher_path = bundle_dir / "launcher.json"
    sample_config_path = bundle_dir / "sample_config.json"
    eval_config_path = bundle_dir / "eval_config.json"
    logs_dir = bundle_dir / "logs"

    # Write configs
    sample_config_path.write_text(json.dumps(bundle["_sample_config"], indent=2) + "\n")
    eval_config_path.write_text(json.dumps(bundle["_eval_config"], indent=2) + "\n")

    # Write sample script (Python wrapper)
    sample_python = gen_sampling_script(bundle["_sample_config"])
    sample_script_path.write_text(_render_python_wrapper(sample_python, "sample"))
    sample_script_path.chmod(0o755)

    # Write train script
    train_script_path.write_text(_render_train_script(bundle))
    train_script_path.chmod(0o755)

    # Write eval script (Python wrapper)
    eval_python = gen_eval_script(bundle["_eval_config"])
    eval_script_path.write_text(_render_python_wrapper(eval_python, "eval"))
    eval_script_path.chmod(0o755)

    # Write import_results script
    import_results_path.write_text(_render_import_results_script(bundle))
    import_results_path.chmod(0o755)

    # Write env.sh
    env_path.write_text(_render_env(bundle))

    # Write launcher.json (strip internal keys)
    serializable = {k: v for k, v in bundle.items() if not k.startswith("_")}
    launcher_path.write_text(json.dumps(serializable, indent=2) + "\n")

    # Create logs dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "bundle_dir": str(bundle_dir),
        "sample_script": str(sample_script_path),
        "train_script": str(train_script_path),
        "eval_script": str(eval_script_path),
        "import_results_script": str(import_results_path),
        "env_script": str(env_path),
        "launcher_json": str(launcher_path),
        "sample_config": str(sample_config_path),
        "eval_config": str(eval_config_path),
    }


# ---------------------------------------------------------------------------
# SLURM pipeline submission
# ---------------------------------------------------------------------------


def run_ssd_pipeline(
    bundle_dir: str | Path,
    slurm_config: dict[str, Any],
) -> dict[str, Any]:
    """Submit the full SSD pipeline with SLURM dependency chain.

    Stages: sample → train → eval → import_results
    """
    from trainers.slurm.submitter import (
        render_sbatch,
        submit_job,
        submit_with_dependency,
        write_sbatch_script,
    )

    bundle_dir = Path(bundle_dir).resolve()
    slurm_dir = bundle_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    cfg = {**slurm_config, "bundle_dir": str(bundle_dir)}
    recipe_id = bundle_dir.parent.name

    stages = [
        ("sample", "sample.sh", "sample.sbatch"),
        ("train", "train.sh", "train.sbatch"),
        ("eval", "eval.sh", "eval.sbatch"),
        ("import_results", "import_results.sh", "import_results.sbatch"),
    ]

    # Render sbatch scripts
    sbatch_paths: dict[str, Path] = {}
    for key, script, fname in stages:
        job_name = f"act-{recipe_id}-ssd-{key}"
        content = render_sbatch(job_name, script, cfg, slurm_dir)
        sbatch_paths[key] = write_sbatch_script(content, slurm_dir / fname)

    # Submit with dependency chain: sample → train → eval → import
    job_ids: dict[str, str] = {}
    job_ids["sample"] = submit_job(sbatch_paths["sample"])
    job_ids["train"] = submit_with_dependency(sbatch_paths["train"], job_ids["sample"])
    job_ids["eval"] = submit_with_dependency(sbatch_paths["eval"], job_ids["train"])
    job_ids["import_results"] = submit_with_dependency(
        sbatch_paths["import_results"], job_ids["eval"]
    )

    return {
        "pipeline_id": recipe_id,
        "job_ids": job_ids,
        "bundle_dir": str(bundle_dir),
    }


# ---------------------------------------------------------------------------
# SSDLauncher (BaseTrainer integration)
# ---------------------------------------------------------------------------


class SSDLauncher(BaseTrainer):
    """Launcher-bundle trainer for Simple Self-Distillation.

    Generates a bundle of shell scripts and SLURM configs for the
    Sample → Fine-tune → Evaluate pipeline, then submits them.
    """

    def __init__(self, config: dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        self._bundle: dict[str, Any] | None = None
        self._paths: dict[str, str] | None = None

    def prepare_data(self) -> Any:
        """Build the launcher bundle (dry-run: generate scripts only)."""
        self._bundle = build_ssd_launcher_bundle(self.config, self.output_dir)
        self._paths = write_ssd_launcher_bundle(self._bundle)
        return {
            "artifact_dir": self._bundle["artifact_dir"],
            "warnings": self._bundle["warnings"],
            "scripts": list(self._paths.keys()),
        }

    def train(self) -> TrainResult:
        """Submit the SSD SLURM pipeline."""
        recipe_id = self.config.get("recipe_id", "unknown")

        if self._bundle is None:
            self.prepare_data()

        budget = self.config.get("budget", {})
        slurm_config = budget.get("slurm", {})
        if not slurm_config:
            # Default SLURM config
            slurm_config = {
                "partition": "gpu",
                "nodes": 1,
                "gpus_per_node": str(
                    self.config.get("training_params", {}).get("tensor_parallel_size", 4)
                ),
                "cpus_per_task": 8,
                "mem": "64G",
                "time": "08:00:00",
            }

        try:
            result = run_ssd_pipeline(
                bundle_dir=self._bundle["artifact_dir"],
                slurm_config=slurm_config,
            )
            return TrainResult(
                recipe_id=recipe_id,
                trainer_type="ssd",
                backend="ssd",
                status="success",
                metrics={"pipeline_id": result["pipeline_id"]},
                checkpoint_path=self._bundle["artifact_dir"],
            )
        except Exception as exc:
            return TrainResult(
                recipe_id=recipe_id,
                trainer_type="ssd",
                backend="ssd",
                status="failed",
                error=str(exc),
            )

    def evaluate(self, checkpoint_path: str, benchmark: str, seed: int = 42) -> EvalResult:
        """Evaluate is handled by the eval.sh stage in the SLURM pipeline."""
        recipe_id = self.config.get("recipe_id", "unknown")
        return EvalResult(
            recipe_id=recipe_id,
            benchmark=benchmark,
            metrics={},
            seed=seed,
            details={"note": "SSD evaluation runs in the SLURM eval.sh stage"},
        )


# ---------------------------------------------------------------------------
# Script renderers
# ---------------------------------------------------------------------------


def _render_python_wrapper(python_script: str, stage: str) -> str:
    """Wrap a Python script in a bash runner that logs output."""
    # Escape single quotes for bash embedding
    escaped = python_script.replace("'", "'\\''")
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "$SCRIPT_DIR/env.sh"',
        "",
        f"# SSD {stage} stage",
        f"echo 'Starting SSD {stage} stage...'",
        'mkdir -p "$SCRIPT_DIR/logs"',
        f"python3 -c '{escaped}' 2>&1 | tee \"$SCRIPT_DIR/logs/{stage}.log\"",
        'RC=${PIPESTATUS[0]}',
        'if [[ "$RC" -ne 0 ]]; then',
        f'  echo "SSD {stage} failed with exit code $RC"',
        '  exit "$RC"',
        'fi',
        f'echo "SSD {stage} completed successfully"',
        "",
    ])


def _render_train_script(bundle: dict[str, Any]) -> str:
    """Render train.sh — runs SFTTrainer on sampled data."""
    recipe_id = bundle["recipe_id"]
    artifact_dir = bundle.get("artifact_dir", "")
    train_config = bundle.get("_train_config", {})
    train_config_json = json.dumps(train_config)

    # Escape for bash
    train_config_escaped = train_config_json.replace("'", "'\\''")

    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "$SCRIPT_DIR/env.sh"',
        "",
        "# SSD Fine-tune stage — standard cross-entropy SFT on sampled data",
        "echo 'Starting SSD fine-tune stage...'",
        'mkdir -p "$SCRIPT_DIR/logs"',
        'mkdir -p "$SCRIPT_DIR/model"',
        "",
        "# Verify sample data exists",
        'if [[ ! -f "$SCRIPT_DIR/sample_data.jsonl" ]]; then',
        '  echo "ERROR: sample_data.jsonl not found. Did sample.sh run successfully?"',
        '  exit 1',
        'fi',
        "",
        "# Count samples",
        f'SAMPLE_COUNT=$(wc -l < "$SCRIPT_DIR/sample_data.jsonl")',
        'echo "Found $SAMPLE_COUNT training samples"',
        "",
        "# Run SFT training",
        f"python3 -c '",
        f"import json, os, sys",
        f"from trainers.sft.trainer import SFTTrainer",
        f"",
        f"config = json.loads(\"\"\"{train_config_escaped}\"\"\")",
        f'config[\"recipe_id\"] = {shlex.quote(recipe_id)}',
        f'config[\"trainer_type\"] = \"sft\"',
        f'config[\"backend\"] = \"trl\"',
        f"",
        f'trainer = SFTTrainer(config, {shlex.quote(os.path.join(artifact_dir, "model"))})',
        f"trainer.prepare_data()",
        f"result = trainer.train()",
        f"",
        f'result_path = os.path.join({shlex.quote(artifact_dir)}, \"train_result.json\")',
        f'with open(result_path, \"w\") as f:',
        f'    json.dump({{\"status\": result.status, \"metrics\": result.metrics}}, f, default=str)',
        f'print(f\"Training result: {{result.status}}\")',
        f"' 2>&1 | tee \"$SCRIPT_DIR/logs/train.log\"",
        'RC=${PIPESTATUS[0]}',
        'if [[ "$RC" -ne 0 ]]; then',
        '  echo "SSD train failed with exit code $RC"',
        '  exit "$RC"',
        'fi',
        'echo "SSD train completed successfully"',
        "",
    ])


def _render_import_results_script(bundle: dict[str, Any]) -> str:
    """Render import_results.sh — imports eval results into DB."""
    artifact_dir = bundle.get("artifact_dir", "")
    recipe_id = bundle.get("recipe_id", "")
    report_dir = str(Path(artifact_dir).parent / "reports") if artifact_dir else ""

    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "$SCRIPT_DIR/env.sh"',
        "",
        "# SSD import results stage",
        "echo 'Importing SSD results into DB...'",
        "",
        "# Verify eval results exist",
        'if [[ ! -f "$SCRIPT_DIR/eval_results.json" ]]; then',
        '  echo "WARNING: eval_results.json not found. Skipping import."',
        '  exit 0',
        'fi',
        "",
        f"python3 -m cli.main train --import-results \"$SCRIPT_DIR\" \\",
        f"  --recipe-id {shlex.quote(recipe_id)} \\",
        "  --experiment-id ${ACT_EXPERIMENT_ID:-ssd-exp-001} \\",
        "  --report-format ${ACT_REPORT_FORMAT:-blog} \\",
        f"  --report-output {shlex.quote(report_dir)}",
        "",
        'echo "SSD results imported successfully"',
        "",
    ])


def _render_env(bundle: dict[str, Any]) -> str:
    """Render env.sh with export variables."""
    recipe_id = bundle.get("recipe_id", "unknown")
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by auto-coder-trainer (SSD launcher).",
        "",
        f"export ACT_RECIPE_ID={shlex.quote(recipe_id)}",
        'export ACT_EXPERIMENT_ID="${ACT_EXPERIMENT_ID:-ssd-exp-001}"',
        "",
    ]

    # Add model info
    model_name = bundle.get("_model_config", {}).get("base", "")
    if model_name:
        lines.append(f"export ACT_MODEL_NAME={shlex.quote(model_name)}")

    lines.append("")
    return "\n".join(lines)
