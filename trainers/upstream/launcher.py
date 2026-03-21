"""Build external upstream launch bundles from compiled recipe configs."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any


SUPPORTED_UPSTREAM_BACKENDS = {"openr1", "agent_distill", "redi"}


def build_upstream_launcher_bundle(
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compile a training config into an external-upstream launch bundle."""
    backend = str(config.get("backend", "")).strip()
    if backend not in SUPPORTED_UPSTREAM_BACKENDS:
        raise ValueError(
            f"Unsupported upstream backend {backend!r}; expected one of {sorted(SUPPORTED_UPSTREAM_BACKENDS)}"
        )

    recipe_id = config.get("recipe_id", "unknown")
    trainer_type = config.get("trainer_type", "unknown")
    bundle_dir = Path(output_dir) / recipe_id / backend
    model_cfg = config.get("model_config", {})
    data_cfg = config.get("data_config", {})
    training_params = config.get("training_params", {})
    distill_cfg = config.get("distill_config", {})

    if backend == "openr1":
        spec = _build_openr1_spec(
            recipe_id=recipe_id,
            trainer_type=trainer_type,
            bundle_dir=bundle_dir,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            training_params=training_params,
            distill_cfg=distill_cfg,
        )
    elif backend == "agent_distill":
        spec = _build_agent_distill_spec(
            recipe_id=recipe_id,
            trainer_type=trainer_type,
            bundle_dir=bundle_dir,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            distill_cfg=distill_cfg,
        )
    else:
        spec = _build_redi_spec(
            recipe_id=recipe_id,
            trainer_type=trainer_type,
            bundle_dir=bundle_dir,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            training_params=training_params,
            distill_cfg=distill_cfg,
        )

    launcher_path = bundle_dir / "launcher.json"
    env_path = bundle_dir / "env.sh"
    run_path = bundle_dir / "run.sh"
    notes_path = bundle_dir / "UPSTREAM_NOTES.md"

    return {
        "backend": backend,
        "recipe_id": recipe_id,
        "trainer_type": trainer_type,
        "artifact_dir": str(bundle_dir),
        "entrypoint": spec["entrypoint"],
        "command_preview": spec["command_preview"],
        "env": spec["env"],
        "warnings": spec["warnings"],
        "requirements": spec["requirements"],
        "source_dataset_refs": [
            {
                "name": source.get("name", f"source-{idx}"),
                "path": source.get("path", ""),
                "mix_weight": source.get("mix_weight", 1.0),
            }
            for idx, source in enumerate(data_cfg.get("sources", []))
        ],
        "upstream": spec["upstream"],
        "notes": spec["notes"],
        "files": {
            "bundle_dir": str(bundle_dir),
            "env": str(env_path),
            "run_script": str(run_path),
            "launcher_json": str(launcher_path),
            "notes": str(notes_path),
            "hydra_overrides": spec.get("hydra_overrides", "n/a"),
        },
    }


def write_upstream_launcher_bundle(bundle: dict[str, Any]) -> dict[str, str]:
    """Persist an external-upstream launch bundle to disk."""
    bundle_dir = Path(bundle["artifact_dir"])
    bundle_dir.mkdir(parents=True, exist_ok=True)

    launcher_path = Path(bundle["files"]["launcher_json"])
    env_path = Path(bundle["files"]["env"])
    run_path = Path(bundle["files"]["run_script"])
    notes_path = Path(bundle["files"]["notes"])

    launcher_path.write_text(json.dumps(bundle, indent=2))
    env_path.write_text(_render_env(bundle))
    run_path.write_text(_render_run_script(bundle))
    notes_path.write_text(_render_notes(bundle))
    run_path.chmod(0o755)

    return {
        "bundle_dir": str(bundle_dir),
        "launcher_json": str(launcher_path),
        "env": str(env_path),
        "run_script": str(run_path),
        "notes": str(notes_path),
    }


def _build_openr1_spec(
    *,
    recipe_id: str,
    trainer_type: str,
    bundle_dir: Path,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    training_params: dict[str, Any],
    distill_cfg: dict[str, Any],
) -> dict[str, Any]:
    script = "grpo.py" if trainer_type in {"rl", "grpo"} else "sft.py"
    command = [
        "accelerate",
        "launch",
        "--config_file",
        "${ACT_ACCELERATE_CONFIG}",
        f"src/open_r1/{script}",
        "${ACT_OPENR1_RECIPE}",
    ]
    return {
        "entrypoint": {
            "kind": "accelerate",
            "module": f"src/open_r1/{script}",
            "command_prefix": command,
        },
        "command_preview": " ".join(command),
        "env": {
            "ACT_UPSTREAM_DIR": str(bundle_dir / "open-r1"),
            "ACT_UPSTREAM_REPO": distill_cfg.get("upstream_repo", "https://github.com/huggingface/open-r1.git"),
            "ACT_ACCELERATE_CONFIG": "recipes/accelerate_configs/zero3.yaml",
            "ACT_OPENR1_RECIPE": "<set-openr1-yaml-recipe>",
            "ACT_MODEL_ID": model_cfg.get("base", ""),
            "ACT_DATASET_ID": _primary_dataset_path(data_cfg),
            "ACT_OUTPUT_DIR": str(bundle_dir / "runs"),
        },
        "warnings": [
            "Open-R1 is best suited for SFT/GRPO and reasoning-style distillation; pairwise REDI refinement is not native here.",
            "Map Recipe IR fields into an Open-R1 YAML recipe before launch.",
        ],
        "requirements": [
            "Clone the official open-r1 repo and install its dependencies.",
            "Create or adapt an Open-R1 YAML recipe for your dataset/model.",
            "Run the generated run.sh from inside the upstream repo checkout.",
        ],
        "upstream": {
            "name": "open-r1",
            "repo": "https://github.com/huggingface/open-r1",
            "license": "Apache-2.0",
        },
        "notes": [
            f"Use Open-R1 for recipe-grounded {trainer_type} jobs when you want to stay close to the upstream training stack.",
            f"Teacher model hint: {distill_cfg.get('teacher_model', 'n/a')}",
        ],
    }


def _build_agent_distill_spec(
    *,
    recipe_id: str,
    trainer_type: str,
    bundle_dir: Path,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    distill_cfg: dict[str, Any],
) -> dict[str, Any]:
    command = [
        "bash",
        "scripts/training/train_agent.sh",
        "${ACT_MODEL_ID}",
    ]
    return {
        "entrypoint": {
            "kind": "bash",
            "module": "scripts/training/train_agent.sh",
            "command_prefix": command,
        },
        "command_preview": " ".join(command),
        "env": {
            "ACT_UPSTREAM_DIR": str(bundle_dir / "agent-distillation"),
            "ACT_UPSTREAM_REPO": distill_cfg.get("upstream_repo", "https://github.com/Nardien/agent-distillation.git"),
            "ACT_MODEL_ID": model_cfg.get("base", ""),
            "ACT_TEACHER_MODEL": distill_cfg.get("teacher_model", ""),
            "ACT_TRAJECTORY_FILE": _primary_dataset_path(data_cfg),
            "ACT_OUTPUT_DIR": str(bundle_dir / "training_outputs"),
        },
        "warnings": [
            "The official repo expects its own trajectory generation / student training scripts and tool environment.",
            "If you already have offline trajectories, point ACT_TRAJECTORY_FILE to them before running.",
        ],
        "requirements": [
            "Clone the official agent-distillation repo and install its dependencies.",
            "Generate teacher trajectories if you do not already have them.",
            "Run the generated training script and then evaluate with the upstream student-agent script.",
        ],
        "upstream": {
            "name": "agent-distillation",
            "repo": "https://github.com/Nardien/agent-distillation",
            "license": "Apache-2.0",
        },
        "notes": [
            f"Agent Distillation is the closest upstream match for tool-using small coding agents distilled from teacher trajectories (recipe: {recipe_id}).",
            f"Teacher model hint: {distill_cfg.get('teacher_model', 'n/a')}",
        ],
    }


def _build_redi_spec(
    *,
    recipe_id: str,
    trainer_type: str,
    bundle_dir: Path,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    training_params: dict[str, Any],
    distill_cfg: dict[str, Any],
) -> dict[str, Any]:
    command = [
        "bash",
        "${ACT_REDI_STAGE2_SCRIPT}",
    ]
    return {
        "entrypoint": {
            "kind": "bash",
            "module": "${ACT_REDI_STAGE2_SCRIPT}",
            "command_prefix": command,
        },
        "command_preview": " ".join(command),
        "env": {
            "ACT_UPSTREAM_DIR": str(bundle_dir / "reinforcement-distillation"),
            "ACT_UPSTREAM_REPO": distill_cfg.get("upstream_repo", "https://github.com/Tim-Siu/reinforcement-distillation.git"),
            "ACT_MODEL_ID": model_cfg.get("base", ""),
            "ACT_TEACHER_MODEL": distill_cfg.get("teacher_model", ""),
            "ACT_REDI_DATA_DIR": _primary_dataset_path(data_cfg),
            "ACT_REDI_STAGE1_SCRIPT": "<set-positive-sft-train.sh>",
            "ACT_REDI_STAGE2_SCRIPT": "<set-redi-refine-train.sh>",
            "ACT_OUTPUT_DIR": str(bundle_dir / "experiments_trl"),
            "ACT_PAIRWISE_BETA": str(distill_cfg.get("pairwise_beta", training_params.get("pairwise_beta", 0.1))),
        },
        "warnings": [
            "REDI is an external upstream recipe; use this launcher when you want the official negative-signal refinement rather than a native approximation.",
            "You will need to map your distilled positives/pairs into the upstream repo's expected dataset layout.",
        ],
        "requirements": [
            "Clone the official reinforcement-distillation repo and install its dependencies.",
            "Prepare positive traces and chosen/rejected pairs in the upstream data directory.",
            "Run ACT_REDI_STAGE1_SCRIPT for positive SFT, then ACT_REDI_STAGE2_SCRIPT for REDI refinement.",
        ],
        "upstream": {
            "name": "reinforcement-distillation",
            "repo": "https://github.com/Tim-Siu/reinforcement-distillation",
            "license": "MIT (verify upstream before redistribution)",
        },
        "notes": [
            f"REDI is the recommended upstream for negative-signal refinement on top of positive distillation (recipe: {recipe_id}).",
            f"Teacher model hint: {distill_cfg.get('teacher_model', 'n/a')}",
            f"Configured pairwise beta: {distill_cfg.get('pairwise_beta', training_params.get('pairwise_beta', 0.1))}",
        ],
    }


def _render_env(bundle: dict[str, Any]) -> str:
    lines = ["#!/usr/bin/env bash", ""]
    for key, value in bundle.get("env", {}).items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append("")
    return "\n".join(lines)


def _render_run_script(bundle: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "$ROOT_DIR/env.sh"',
        "",
        'if [ ! -d "${ACT_UPSTREAM_DIR}" ]; then',
        '  git clone "${ACT_UPSTREAM_REPO}" "${ACT_UPSTREAM_DIR}"',
        "fi",
        "",
        'cd "${ACT_UPSTREAM_DIR}"',
        "",
    ]
    for warning in bundle.get("warnings", []):
        lines.append(f'echo "warning: {warning}"')
    lines.append("")
    lines.append(" ".join(shlex.quote(str(part)) for part in bundle["entrypoint"]["command_prefix"]))
    lines.append("")
    return "\n".join(lines)


def _render_notes(bundle: dict[str, Any]) -> str:
    parts = [
        f"# Upstream Launcher: {bundle.get('backend', 'unknown')}",
        "",
        f"- **Recipe ID**: {bundle.get('recipe_id', 'unknown')}",
        f"- **Trainer Type**: {bundle.get('trainer_type', 'unknown')}",
        f"- **Upstream Repo**: {bundle.get('upstream', {}).get('repo', 'unknown')}",
        f"- **License**: {bundle.get('upstream', {}).get('license', 'unknown')}",
        "",
        "## Requirements",
    ]
    for requirement in bundle.get("requirements", []):
        parts.append(f"- {requirement}")
    if bundle.get("warnings"):
        parts.extend(["", "## Warnings"])
        for warning in bundle["warnings"]:
            parts.append(f"- {warning}")
    if bundle.get("notes"):
        parts.extend(["", "## Notes"])
        for note in bundle["notes"]:
            parts.append(f"- {note}")
    parts.append("")
    return "\n".join(parts)


def _primary_dataset_path(data_cfg: dict[str, Any]) -> str:
    sources = data_cfg.get("sources", [])
    if not sources:
        return "<set-dataset-path>"
    return str(sources[0].get("path", "<set-dataset-path>"))
