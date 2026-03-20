"""Train command — execute a training experiment from a recipe.

Compiles a Recipe IR JSON into a training configuration, runs the
experiment, evaluates results, and submits to the experiment judge.
"""

import argparse
import json
import uuid
from pathlib import Path


def _plan_dir(output_dir: Path, recipe_id: str) -> Path:
    """Return the directory used to store execution-plan artifacts."""
    return output_dir / recipe_id


def _build_execution_plan(
    recipe: dict,
    config,
    output_dir: Path,
    *,
    reason: str,
    dry_run: bool,
    launcher: dict | None = None,
) -> dict:
    """Build a human-usable execution plan for blocked or dry-run jobs."""
    from recipes.compiler import normalize_recipe

    plan_dir = _plan_dir(output_dir, config.recipe_id)
    if launcher:
        next_steps = [
            "Open the generated env.sh and map ACT_TRAIN_FILE / ACT_VAL_FILE to real parquet files.",
            "Install a TinyZero/veRL-compatible environment on the training machine.",
            "Launch the generated run.sh script and append any extra Hydra overrides if needed.",
        ]
        mode = "dry_run" if dry_run else "prepared"
    else:
        next_steps = [
            "Implement prepare_data/train/evaluate in the selected trainer backend.",
            "Run the recipe again with --dry-run to re-check validation and the compiled config.",
            "Once the trainer is available, run a small sanity job before full deployment.",
        ]
        mode = "dry_run" if dry_run else "blocked"

    return {
        "recipe_id": config.recipe_id,
        "reason": reason,
        "mode": mode,
        "trainer": {
            "type": config.trainer_type,
            "backend": config.backend,
        },
        "model": normalize_recipe(config.model_config),
        "dataset": normalize_recipe(config.data_config),
        "eval": normalize_recipe(config.eval_config),
        "ablation": normalize_recipe(config.ablation_configs),
        "budget": normalize_recipe(config.budget),
        "output_dir": str(output_dir),
        "artifact_dir": str(plan_dir),
        "next_steps": next_steps,
        "launcher": launcher,
        "recipe": recipe,
    }


def _format_execution_plan(plan: dict) -> str:
    """Render a compact Markdown execution plan."""
    lines = [
        f"# Execution Plan: {plan['recipe_id']}",
        "",
        f"- **Mode**: {plan['mode']}",
        f"- **Reason**: {plan['reason']}",
        f"- **Trainer**: {plan['trainer']['type']} / {plan['trainer']['backend']}",
        f"- **Artifact dir**: {plan['artifact_dir']}",
        "",
        "## Model",
        f"- Base: {plan['model'].get('base', '?')}",
        f"- Adapter: {plan['model'].get('adapter', '?')}",
        "",
        "## Data",
    ]
    sources = plan["dataset"].get("sources", [])
    if sources:
        for src in sources:
            lines.append(
                f"- {src.get('name', '?')}: {src.get('path', '?')} (weight={src.get('mix_weight', 1.0)})"
            )
    else:
        lines.append("- No data sources configured yet.")

    lines.extend(
        [
            "",
            "## Evaluation",
            f"- Benchmarks: {', '.join(plan['eval'].get('benchmarks', [])) or 'none'}",
            f"- Seeds: {', '.join(str(s) for s in plan['eval'].get('seeds', [])) or 'none'}",
            "",
            "## Budget",
            f"- GPU hours: {plan['budget'].get('max_gpu_hours', 'unspecified')}",
            f"- GPU type: {plan['budget'].get('gpu_type', 'unspecified')}",
        ]
    )

    if plan["ablation"]:
        lines.extend(["", "## Ablations"])
        for abl in plan["ablation"]:
            lines.append(
                f"- {abl.get('name', '?')}: {abl.get('variable', '?')} -> {abl.get('values', [])}"
            )

    launcher = plan.get("launcher")
    if launcher:
        lines.extend(
            [
                "",
                "## Launcher",
                f"- Backend: {launcher.get('backend', '?')}",
                f"- Module: {launcher.get('entrypoint', {}).get('module', '?')}",
                f"- Bundle dir: {launcher.get('artifact_dir', '?')}",
                f"- Run script: {launcher.get('files', {}).get('run_script', '?')}",
                f"- Overrides: {launcher.get('files', {}).get('hydra_overrides', '?')}",
            ]
        )
        for warning in launcher.get("warnings", []):
            lines.append(f"- Warning: {warning}")

    lines.extend(
        [
            "",
            "## Next Steps",
        ]
    )
    for step in plan["next_steps"]:
        lines.append(f"- {step}")

    return "\n".join(lines) + "\n"


def _write_execution_plan(plan: dict, output_dir: Path) -> tuple[Path, Path]:
    """Persist the execution plan as JSON and Markdown artifacts."""
    plan_dir = Path(plan["artifact_dir"])
    plan_dir.mkdir(parents=True, exist_ok=True)
    json_path = plan_dir / "execution-plan.json"
    md_path = plan_dir / "execution-plan.md"
    json_path.write_text(json.dumps(plan, indent=2))
    md_path.write_text(_format_execution_plan(plan))
    return json_path, md_path


def _trainer_unavailable_message(trainer_type: str, backend: str, blocked_reason: str) -> str:
    """Return a short human-readable reason for a blocked training run."""
    return (
        f"Training backend is not ready for {trainer_type}/{backend}: {blocked_reason}. "
        "An execution plan has been written instead."
    )


def run_train(args: argparse.Namespace) -> None:
    """Execute the training pipeline.

    Pipeline:
        1. Load and validate recipe JSON
        2. Compile recipe to training config
        3. Select trainer backend (TRL for SFT, veRL for RL)
        4. Run training
        5. Evaluate on specified benchmarks
        6. Submit to experiment judge
        7. Store results in result DB
    """
    recipe_path = Path(args.recipe)
    output_dir = Path(getattr(args, "output_dir", "outputs/"))
    dry_run = getattr(args, "dry_run", False)

    # ------------------------------------------------------------------
    # 1. Load recipe
    # ------------------------------------------------------------------
    print(f"[train] Loading recipe: {recipe_path}")
    try:
        with open(recipe_path) as f:
            recipe = json.load(f)
    except FileNotFoundError:
        print(f"[train] Error: recipe file not found: {recipe_path}")
        return
    except json.JSONDecodeError as exc:
        print(f"[train] Error: invalid JSON in {recipe_path}: {exc}")
        return

    recipe_id = recipe.get("id", "unknown")
    print(f"[train] Recipe ID: {recipe_id}")

    # ------------------------------------------------------------------
    # 2. Validate recipe
    # ------------------------------------------------------------------
    try:
        from recipes.compiler import load_schema, normalize_recipe, validate_recipe

        recipe = normalize_recipe(recipe)

        schema = load_schema()
        errors = validate_recipe(recipe, schema)
        if errors:
            print(f"[train] Validation errors ({len(errors)}):")
            for err in errors:
                print(f"[train]   - {err}")
            print("[train] Aborting — fix the recipe and retry.")
            return
        print("[train] Recipe validation passed.")
    except Exception as exc:
        print(f"[train] Warning: could not validate recipe ({exc}). Proceeding anyway.")

    # ------------------------------------------------------------------
    # 3. Compile recipe to TrainingConfig
    # ------------------------------------------------------------------
    try:
        from recipes.compiler import compile_recipe

        config = compile_recipe(recipe)
        print(f"[train] Compiled config: {config.backend}/{config.trainer_type}")
    except Exception as exc:
        print(f"[train] Error compiling recipe: {exc}")
        return

    launcher_bundle = None
    launcher_paths = None
    if config.backend == "tinyzero":
        try:
            from trainers.tinyzero import (
                build_tinyzero_launcher_bundle,
                write_tinyzero_launcher_bundle,
            )

            launcher_bundle = build_tinyzero_launcher_bundle(config.__dict__, output_dir)
            launcher_paths = write_tinyzero_launcher_bundle(launcher_bundle)
            print(f"[train] TinyZero launch bundle ready: {launcher_paths['run_script']}")
        except Exception as exc:
            print(f"[train] Error building TinyZero launch bundle: {exc}")
            return

    if dry_run:
        plan_reason = "dry-run requested"
    elif config.backend == "tinyzero":
        plan_reason = "TinyZero launch bundle prepared"
    else:
        plan_reason = "trainer backend not yet implemented"
    execution_plan = _build_execution_plan(
        recipe,
        config,
        output_dir,
        reason=plan_reason,
        dry_run=dry_run,
        launcher=launcher_bundle,
    )

    if dry_run:
        json_path, md_path = _write_execution_plan(execution_plan, output_dir)
        print("[train] Dry-run mode — skipping training.")
        print(f"[train] Execution plan written to {json_path}")
        print(f"[train] Plan summary written to {md_path}")
        print(f"[train]   Trainer : {config.trainer_type} ({config.backend})")
        print(f"[train]   Model   : {config.model_config}")
        print(f"[train]   Data    : {config.data_config}")
        print(f"[train]   Eval    : {config.eval_config}")
        if launcher_paths:
            print(f"[train]   Launch  : {launcher_paths['run_script']}")
        return

    if config.backend == "tinyzero":
        json_path, md_path = _write_execution_plan(execution_plan, output_dir)
        print("[train] TinyZero backend selected — external launch bundle prepared.")
        print(f"[train] Execution plan written to {json_path}")
        print(f"[train] Plan summary written to {md_path}")
        if launcher_paths:
            print(f"[train] Launcher JSON  : {launcher_paths['launcher_json']}")
            print(f"[train] Env template   : {launcher_paths['env']}")
            print(f"[train] Run script     : {launcher_paths['run_script']}")
        print("\n[train] === Summary ===")
        print(f"[train] Recipe     : {recipe_id}")
        print(f"[train] Trainer    : {config.trainer_type} / {config.backend}")
        print("[train] Status     : prepared")
        print("[train] Done.")
        return

    # ------------------------------------------------------------------
    # 4. Select and instantiate trainer
    # ------------------------------------------------------------------
    trainer = None
    trainer_init_error = None
    try:
        if config.trainer_type == "sft":
            try:
                from trainers.sft import SFTTrainer
                trainer = SFTTrainer(config.__dict__, output_dir)
                print("[train] Using SFT trainer (TRL backend).")
            except ImportError:
                print("[train] SFT trainer module not yet available.")
                trainer_init_error = "trainer module not available"
        elif config.trainer_type in ("rl", "grpo"):
            try:
                from trainers.rl import RLTrainer
                trainer = RLTrainer(config.__dict__, output_dir)
                print("[train] Using RL trainer (veRL backend).")
            except ImportError:
                print("[train] RL trainer module not yet available.")
                trainer_init_error = "trainer module not available"
        else:
            print(f"[train] Unknown trainer type: {config.trainer_type}")
            trainer_init_error = "unknown trainer type"
    except Exception as exc:
        print(f"[train] Error initializing trainer: {exc}")
        trainer_init_error = str(exc)

    if trainer is None:
        execution_plan["reason"] = _trainer_unavailable_message(
            config.trainer_type,
            config.backend,
            trainer_init_error or "no trainer class available",
        )
        json_path, md_path = _write_execution_plan(execution_plan, output_dir)
        print(f"[train] {execution_plan['reason']}")
        print(f"[train] Execution plan written to {json_path}")
        print(f"[train] Plan summary written to {md_path}")
        return

    # ------------------------------------------------------------------
    # 5. Run training + evaluation
    # ------------------------------------------------------------------
    print("[train] Starting training run ...")
    try:
        train_result, eval_results = trainer.run()
        print(f"[train] Training finished — status: {train_result.status}")
        if train_result.metrics:
            print(f"[train] Train metrics: {train_result.metrics}")
        for er in eval_results:
            print(f"[train] Eval [{er.benchmark}] seed={er.seed}: {er.metrics}")
    except NotImplementedError as exc:
        execution_plan["reason"] = _trainer_unavailable_message(
            config.trainer_type,
            config.backend,
            str(exc),
        )
        json_path, md_path = _write_execution_plan(execution_plan, output_dir)
        print(f"[train] {execution_plan['reason']}")
        print(f"[train] Execution plan written to {json_path}")
        print(f"[train] Plan summary written to {md_path}")
        train_result = None
        eval_results = []
    except Exception as exc:
        print(f"[train] Training failed: {exc}")
        train_result = None
        eval_results = []

    # ------------------------------------------------------------------
    # 6. Submit to experiment judge
    # ------------------------------------------------------------------
    verdict = None
    db = None
    experiment_id = None
    config_hash = None

    if train_result:
        try:
            from results.db import ResultDB
            from judge.dedup import compute_config_hash

            db = ResultDB()
            db.connect()
            experiment_id = f"exp-{uuid.uuid4().hex[:8]}"
            config_hash = compute_config_hash(recipe)
        except Exception as exc:
            print(f"[train] Warning: could not prepare results DB: {exc}")
            db = None

    if train_result and train_result.status == "success":
        try:
            from judge.judge import ExperimentJudge

            judge = ExperimentJudge(result_db=db)
            results_dict = {
                "train": train_result.__dict__ if hasattr(train_result, "__dict__") else {},
                "eval": [er.__dict__ for er in eval_results] if eval_results else [],
                "recipe": recipe,
                "ablation": recipe.get("ablation", []),
                "expected_seeds": config.eval_config.get("seeds", []),
                "status": train_result.status,
                "trainer_type": config.trainer_type,
                "backend": config.backend,
                "experiment_id": experiment_id,
            }
            verdict = judge.judge(recipe_id, results_dict)
            print(f"[train] Judge verdict: {verdict.verdict.value} — {verdict.reasoning}")
        except NotImplementedError:
            print("[train] Judge not fully implemented — skipping verdict.")
        except Exception as exc:
            print(f"[train] Judge error: {exc}")

    # ------------------------------------------------------------------
    # 7. Store results in DB
    # ------------------------------------------------------------------
    if train_result and db is not None and experiment_id is not None and config_hash is not None:
        try:
            db.insert_experiment({
                "id": experiment_id,
                "recipe_id": recipe_id,
                "config_hash": config_hash,
                "status": train_result.status,
                "trainer_type": config.trainer_type,
                "backend": config.backend,
                "model_base": config.model_config.get("base", ""),
                "metrics_json": train_result.metrics or {},
                "checkpoint_path": train_result.checkpoint_path,
                "error": train_result.error,
            })

            if verdict:
                db.insert_verdict({
                    "experiment_id": experiment_id,
                    "verdict": verdict.verdict.value,
                    "reasoning": verdict.reasoning,
                    "checks_json": verdict.checks,
                    "suggestions_json": verdict.suggestions,
                })

            print(f"[train] Results stored — experiment_id: {experiment_id}")
        except Exception as exc:
            print(f"[train] Warning: could not store results in DB: {exc}")
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n[train] === Summary ===")
    print(f"[train] Recipe     : {recipe_id}")
    print(f"[train] Trainer    : {config.trainer_type} / {config.backend}")
    status = train_result.status if train_result else "blocked"
    print(f"[train] Status     : {status}")
    if verdict:
        print(f"[train] Verdict    : {verdict.verdict.value}")
    print("[train] Done.")
