"""Train command — execute a training experiment from a recipe.

Compiles a Recipe IR JSON into a training configuration, runs the
experiment, evaluates results, and submits to the experiment judge.
"""

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from trainers.base import TrainResult


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


def _task_id(
    recipe_id: str,
    experiment_id: str | None,
    kind: str,
    title: str,
    payload: dict | None = None,
) -> str:
    raw = json.dumps(
        {
            "recipe_id": recipe_id,
            "experiment_id": experiment_id,
            "kind": kind,
            "title": title,
            "payload": payload or {},
        },
        sort_keys=True,
        default=str,
    )
    return "task-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _make_task(
    *,
    recipe_id: str,
    experiment_id: str | None,
    kind: str,
    title: str,
    status: str = "pending",
    priority: str = "medium",
    payload: dict | None = None,
    notes: str | None = None,
) -> dict:
    return {
        "id": _task_id(recipe_id, experiment_id, kind, title, payload),
        "recipe_id": recipe_id,
        "experiment_id": experiment_id,
        "kind": kind,
        "title": title,
        "status": status,
        "priority": priority,
        "payload_json": payload or {},
        "notes": notes,
    }


def _aggregate_eval_results(eval_results: list) -> dict[str, float]:
    """Aggregate eval metrics across seeds and benchmarks for persistence."""
    by_benchmark: dict[str, dict[str, list[float]]] = {}
    global_metrics: dict[str, list[float]] = {}
    for result in eval_results:
        benchmark = getattr(result, "benchmark", "main")
        metrics = getattr(result, "metrics", {}) or {}
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            by_benchmark.setdefault(benchmark, {}).setdefault(key, []).append(float(value))
            global_metrics.setdefault(key, []).append(float(value))

    summary: dict[str, float] = {}
    for benchmark, metrics in by_benchmark.items():
        for key, values in metrics.items():
            summary[f"{benchmark}/{key}"] = sum(values) / len(values)
    for key, values in global_metrics.items():
        summary[key] = sum(values) / len(values)
    return summary


def _execution_plan_tasks(plan: dict, experiment_id: str | None) -> list[dict]:
    recipe_id = plan["recipe_id"]
    tasks = [
        _make_task(
            recipe_id=recipe_id,
            experiment_id=experiment_id,
            kind="compile_recipe",
            title=f"Recipe compiled in {plan['mode']} mode",
            status="completed",
            priority="medium",
            payload={"reason": plan.get("reason"), "trainer": plan.get("trainer")},
        )
    ]
    for step in plan.get("next_steps", []):
        tasks.append(
            _make_task(
                recipe_id=recipe_id,
                experiment_id=experiment_id,
                kind="execution_step",
                title=step,
                status="pending",
                priority="high" if plan.get("mode") in {"prepared", "blocked"} else "medium",
                payload={"mode": plan.get("mode")},
            )
        )
    return tasks


def _post_train_tasks(
    *,
    recipe_id: str,
    experiment_id: str | None,
    train_result: TrainResult,
    verdict,
    eval_results: list,
    expected_seeds: list[int],
    ablation_config: list[dict] | None = None,
    result_db=None,
) -> list[dict]:
    tasks: list[dict] = []
    tasks.append(
        _make_task(
            recipe_id=recipe_id,
            experiment_id=experiment_id,
            kind="record_training_run",
            title=f"Record training run ({train_result.status})",
            status="completed",
            priority="medium",
            payload={
                "status": train_result.status,
                "checkpoint_path": train_result.checkpoint_path,
            },
            notes=train_result.error,
        )
    )
    if train_result.status != "success":
        tasks.append(
            _make_task(
                recipe_id=recipe_id,
                experiment_id=experiment_id,
                kind="debug_training_failure",
                title="Investigate failed training run",
                status="pending",
                priority="high",
                payload={"error": train_result.error},
                notes=train_result.error,
            )
        )
        return tasks

    observed_seeds = sorted(
        {
            getattr(result, "seed", None)
            for result in eval_results
            if getattr(result, "seed", None) is not None
        }
    )
    missing_seeds = [seed for seed in expected_seeds if seed not in observed_seeds]
    for result in eval_results:
        tasks.append(
            _make_task(
                recipe_id=recipe_id,
                experiment_id=experiment_id,
                kind="record_eval_run",
                title=f"Record evaluation for {result.benchmark} seed {result.seed}",
                status="completed",
                priority="medium",
                payload={
                    "benchmark": result.benchmark,
                    "seed": result.seed,
                    "metrics": result.metrics,
                },
            )
        )

    if verdict is None:
        tasks.append(
            _make_task(
                recipe_id=recipe_id,
                experiment_id=experiment_id,
                kind="review_results",
                title="Review experiment outputs and generate a report",
                status="pending",
                priority="medium",
            )
        )
        return tasks

    verdict_value = verdict.verdict.value
    tasks.append(
        _make_task(
            recipe_id=recipe_id,
            experiment_id=experiment_id,
            kind="record_judge_verdict",
            title=f"Record judge verdict ({verdict_value})",
            status="completed",
            priority="medium",
            payload={
                "verdict": verdict_value,
                "reasoning": verdict.reasoning,
                "suggestions": verdict.suggestions,
            },
        )
    )
    if verdict_value == "accept":
        tasks.append(
            _make_task(
                recipe_id=recipe_id,
                experiment_id=experiment_id,
                kind="generate_report",
                title="Generate and review experiment report",
                status="pending",
                priority="low",
            )
        )
    elif verdict_value == "needs_rerun":
        if missing_seeds:
            for seed in missing_seeds:
                tasks.append(
                    _make_task(
                        recipe_id=recipe_id,
                        experiment_id=experiment_id,
                        kind="rerun_seed",
                        title=f"Run evaluation for missing seed {seed}",
                        status="pending",
                        priority="high",
                        payload={"seed": seed},
                    )
                )
        else:
            tasks.append(
                _make_task(
                    recipe_id=recipe_id,
                    experiment_id=experiment_id,
                    kind="rerun_experiment",
                    title="Re-run experiment to satisfy judge requirements",
                    status="pending",
                    priority="high",
                    payload={"suggestions": verdict.suggestions},
                )
            )
    elif verdict_value == "needs_ablation":
        before_ablation_tasks = len(tasks)
        missing = []
        if result_db is not None:
            try:
                from judge.ablation import validate_ablation_coverage

                missing = validate_ablation_coverage(
                    {"recipe_id": recipe_id, "ablation": ablation_config or []},
                    result_db,
                )
            except Exception:
                missing = []
        suggestions = verdict.suggestions or []
        for missing_ablation in missing:
            tasks.append(
                _make_task(
                    recipe_id=recipe_id,
                    experiment_id=experiment_id,
                    kind="run_ablation",
                    title=f"Run missing ablation: {missing_ablation}",
                    status="pending",
                    priority="high",
                    payload={"missing": missing_ablation},
                )
            )
        for suggestion in suggestions:
            tasks.append(
                _make_task(
                    recipe_id=recipe_id,
                    experiment_id=experiment_id,
                    kind="run_ablation",
                    title=suggestion,
                    status="pending",
                    priority="high",
                    payload={"missing": missing},
                )
            )
        if len(tasks) == before_ablation_tasks:
            tasks.append(
                _make_task(
                    recipe_id=recipe_id,
                    experiment_id=experiment_id,
                    kind="run_ablation",
                    title="Run missing ablation experiments before acceptance",
                    status="pending",
                    priority="high",
                    payload={"suggestions": verdict.suggestions},
                )
            )
    elif verdict_value == "reject":
        tasks.append(
            _make_task(
                recipe_id=recipe_id,
                experiment_id=experiment_id,
                kind="failure_analysis",
                title="Review rejection reasoning and prepare the next training iteration",
                status="pending",
                priority="high",
                notes=verdict.reasoning,
            )
        )
    return tasks


def _persist_artifacts(result_db, recipe_id: str, experiment_id: str | None, artifacts: list[dict]) -> None:
    if result_db is None:
        return
    for artifact in artifacts:
        result_db.insert_artifact(
            {
                "recipe_id": recipe_id,
                "experiment_id": experiment_id,
                "kind": artifact["kind"],
                "path": artifact["path"],
                "metadata_json": artifact.get("metadata", {}),
            }
        )


def _write_task_ledger(result_db, recipe_id: str, experiment_id: str | None, ledger_dir: Path) -> tuple[Path, Path] | None:
    if result_db is None or experiment_id is None:
        return None
    from results.ledger import build_task_ledger, write_task_ledger

    bundle = result_db.get_experiment_bundle(experiment_id)
    latest_verdict = bundle.get("verdicts", [])[-1] if bundle.get("verdicts") else None
    ledger = build_task_ledger(
        recipe_id=recipe_id,
        experiment_id=experiment_id,
        experiment=bundle.get("experiment"),
        tasks=bundle.get("tasks", []),
        artifacts=bundle.get("artifacts", []),
        verdict=latest_verdict,
    )
    paths = write_task_ledger(ledger, ledger_dir)
    return Path(paths["json"]), Path(paths["markdown"])


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
    db = None
    verdict = None
    config_hash = None
    experiment_id = f"exp-{uuid.uuid4().hex[:8]}"
    final_status = "blocked"

    try:
        from results.db import ResultDB
        from judge.dedup import compute_config_hash

        db = ResultDB()
        db.connect()
        config_hash = compute_config_hash(recipe)
        db.insert_experiment(
            {
                "id": experiment_id,
                "recipe_id": recipe_id,
                "config_hash": config_hash,
                "status": "planned",
                "trainer_type": config.trainer_type,
                "backend": config.backend,
                "model_base": config.model_config.get("base", ""),
                "metrics_json": {},
                "train_metrics_json": {},
                "recipe_json": recipe,
                "budget_json": config.budget,
                "checkpoint_path": None,
                "error": None,
            }
        )
    except Exception as exc:
        print(f"[train] Warning: could not prepare results DB: {exc}")
        db = None
        config_hash = None

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
        if db is not None and config_hash is not None:
            db.insert_experiment(
                {
                    "id": experiment_id,
                    "recipe_id": recipe_id,
                    "config_hash": config_hash,
                    "status": "planned",
                    "trainer_type": config.trainer_type,
                    "backend": config.backend,
                    "model_base": config.model_config.get("base", ""),
                    "metrics_json": {},
                    "train_metrics_json": {},
                    "recipe_json": recipe,
                    "budget_json": config.budget,
                    "checkpoint_path": None,
                    "error": None,
                }
            )
            for task in _execution_plan_tasks(execution_plan, experiment_id):
                db.upsert_task(task)
            _persist_artifacts(
                db,
                recipe_id,
                experiment_id,
                [
                    {"kind": "execution_plan_json", "path": str(json_path)},
                    {"kind": "execution_plan_markdown", "path": str(md_path)},
                ],
            )
            ledger_paths = _write_task_ledger(db, recipe_id, experiment_id, Path(execution_plan["artifact_dir"]))
            if ledger_paths:
                print(f"[train] Task ledger written to {ledger_paths[0]}")
            db.close()
            db = None
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
        if db is not None and config_hash is not None:
            db.insert_experiment(
                {
                    "id": experiment_id,
                    "recipe_id": recipe_id,
                    "config_hash": config_hash,
                    "status": "prepared",
                    "trainer_type": config.trainer_type,
                    "backend": config.backend,
                    "model_base": config.model_config.get("base", ""),
                    "metrics_json": {},
                    "train_metrics_json": {},
                    "recipe_json": recipe,
                    "budget_json": config.budget,
                    "checkpoint_path": None,
                    "error": None,
                }
            )
            for task in _execution_plan_tasks(execution_plan, experiment_id):
                db.upsert_task(task)
            artifact_rows = [
                {"kind": "execution_plan_json", "path": str(json_path)},
                {"kind": "execution_plan_markdown", "path": str(md_path)},
            ]
            if launcher_paths:
                artifact_rows.extend(
                    [
                        {"kind": "tinyzero_launcher_json", "path": launcher_paths["launcher_json"]},
                        {"kind": "tinyzero_env", "path": launcher_paths["env"]},
                        {"kind": "tinyzero_run_script", "path": launcher_paths["run_script"]},
                        {"kind": "tinyzero_hydra_overrides", "path": launcher_paths["hydra_overrides"]},
                    ]
                )
            _persist_artifacts(db, recipe_id, experiment_id, artifact_rows)
            ledger_paths = _write_task_ledger(db, recipe_id, experiment_id, Path(execution_plan["artifact_dir"]))
            if ledger_paths:
                print(f"[train] Task ledger written to {ledger_paths[0]}")
            db.close()
            db = None
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
        if db is not None and config_hash is not None:
            db.insert_experiment(
                {
                    "id": experiment_id,
                    "recipe_id": recipe_id,
                    "config_hash": config_hash,
                    "status": "blocked",
                    "trainer_type": config.trainer_type,
                    "backend": config.backend,
                    "model_base": config.model_config.get("base", ""),
                    "metrics_json": {},
                    "train_metrics_json": {},
                    "recipe_json": recipe,
                    "budget_json": config.budget,
                    "checkpoint_path": None,
                    "error": execution_plan["reason"],
                }
            )
            for task in _execution_plan_tasks(execution_plan, experiment_id):
                db.upsert_task(task)
            _persist_artifacts(
                db,
                recipe_id,
                experiment_id,
                [
                    {"kind": "execution_plan_json", "path": str(json_path)},
                    {"kind": "execution_plan_markdown", "path": str(md_path)},
                ],
            )
            ledger_paths = _write_task_ledger(db, recipe_id, experiment_id, Path(execution_plan["artifact_dir"]))
            if ledger_paths:
                print(f"[train] Task ledger written to {ledger_paths[0]}")
            db.close()
            db = None
        print(f"[train] {execution_plan['reason']}")
        print(f"[train] Execution plan written to {json_path}")
        print(f"[train] Plan summary written to {md_path}")
        print("\n[train] === Summary ===")
        print(f"[train] Recipe     : {recipe_id}")
        print(f"[train] Trainer    : {config.trainer_type} / {config.backend}")
        print("[train] Status     : blocked")
        print("[train] Done.")
        return

    # ------------------------------------------------------------------
    # 5. Run training + evaluation
    # ------------------------------------------------------------------
    print("[train] Starting training run ...")
    if db is not None and config_hash is not None:
        try:
            db.insert_experiment(
                {
                    "id": experiment_id,
                    "recipe_id": recipe_id,
                    "config_hash": config_hash,
                    "status": "running",
                    "trainer_type": config.trainer_type,
                    "backend": config.backend,
                    "model_base": config.model_config.get("base", ""),
                    "metrics_json": {},
                    "train_metrics_json": {},
                    "recipe_json": recipe,
                    "budget_json": config.budget,
                    "checkpoint_path": None,
                    "error": None,
                }
            )
        except Exception as exc:
            print(f"[train] Warning: could not mark experiment as running: {exc}")
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
        if db is not None and config_hash is not None:
            try:
                db.insert_experiment(
                    {
                        "id": experiment_id,
                        "recipe_id": recipe_id,
                        "config_hash": config_hash,
                        "status": "blocked",
                        "trainer_type": config.trainer_type,
                        "backend": config.backend,
                        "model_base": config.model_config.get("base", ""),
                        "metrics_json": {},
                        "train_metrics_json": {},
                        "recipe_json": recipe,
                        "budget_json": config.budget,
                        "checkpoint_path": None,
                        "error": execution_plan["reason"],
                    }
                )
                for task in _execution_plan_tasks(execution_plan, experiment_id):
                    db.upsert_task(task)
                _persist_artifacts(
                    db,
                    recipe_id,
                    experiment_id,
                    [
                        {"kind": "execution_plan_json", "path": str(json_path)},
                        {"kind": "execution_plan_markdown", "path": str(md_path)},
                    ],
                )
                ledger_paths = _write_task_ledger(
                    db,
                    recipe_id,
                    experiment_id,
                    Path(execution_plan["artifact_dir"]),
                )
                if ledger_paths:
                    print(f"[train] Task ledger written to {ledger_paths[0]}")
            except Exception as store_exc:
                print(f"[train] Warning: could not store blocked execution plan: {store_exc}")
            finally:
                db.close()
                db = None
        print(f"[train] {execution_plan['reason']}")
        print(f"[train] Execution plan written to {json_path}")
        print(f"[train] Plan summary written to {md_path}")
        print("\n[train] === Summary ===")
        print(f"[train] Recipe     : {recipe_id}")
        print(f"[train] Trainer    : {config.trainer_type} / {config.backend}")
        print("[train] Status     : blocked")
        print("[train] Done.")
        return
    except Exception as exc:
        print(f"[train] Training failed: {exc}")
        train_result = TrainResult(
            recipe_id=recipe_id,
            trainer_type=config.trainer_type,
            backend=config.backend,
            status="failed",
            error=str(exc),
        )
        eval_results = []
    final_status = train_result.status

    # ------------------------------------------------------------------
    # 6. Submit to experiment judge
    # ------------------------------------------------------------------
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
            summary_metrics = _aggregate_eval_results(eval_results) or (train_result.metrics or {})
            db.insert_experiment(
                {
                    "id": experiment_id,
                    "recipe_id": recipe_id,
                    "config_hash": config_hash,
                    "status": train_result.status,
                    "trainer_type": config.trainer_type,
                    "backend": config.backend,
                    "model_base": config.model_config.get("base", ""),
                    "metrics_json": summary_metrics,
                    "train_metrics_json": train_result.metrics or {},
                    "recipe_json": recipe,
                    "budget_json": config.budget,
                    "checkpoint_path": train_result.checkpoint_path,
                    "error": train_result.error,
                }
            )
            if eval_results:
                db.insert_eval_runs(
                    [
                        {
                            "experiment_id": experiment_id,
                            "benchmark": er.benchmark,
                            "seed": er.seed,
                            "metrics_json": er.metrics,
                            "details_json": er.details,
                        }
                        for er in eval_results
                    ]
                )

            if verdict:
                db.insert_verdict(
                    {
                        "experiment_id": experiment_id,
                        "verdict": verdict.verdict.value,
                        "reasoning": verdict.reasoning,
                        "checks_json": verdict.checks,
                        "suggestions_json": verdict.suggestions,
                    }
                )

            tasks = _post_train_tasks(
                recipe_id=recipe_id,
                experiment_id=experiment_id,
                train_result=train_result,
                verdict=verdict,
                eval_results=eval_results,
                expected_seeds=config.eval_config.get("seeds", []),
                ablation_config=recipe.get("ablation", []),
                result_db=db,
            )
            for task in tasks:
                db.upsert_task(task)

            artifact_rows = []
            if train_result.checkpoint_path:
                artifact_rows.append({"kind": "checkpoint", "path": train_result.checkpoint_path})
            plan_dir = _plan_dir(output_dir, recipe_id)
            ledger_paths = _write_task_ledger(db, recipe_id, experiment_id, plan_dir)
            if ledger_paths:
                artifact_rows.extend(
                    [
                        {"kind": "task_ledger_json", "path": str(ledger_paths[0])},
                        {"kind": "task_ledger_markdown", "path": str(ledger_paths[1])},
                    ]
                )
                print(f"[train] Task ledger written to {ledger_paths[0]}")
            _persist_artifacts(db, recipe_id, experiment_id, artifact_rows)

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
    print(f"[train] Status     : {final_status}")
    if verdict:
        print(f"[train] Verdict    : {verdict.verdict.value}")
    print("[train] Done.")
