"""Train command — execute a training experiment from a recipe.

Compiles a Recipe IR JSON into a training configuration, runs the
experiment, evaluates results, and submits to the experiment judge.
"""

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from trainers.base import EvalResult, TrainResult


_EXTERNAL_PREPARED_BACKENDS = {"tinyzero", "openr1", "agent_distill", "redi", "swe_lego"}


def _plan_dir(output_dir: Path, recipe_id: str) -> Path:
    """Return the directory used to store execution-plan artifacts."""
    return output_dir / recipe_id


def _backend_label(backend: str) -> str:
    labels = {
        "tinyzero": "TinyZero",
        "openr1": "Open-R1",
        "agent_distill": "Agent Distillation",
        "redi": "REDI",
        "swe_lego": "SWE-Lego (LLaMA-Factory)",
    }
    return labels.get(backend, backend)


def _launcher_next_steps(launcher: dict[str, Any]) -> list[str]:
    requirements = launcher.get("requirements")
    if isinstance(requirements, list) and requirements:
        return [str(item) for item in requirements]
    return [
        "Inspect the generated launcher bundle and fill in any placeholder environment variables.",
        "Install the upstream backend's dependencies on the training machine.",
        "Run the generated run.sh script from the launcher bundle directory.",
    ]


def _launcher_artifacts(backend: str, launcher_paths: dict[str, str] | None) -> list[dict]:
    if not launcher_paths:
        return []
    artifact_rows = []
    for key, path in launcher_paths.items():
        if key == "bundle_dir":
            continue
        artifact_rows.append({"kind": f"{backend}_{key}", "path": path})
    return artifact_rows


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
        next_steps = _launcher_next_steps(launcher)
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
        "distill": normalize_recipe(getattr(config, "distill_config", {})),
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

    if plan.get("distill"):
        lines.extend(
            [
                "",
                "## Distillation",
                f"- Strategy: {plan['distill'].get('strategy', 'trajectory')}",
                f"- Teacher: {plan['distill'].get('teacher_model', 'unspecified')}",
                f"- Teacher mode: {plan['distill'].get('teacher_mode', 'offline_dataset')}",
                f"- Stages: {', '.join(plan['distill'].get('stages', [])) or 'positive_sft'}",
            ]
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
                    payload={"targets": [missing_ablation]},
                )
            )
        if not missing and suggestions:
            tasks.append(
                _make_task(
                    recipe_id=recipe_id,
                    experiment_id=experiment_id,
                    kind="run_ablation",
                    title="Run missing ablation experiments before acceptance",
                    status="pending",
                    priority="high",
                    payload={"suggestions": suggestions},
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


def _persist_ablation_record(
    result_db,
    recipe: dict[str, Any],
    experiment_id: str | None,
    metrics: dict[str, float] | None = None,
) -> None:
    """Register an ablation result when the recipe carries ablation metadata."""
    if result_db is None or experiment_id is None:
        return

    ablation_run = recipe.get("ablation_run")
    if not isinstance(ablation_run, dict):
        return

    variable = ablation_run.get("variable")
    if not variable:
        return

    result_db.insert_ablation(
        {
            "experiment_id": experiment_id,
            "variable": str(variable),
            "value": ablation_run.get("value"),
            "metrics_json": metrics or {},
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


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning {} on absence or parse failure."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_external_experiment(db, recipe_id: str, experiment_id: str | None) -> dict[str, Any] | None:
    """Locate the prepared/running external experiment that should receive imported results."""
    if experiment_id:
        experiment = db.get_experiment(experiment_id)
        if experiment is not None:
            return experiment

    experiments = [
        experiment
        for experiment in db.find_by_recipe(recipe_id)
        if experiment.get("backend") == "swe_lego"
    ]
    if not experiments:
        return None

    for status in ("running", "prepared", "planned"):
        for experiment in experiments:
            if experiment.get("status") == status:
                return experiment
    return experiments[0]


def _load_swe_lego_import_context(
    bundle_dir: Path,
    db,
    *,
    recipe_id_override: str | None = None,
    experiment_id_override: str | None = None,
) -> dict[str, Any]:
    """Resolve recipe/experiment metadata for a completed SWE-Lego bundle."""
    launcher = _load_json_if_exists(bundle_dir / "launcher.json")
    execution_plan = _load_json_if_exists(bundle_dir.parent / "execution-plan.json")

    recipe_id = (
        recipe_id_override
        or launcher.get("recipe_id")
        or execution_plan.get("recipe_id")
        or "imported"
    )
    experiment_id = experiment_id_override or launcher.get("experiment_id")

    existing = _find_external_experiment(db, recipe_id, experiment_id)
    if existing is not None:
        recipe_id = existing.get("recipe_id", recipe_id)
        experiment_id = existing.get("id", experiment_id)

    if experiment_id is None:
        experiment_id = f"exp-{uuid.uuid4().hex[:8]}"

    recipe = {}
    if existing is not None and isinstance(existing.get("recipe_json"), dict):
        recipe = existing["recipe_json"]
    if not recipe and isinstance(execution_plan.get("recipe"), dict):
        recipe = execution_plan["recipe"]
    if not recipe and isinstance(launcher.get("recipe"), dict):
        recipe = launcher["recipe"]

    config_hash = existing.get("config_hash") if existing is not None else launcher.get("config_hash")
    if config_hash is None and recipe:
        try:
            from judge.dedup import compute_config_hash

            config_hash = compute_config_hash(recipe)
        except Exception:
            config_hash = None
    if config_hash is None:
        config_hash = hashlib.sha1(
            json.dumps(
                {"recipe_id": recipe_id, "bundle_dir": str(bundle_dir)},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    model_base = ""
    if existing is not None:
        model_base = existing.get("model_base", "")
    if not model_base and isinstance(recipe.get("model"), dict):
        model_base = recipe["model"].get("base", "")

    budget = {}
    if existing is not None and isinstance(existing.get("budget_json"), dict):
        budget = existing["budget_json"]
    if not budget and isinstance(recipe.get("budget"), dict):
        budget = recipe["budget"]

    trainer_type = "sft"
    backend = "swe_lego"
    if existing is not None:
        trainer_type = existing.get("trainer_type", trainer_type)
        backend = existing.get("backend", backend)

    return {
        "recipe_id": recipe_id,
        "experiment_id": experiment_id,
        "recipe": recipe,
        "config_hash": config_hash,
        "model_base": model_base,
        "budget": budget,
        "trainer_type": trainer_type,
        "backend": backend,
        "bundle_dir": bundle_dir,
        "plan_dir": bundle_dir.parent,
    }


def _coerce_import_train_result(
    imported: dict[str, Any],
    *,
    recipe_id: str,
    trainer_type: str,
    backend: str,
) -> TrainResult:
    """Convert results_bridge output into the canonical TrainResult dataclass."""
    payload = imported.get("train_result", {}) if isinstance(imported, dict) else {}
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return TrainResult(
        recipe_id=recipe_id,
        trainer_type=trainer_type,
        backend=backend,
        status=str(payload.get("status", "failed")),
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        checkpoint_path=payload.get("checkpoint_path"),
        error=payload.get("error"),
    )


def _coerce_import_eval_results(
    imported: dict[str, Any],
    *,
    recipe_id: str,
) -> list[EvalResult]:
    """Convert results_bridge eval payloads into canonical EvalResult dataclasses."""
    raw_results = imported.get("eval_results", []) if isinstance(imported, dict) else []
    eval_results: list[EvalResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        details = item.get("details", {})
        if not isinstance(details, dict):
            details = {}
        eval_results.append(
            EvalResult(
                recipe_id=recipe_id,
                benchmark=str(item.get("benchmark", "")),
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                seed=int(item.get("seed", 42)),
                details=details,
            )
        )
    return eval_results


def _mark_external_execution_tasks_completed(db, recipe_id: str, experiment_id: str) -> None:
    """Close out preparatory execution tasks once imported results arrive."""
    tasks = db.get_tasks(recipe_id=recipe_id, experiment_id=experiment_id)
    for task in tasks:
        if task.get("kind") != "execution_step":
            continue
        if task.get("status") == "completed":
            continue
        task["status"] = "completed"
        note = "Completed by imported SWE-Lego bundle results."
        task["notes"] = note if not task.get("notes") else f"{task['notes']} | {note}"
        db.upsert_task(task)


def _persist_import_summary(
    bundle_dir: Path,
    *,
    recipe_id: str,
    experiment_id: str,
    train_result: TrainResult,
    eval_results,
    verdict,
) -> Path:
    """Write a compact import summary artifact beside the completed bundle."""
    report_dir = bundle_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"{experiment_id}_import_summary.json"
    payload = {
        "recipe_id": recipe_id,
        "experiment_id": experiment_id,
        "train_result": train_result.__dict__,
        "eval_results": [result.__dict__ for result in eval_results],
        "verdict": verdict.verdict.value if verdict else None,
        "reasoning": verdict.reasoning if verdict else None,
        "checks": verdict.checks if verdict else {},
        "suggestions": verdict.suggestions if verdict else [],
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str))
    return summary_path


def _import_swe_lego_results(args: argparse.Namespace) -> None:
    """Import completed SWE-Lego results, update the DB, and auto-generate a report."""
    from judge.judge import ExperimentJudge
    from results.db import ResultDB
    from trainers.swe_lego.results_bridge import import_results

    bundle_dir = Path(getattr(args, "import_results"))
    if not bundle_dir.exists():
        print(f"[train] Error: import bundle not found: {bundle_dir}")
        return

    report_format = getattr(args, "report_format", "blog")
    report_output = getattr(args, "report_output", None)

    db = ResultDB()
    db.connect()
    try:
        context = _load_swe_lego_import_context(
            bundle_dir,
            db,
            recipe_id_override=getattr(args, "recipe_id", None),
            experiment_id_override=getattr(args, "experiment_id", None),
        )

        print(f"[train] Importing SWE-Lego results from {bundle_dir}")
        print(f"[train] Recipe ID: {context['recipe_id']}")
        print(f"[train] Experiment ID: {context['experiment_id']}")

        expected_seeds = (
            context["recipe"].get("eval", {}).get("seeds", [])
            if isinstance(context["recipe"], dict)
            else []
        )
        imported = import_results(
            bundle_dir,
            recipe_id=context["recipe_id"],
            experiment_id=context["experiment_id"],
            expected_seeds=expected_seeds,
        )
        train_result = _coerce_import_train_result(
            imported,
            recipe_id=context["recipe_id"],
            trainer_type=context["trainer_type"],
            backend=context["backend"],
        )
        eval_results: list[EvalResult] = _coerce_import_eval_results(
            imported,
            recipe_id=context["recipe_id"],
        )

        judge = ExperimentJudge(result_db=db)
        results_dict = {
            "train": train_result.__dict__,
            "eval": [result.__dict__ for result in eval_results],
            "recipe": context["recipe"],
            "ablation": context["recipe"].get("ablation", []) if isinstance(context["recipe"], dict) else [],
            "expected_seeds": expected_seeds,
            "status": train_result.status,
            "trainer_type": context["trainer_type"],
            "backend": context["backend"],
            "experiment_id": context["experiment_id"],
        }
        verdict = judge.judge(context["recipe_id"], results_dict)
        print(f"[train] Judge verdict: {verdict.verdict.value} — {verdict.reasoning}")

        summary_metrics = _aggregate_eval_results(eval_results) or (train_result.metrics or {})
        db.insert_experiment(
            {
                "id": context["experiment_id"],
                "recipe_id": context["recipe_id"],
                "config_hash": context["config_hash"],
                "status": train_result.status,
                "trainer_type": context["trainer_type"],
                "backend": context["backend"],
                "model_base": context["model_base"],
                "metrics_json": summary_metrics,
                "train_metrics_json": train_result.metrics or {},
                "recipe_json": context["recipe"],
                "budget_json": context["budget"],
                "checkpoint_path": train_result.checkpoint_path,
                "error": train_result.error,
            }
        )
        if eval_results:
            db.insert_eval_runs(
                [
                    {
                        "experiment_id": context["experiment_id"],
                        "benchmark": result.benchmark,
                        "seed": result.seed,
                        "metrics_json": result.metrics,
                        "details_json": result.details,
                    }
                    for result in eval_results
                ]
            )
        db.insert_verdict(
            {
                "experiment_id": context["experiment_id"],
                "verdict": verdict.verdict.value,
                "reasoning": verdict.reasoning,
                "checks_json": verdict.checks,
                "suggestions_json": verdict.suggestions,
                "research_suggestions_json": verdict.research_suggestions,
            }
        )
        _persist_ablation_record(
            db,
            context["recipe"],
            context["experiment_id"],
            summary_metrics,
        )

        _mark_external_execution_tasks_completed(
            db,
            context["recipe_id"],
            context["experiment_id"],
        )
        tasks = _post_train_tasks(
            recipe_id=context["recipe_id"],
            experiment_id=context["experiment_id"],
            train_result=train_result,
            verdict=verdict,
            eval_results=eval_results,
            expected_seeds=results_dict["expected_seeds"],
            ablation_config=context["recipe"].get("ablation", []) if isinstance(context["recipe"], dict) else [],
            result_db=db,
        )
        for task in tasks:
            db.upsert_task(task)

        summary_path = _persist_import_summary(
            bundle_dir,
            recipe_id=context["recipe_id"],
            experiment_id=context["experiment_id"],
            train_result=train_result,
            eval_results=eval_results,
            verdict=verdict,
        )
        artifact_rows = [
            {"kind": "external_import_summary", "path": str(summary_path)},
        ]
        if train_result.checkpoint_path:
            artifact_rows.append({"kind": "checkpoint", "path": train_result.checkpoint_path})
        for result in eval_results:
            report_path = result.details.get("report_path")
            if report_path:
                artifact_rows.append(
                    {
                        "kind": "swebench_report",
                        "path": str(report_path),
                        "metadata": {"benchmark": result.benchmark, "seed": result.seed},
                    }
                )
        ledger_paths = _write_task_ledger(
            db,
            context["recipe_id"],
            context["experiment_id"],
            context["plan_dir"],
        )
        if ledger_paths:
            artifact_rows.extend(
                [
                    {"kind": "task_ledger_json", "path": str(ledger_paths[0])},
                    {"kind": "task_ledger_markdown", "path": str(ledger_paths[1])},
                ]
            )
            print(f"[train] Task ledger written to {ledger_paths[0]}")
        _persist_artifacts(db, context["recipe_id"], context["experiment_id"], artifact_rows)

        report_dir = Path(report_output) if report_output else context["plan_dir"] / "reports"
        from cli.report import run_report

        run_report(
            argparse.Namespace(
                experiment_id=context["experiment_id"],
                recipe_id=None,
                format=report_format,
                output=str(report_dir),
            )
        )
        report_ext = ".tex" if report_format == "latex" else ".md"
        report_path = report_dir / f"report{report_ext}"
        if report_path.exists():
            _persist_artifacts(
                db,
                context["recipe_id"],
                context["experiment_id"],
                [
                    {
                        "kind": f"auto_report_{report_format}",
                        "path": str(report_path),
                    }
                ],
            )
        db.upsert_task(
            _make_task(
                recipe_id=context["recipe_id"],
                experiment_id=context["experiment_id"],
                kind="generate_report",
                title="Generate and review experiment report",
                status="completed",
                priority="low",
            )
        )

        print("[train] Results imported and stored in DB.")
        if report_path.exists():
            print(f"[train] Report written to {report_path}")
        print("\n[train] === Summary ===")
        print(f"[train] Recipe     : {context['recipe_id']}")
        print(f"[train] Trainer    : {context['trainer_type']} / {context['backend']}")
        print(f"[train] Status     : {train_result.status}")
        print(f"[train] Verdict    : {verdict.verdict.value}")
        print("[train] Done.")
    finally:
        db.close()


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
    # Handle --import-results mode
    import_results_dir = getattr(args, "import_results", None)
    if import_results_dir:
        _import_swe_lego_results(args)
        return

    if not getattr(args, "recipe", None):
        print("[train] Error: provide a recipe path or use --import-results for completed external runs.")
        return

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
    elif config.backend == "swe_lego":
        try:
            from trainers.swe_lego import (
                build_swe_lego_launcher_bundle,
                write_swe_lego_launcher_bundle,
            )

            launcher_bundle = build_swe_lego_launcher_bundle(config.__dict__, output_dir)
            launcher_bundle["experiment_id"] = experiment_id
            launcher_bundle["recipe"] = recipe
            launcher_bundle["config_hash"] = config_hash
            launcher_paths = write_swe_lego_launcher_bundle(launcher_bundle)
            print(f"[train] SWE-Lego launch bundle ready: {launcher_paths['run_script']}")

            # Auto SLURM pipeline if slurm config present and not --no-submit
            slurm_cfg = config.budget.get("slurm")
            if slurm_cfg and not getattr(args, "no_submit", False) and not dry_run:
                from trainers.slurm.submitter import run_swe_lego_pipeline

                pipeline_result = run_swe_lego_pipeline(
                    launcher_paths["bundle_dir"], slurm_cfg
                )
                print(f"[train] SLURM pipeline submitted: {pipeline_result}")

                # Persist SLURM job IDs in DB for tracking
                if db is not None:
                    slurm_records = []
                    for stage, jid in pipeline_result["job_ids"].items():
                        slurm_records.append({
                            "job_id": jid,
                            "experiment_id": experiment_id,
                            "recipe_id": recipe_id,
                            "pipeline_id": pipeline_result["pipeline_id"],
                            "stage": stage,
                            "bundle_dir": str(launcher_paths["bundle_dir"]),
                            "status": "PENDING",
                        })
                    db.insert_slurm_jobs(slurm_records)
                    print(f"[train] Tracked {len(slurm_records)} SLURM jobs in DB")
        except Exception as exc:
            print(f"[train] Error building SWE-Lego launch bundle: {exc}")
            return
    elif config.backend in (_EXTERNAL_PREPARED_BACKENDS - {"tinyzero", "swe_lego"}):
        try:
            from trainers.upstream import (
                build_upstream_launcher_bundle,
                write_upstream_launcher_bundle,
            )

            launcher_bundle = build_upstream_launcher_bundle(config.__dict__, output_dir)
            launcher_paths = write_upstream_launcher_bundle(launcher_bundle)
            print(f"[train] {_backend_label(config.backend)} launch bundle ready: {launcher_paths['run_script']}")
        except Exception as exc:
            print(f"[train] Error building {_backend_label(config.backend)} launch bundle: {exc}")
            return

    if dry_run:
        plan_reason = "dry-run requested"
    elif launcher_bundle is not None:
        plan_reason = f"{_backend_label(config.backend)} launch bundle prepared"
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

    if launcher_bundle is not None:
        json_path, md_path = _write_execution_plan(execution_plan, output_dir)
        # If SLURM jobs were submitted, mark as "running"; otherwise "prepared"
        slurm_submitted = (
            config.backend == "swe_lego"
            and config.budget.get("slurm")
            and not getattr(args, "no_submit", False)
        )
        exp_status = "running" if slurm_submitted else "prepared"
        if db is not None and config_hash is not None:
            db.insert_experiment(
                {
                    "id": experiment_id,
                    "recipe_id": recipe_id,
                    "config_hash": config_hash,
                    "status": exp_status,
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
            artifact_rows.extend(_launcher_artifacts(config.backend, launcher_paths))
            _persist_artifacts(db, recipe_id, experiment_id, artifact_rows)
            ledger_paths = _write_task_ledger(db, recipe_id, experiment_id, Path(execution_plan["artifact_dir"]))
            if ledger_paths:
                print(f"[train] Task ledger written to {ledger_paths[0]}")
            db.close()
            db = None
        print(f"[train] {_backend_label(config.backend)} backend selected — external launch bundle prepared.")
        print(f"[train] Execution plan written to {json_path}")
        print(f"[train] Plan summary written to {md_path}")
        if launcher_paths:
            if launcher_paths.get("launcher_json"):
                print(f"[train] Launcher JSON  : {launcher_paths['launcher_json']}")
            if launcher_paths.get("env"):
                print(f"[train] Env template   : {launcher_paths['env']}")
            if launcher_paths.get("run_script"):
                print(f"[train] Run script     : {launcher_paths['run_script']}")
            if launcher_paths.get("notes"):
                print(f"[train] Notes          : {launcher_paths['notes']}")
        print("\n[train] === Summary ===")
        print(f"[train] Recipe     : {recipe_id}")
        print(f"[train] Experiment : {experiment_id}")
        print(f"[train] Trainer    : {config.trainer_type} / {config.backend}")
        print(f"[train] Status     : {exp_status}")
        if slurm_submitted:
            print("[train] SLURM jobs tracked — use `act sync` to check completion and import results.")
            print(f"[train] Use `act status --recipe-id {recipe_id} --slurm` for live SLURM status.")
        print("[train] Done.")
        return

    # ------------------------------------------------------------------
    # 4. Select and instantiate trainer (registry-based dispatch)
    # ------------------------------------------------------------------
    trainer = None
    trainer_init_error = None
    try:
        from trainers.registry import get_trainer_class

        trainer_cls = get_trainer_class(config.trainer_type, config.backend)
        if trainer_cls is not None:
            trainer = trainer_cls(config.__dict__, output_dir)
            print(f"[train] Using {trainer_cls.__name__} ({config.backend} backend).")
        else:
            print(f"[train] No registered trainer for {config.trainer_type}/{config.backend}")
            trainer_init_error = f"no registered trainer for {config.trainer_type}/{config.backend}"
    except ImportError as exc:
        print(f"[train] Trainer module not available: {exc}")
        trainer_init_error = "trainer module not available"
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
                        "research_suggestions_json": verdict.research_suggestions,
                    }
                )
            _persist_ablation_record(
                db,
                recipe,
                experiment_id,
                summary_metrics,
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
