"""Rerun command — auto-dispatch pending tasks created by the judge.

When the experiment judge issues NEEDS_RERUN or NEEDS_ABLATION verdicts,
it writes tasks with status=pending to the DB.  This command reads those
open tasks and dispatches the appropriate action for each one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Task kinds that this command knows how to dispatch automatically.
_DISPATCHABLE_KINDS = {
    "rerun_seed",
    "rerun_experiment",
    "run_ablation",
    "generate_report",
    "execution_step",
}


def _mark_task(db, task: dict[str, Any], status: str) -> None:
    """Update a task's status in the DB."""
    task_copy = dict(task)
    task_copy["status"] = status
    # payload_json may already be deserialized by ResultDB._row_to_dict
    db.upsert_task(task_copy)


def _describe_task(task: dict[str, Any]) -> str:
    """Return a one-line human-readable description of a task."""
    kind = task.get("kind", "?")
    title = task.get("title", "?")
    priority = task.get("priority", "medium")
    return f"[{priority}] {kind}: {title}"


def _find_recipe_path(recipe_id: str, db) -> str | None:
    """Try to locate the recipe JSON path from the DB experiment records."""
    experiments = db.find_by_recipe(recipe_id)
    for exp in experiments:
        recipe_json = exp.get("recipe_json")
        if isinstance(recipe_json, dict):
            # The recipe is stored inline — we can use it directly later
            return None
        # Check artifacts for the recipe file
    artifacts = db.get_artifacts_for_recipe(recipe_id)
    for artifact in artifacts:
        if artifact.get("kind") in ("recipe", "recipe_json"):
            return artifact.get("path")
    return None


def _get_recipe_from_db(recipe_id: str, db) -> dict | None:
    """Retrieve the stored recipe dict from the most recent experiment."""
    experiments = db.find_by_recipe(recipe_id)
    for exp in experiments:
        recipe_json = exp.get("recipe_json")
        if isinstance(recipe_json, dict) and recipe_json:
            return recipe_json
    return None


def _dispatch_rerun_seeds(
    task: dict[str, Any],
    recipe_id: str,
    recipe: dict | None,
    db,
    *,
    dry_run: bool,
) -> bool:
    """Dispatch a rerun_seed or rerun_experiment task."""
    payload = task.get("payload_json", {}) or {}

    if task["kind"] == "rerun_seed":
        seed = payload.get("seed")
        print(f"[rerun]   Re-running evaluation for missing seed {seed}")
    else:
        print("[rerun]   Re-running full experiment to satisfy judge requirements")
        suggestions = payload.get("suggestions")
        if suggestions:
            print(f"[rerun]   Suggestions: {suggestions}")

    if recipe is None:
        print("[rerun]   Warning: no recipe found in DB — cannot re-invoke training automatically.")
        return False

    if dry_run:
        print("[rerun]   (dry-run) Would invoke `act train` with the stored recipe.")
        return True

    # Build a minimal argparse.Namespace matching what run_train expects
    import tempfile

    recipe_copy = dict(recipe)
    if task["kind"] == "rerun_seed":
        seed = payload.get("seed")
        if seed is not None:
            eval_cfg = recipe_copy.get("eval", {})
            if isinstance(eval_cfg, dict):
                eval_cfg["seeds"] = [seed]

    tmp_fd = tempfile.NamedTemporaryFile(suffix=".json", prefix="rerun_recipe_", delete=False)
    tmp_file = Path(tmp_fd.name)
    tmp_fd.close()
    tmp_file.write_text(json.dumps(recipe_copy, indent=2))

    try:
        from cli.train import run_train

        train_args = argparse.Namespace(
            recipe=str(tmp_file),
            output_dir="outputs/",
            dry_run=False,
        )
        run_train(train_args)
        return True
    except Exception as exc:
        print(f"[rerun]   Error during re-run: {exc}")
        return False
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def _dispatch_run_ablation(
    task: dict[str, Any],
    recipe_id: str,
    recipe: dict | None,
    db,
    *,
    dry_run: bool,
) -> bool:
    """Dispatch a run_ablation task."""
    payload = task.get("payload_json", {}) or {}
    missing = payload.get("missing")
    suggestions = payload.get("suggestions")

    if missing:
        print(f"[rerun]   Ablation target: {missing}")
    if suggestions:
        print(f"[rerun]   Suggestions: {suggestions}")

    if recipe is None:
        print("[rerun]   Warning: no recipe found in DB — cannot generate ablation variants.")
        return False

    ablation_specs = recipe.get("ablation", [])
    if not ablation_specs:
        print("[rerun]   Warning: recipe has no ablation specs defined.")
        return False

    if dry_run:
        print(f"[rerun]   (dry-run) Would generate {len(ablation_specs)} ablation variant(s):")
        for spec in ablation_specs:
            name = spec.get("name", spec.get("variable", "?"))
            values = spec.get("values", [])
            print(f"[rerun]     - {name}: {values}")
        return True

    # Generate and run ablation variants by modifying the recipe for each variant
    import copy
    import tempfile

    success_count = 0
    for spec in ablation_specs:
        variable = spec.get("variable", "")
        values = spec.get("values", [])
        for value in values:
            print(f"[rerun]   Running ablation variant: {variable}={value}")
            recipe_variant = copy.deepcopy(recipe)
            # Apply the ablation override to the training config
            training_cfg = recipe_variant.get("training", {})
            if isinstance(training_cfg, dict):
                training_cfg[variable] = value
                recipe_variant["training"] = training_cfg

            tmp_fd = tempfile.NamedTemporaryFile(suffix=".json", prefix="ablation_", delete=False)
            tmp_file = Path(tmp_fd.name)
            tmp_fd.close()
            tmp_file.write_text(json.dumps(recipe_variant, indent=2))
            try:
                from cli.train import run_train

                train_args = argparse.Namespace(
                    recipe=str(tmp_file),
                    output_dir="outputs/",
                    dry_run=False,
                )
                run_train(train_args)
                success_count += 1
            except Exception as exc:
                print(f"[rerun]   Error running ablation {variable}={value}: {exc}")
            finally:
                if tmp_file.exists():
                    tmp_file.unlink()

    print(f"[rerun]   Completed {success_count} ablation variant(s).")
    return success_count > 0


def _dispatch_generate_report(
    task: dict[str, Any],
    recipe_id: str,
    *,
    dry_run: bool,
) -> bool:
    """Dispatch a generate_report task."""
    print(f"[rerun]   Generating report for recipe: {recipe_id}")

    if dry_run:
        print("[rerun]   (dry-run) Would invoke `act report --recipe-id {recipe_id}`.")
        return True

    try:
        from cli.report import run_report

        report_args = argparse.Namespace(
            experiment_id=None,
            recipe_id=recipe_id,
            format="markdown",
            output="reports/",
        )
        run_report(report_args)
        return True
    except Exception as exc:
        print(f"[rerun]   Error generating report: {exc}")
        return False


def _dispatch_execution_step(
    task: dict[str, Any],
    recipe_id: str,
    *,
    dry_run: bool,
) -> bool:
    """Handle an execution_step task (external launcher — informational only)."""
    title = task.get("title", "?")
    payload = task.get("payload_json", {}) or {}
    mode = payload.get("mode", "?")

    print(f"[rerun]   Execution step ({mode}): {title}")
    print("[rerun]   This step requires manual execution (external launcher).")

    if dry_run:
        print("[rerun]   (dry-run) Would mark as in_progress for manual follow-up.")

    # Execution steps are informational — we cannot auto-run them
    return True


def run_rerun(args: argparse.Namespace) -> None:
    """Auto-dispatch pending tasks for a recipe.

    Reads open tasks from the DB, determines the appropriate action for
    each one, and executes it.  Tasks are marked in_progress before
    starting and completed on success.
    """
    recipe_id = args.recipe_id
    dry_run = getattr(args, "dry_run", False)

    print(f"[rerun] Looking up open tasks for recipe: {recipe_id}")

    # ------------------------------------------------------------------
    # 1. Connect to results DB
    # ------------------------------------------------------------------
    try:
        from results.db import ResultDB
    except ImportError:
        print("[rerun] Error: results.db module not available.")
        sys.exit(1)

    db = ResultDB()
    try:
        db.connect()
    except Exception as exc:
        print(f"[rerun] Error connecting to results DB: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Fetch open tasks
    # ------------------------------------------------------------------
    try:
        open_tasks = db.get_open_tasks(recipe_id=recipe_id)
    except Exception as exc:
        print(f"[rerun] Error fetching tasks: {exc}")
        db.close()
        sys.exit(1)

    dispatchable = [t for t in open_tasks if t.get("kind") in _DISPATCHABLE_KINDS]
    skipped = [t for t in open_tasks if t.get("kind") not in _DISPATCHABLE_KINDS]

    if not open_tasks:
        print("[rerun] No open tasks found — nothing to dispatch.")
        db.close()
        return

    print(f"[rerun] Found {len(open_tasks)} open task(s), {len(dispatchable)} dispatchable.")
    if skipped:
        print(f"[rerun] Skipping {len(skipped)} task(s) with non-dispatchable kinds:")
        for task in skipped:
            print(f"[rerun]   - {_describe_task(task)}")

    if not dispatchable:
        print("[rerun] No dispatchable tasks — nothing to do.")
        db.close()
        return

    # ------------------------------------------------------------------
    # 3. Load recipe from DB (needed for rerun/ablation dispatch)
    # ------------------------------------------------------------------
    recipe = _get_recipe_from_db(recipe_id, db)
    if recipe is None:
        print("[rerun] Warning: could not find stored recipe in DB experiments.")

    # ------------------------------------------------------------------
    # 4. Dispatch each task
    # ------------------------------------------------------------------
    completed_count = 0
    failed_count = 0

    for task in dispatchable:
        task_id = task.get("id", "?")
        kind = task.get("kind", "?")
        print(f"\n[rerun] Dispatching task {task_id}: {_describe_task(task)}")

        # Mark as in_progress (unless dry-run)
        if not dry_run:
            _mark_task(db, task, "in_progress")

        success = False
        try:
            if kind in ("rerun_seed", "rerun_experiment"):
                success = _dispatch_rerun_seeds(
                    task, recipe_id, recipe, db, dry_run=dry_run,
                )
            elif kind == "run_ablation":
                success = _dispatch_run_ablation(
                    task, recipe_id, recipe, db, dry_run=dry_run,
                )
            elif kind == "generate_report":
                success = _dispatch_generate_report(
                    task, recipe_id, dry_run=dry_run,
                )
            elif kind == "execution_step":
                success = _dispatch_execution_step(
                    task, recipe_id, dry_run=dry_run,
                )
        except Exception as exc:
            print(f"[rerun]   Unexpected error dispatching task: {exc}")
            success = False

        # Mark result (unless dry-run)
        if not dry_run:
            if success:
                _mark_task(db, task, "completed")
                completed_count += 1
            else:
                _mark_task(db, task, "pending")
                failed_count += 1
        else:
            if success:
                completed_count += 1
            else:
                failed_count += 1

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    db.close()
    print(f"\n[rerun] === Summary ===")
    print(f"[rerun] Recipe     : {recipe_id}")
    print(f"[rerun] Dispatched : {len(dispatchable)}")
    print(f"[rerun] Completed  : {completed_count}")
    print(f"[rerun] Failed     : {failed_count}")
    if dry_run:
        print("[rerun] Mode       : dry-run (no changes applied)")
    print("[rerun] Done.")
