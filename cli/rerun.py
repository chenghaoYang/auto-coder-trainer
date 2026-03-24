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


def _get_recipe_from_db(recipe_id: str, db) -> dict | None:
    """Retrieve the stored recipe dict from the most recent experiment."""
    experiments = db.find_by_recipe(recipe_id)
    for exp in experiments:
        recipe_json = exp.get("recipe_json")
        if isinstance(recipe_json, dict) and recipe_json:
            return recipe_json
    return None


def _set_nested_value(document: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path like ``trainer.params.lr`` on a recipe dict."""
    parts = [part for part in path.split(".") if part]
    if not parts:
        return

    cursor: dict[str, Any] = document
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
    cursor = next_value
    cursor[parts[-1]] = value


def _normalize_ablation_targets(payload: dict[str, Any]) -> list[str]:
    """Extract target ablation expressions from task payloads."""
    raw = payload.get("targets", payload.get("missing"))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str)]
    return []


def _parse_ablation_target(target: str) -> tuple[str, Any | None]:
    """Parse ``trainer.params.lr=1e-5`` into (path, value)."""
    if "=" not in target:
        return target.strip(), None
    variable, raw_value = target.split("=", 1)
    variable = variable.strip()
    raw_value = raw_value.strip()
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return variable, value


def _ablation_values_match(expected: Any, actual: Any) -> bool:
    """Best-effort equality check for ablation values across JSON/string forms."""
    if expected == actual:
        return True
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)
    return str(expected) == str(actual)


def _select_ablation_variants(
    ablation_specs: list[dict[str, Any]],
    targets: list[str],
) -> list[tuple[dict[str, Any], Any]]:
    """Select only the ablation variants requested by the task payload."""
    if not targets:
        return [
            (spec, value)
            for spec in ablation_specs
            for value in spec.get("values", [])
        ]

    parsed_targets = [_parse_ablation_target(target) for target in targets]
    selected: list[tuple[dict[str, Any], Any]] = []
    seen: set[tuple[str, str]] = set()

    for spec in ablation_specs:
        variable = spec.get("variable", "")
        for value in spec.get("values", []):
            for target_variable, target_value in parsed_targets:
                if target_variable != variable:
                    continue
                if target_value is not None and not _ablation_values_match(target_value, value):
                    continue
                key = (variable, json.dumps(value, sort_keys=True, default=str))
                if key not in seen:
                    seen.add(key)
                    selected.append((spec, value))
                break

    return selected


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
    targets = _normalize_ablation_targets(payload)
    missing = targets or payload.get("missing")
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

    selected_variants = _select_ablation_variants(ablation_specs, targets)
    if targets and not selected_variants:
        print(f"[rerun]   Warning: no ablation variants matched task targets: {targets}")
        return False

    if dry_run:
        print(f"[rerun]   (dry-run) Would generate {len(selected_variants)} ablation variant(s):")
        for spec, value in selected_variants:
            name = spec.get("name", spec.get("variable", "?"))
            print(f"[rerun]     - {name}: {value}")
        return True

    # Generate and run ablation variants by modifying the recipe for each variant
    import copy
    import tempfile

    success_count = 0
    for spec, value in selected_variants:
        variable = spec.get("variable", "")
        print(f"[rerun]   Running ablation variant: {variable}={value}")
        recipe_variant = copy.deepcopy(recipe)
        _set_nested_value(recipe_variant, variable, value)
        recipe_variant["ablation_run"] = {
            "parent_recipe_id": recipe_id,
            "name": spec.get("name", variable),
            "variable": variable,
            "value": value,
        }
        variant_suffix = f"{variable}={value}"
        recipe_variant["name"] = (
            f"{recipe_variant.get('name', recipe_id)} [{variant_suffix}]"
        )

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
        print("[rerun]   (dry-run) Would leave this task blocked for manual follow-up.")

    # Execution steps are informational only. The task should stay blocked
    # until an external run finishes and cli.train/_import_swe_lego_results
    # marks it completed after importing results.
    return True


def run_rerun(args: argparse.Namespace) -> None:
    """Auto-dispatch pending tasks for a recipe.

    Reads open tasks from the DB, determines the appropriate action for
    each one, and executes it. Automatic tasks move through the normal
    in_progress -> completed flow; external/manual execution steps stay
    blocked until a later result import closes them.
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
    awaiting_count = 0

    for task in dispatchable:
        task_id = task.get("id", "?")
        kind = task.get("kind", "?")
        print(f"\n[rerun] Dispatching task {task_id}: {_describe_task(task)}")

        # Mark as in_progress (unless dry-run)
        if not dry_run and kind != "execution_step":
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
        if kind == "execution_step":
            if not dry_run:
                note = "Awaiting manual/external execution."
                task["notes"] = note if not task.get("notes") else f"{task['notes']} | {note}"
                _mark_task(db, task, "blocked")
            awaiting_count += 1
            continue

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
    print(f"[rerun] Awaiting   : {awaiting_count}")
    print(f"[rerun] Failed     : {failed_count}")
    if dry_run:
        print("[rerun] Mode       : dry-run (no changes applied)")
    print("[rerun] Done.")
