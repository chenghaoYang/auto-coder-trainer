"""Sync command — poll SLURM jobs and auto-import completed results.

When SWE-Lego training is submitted via SLURM, the experiment enters
"running" status.  This command checks whether SLURM jobs have finished,
imports results, triggers the judge, and generates a report.

Usage:
    act sync --recipe-id recipe-baseline-sft-001
    act sync --recipe-id recipe-baseline-sft-001 --dry-run
    act sync                                        # sync all active jobs
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# SLURM states that indicate a job is done.
_TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT",
    "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
})


def _poll_jobs(jobs: list[dict[str, Any]], db) -> list[dict[str, Any]]:
    """Poll sacct for each non-terminal job and update the DB.

    Returns the updated job list.
    """
    try:
        from trainers.slurm.submitter import check_job_status
    except ImportError:
        print("[sync] Error: trainers.slurm.submitter not available — cannot poll SLURM.")
        sys.exit(1)

    updated: list[dict[str, Any]] = []
    for job in jobs:
        if job.get("status", "") in _TERMINAL_STATES:
            updated.append(job)
            continue
        try:
            live = check_job_status(job["job_id"])
            job["status"] = live.get("state", job.get("status", "?"))
            job["elapsed"] = live.get("elapsed", job.get("elapsed"))
            job["exit_code"] = live.get("exit_code", job.get("exit_code"))
            if job["status"] in _TERMINAL_STATES and not job.get("finished_at"):
                job["finished_at"] = datetime.now(timezone.utc).isoformat()
            db.update_slurm_job_status(
                job["job_id"],
                job["status"],
                elapsed=job.get("elapsed"),
                exit_code=job.get("exit_code"),
                finished_at=job.get("finished_at"),
            )
        except Exception as exc:
            print(f"[sync] Warning: could not poll job {job.get('job_id')}: {exc}")
        updated.append(job)
    return updated


def _group_by_pipeline(jobs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group jobs by (experiment_id, pipeline_id)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        key = f"{job.get('experiment_id', '?')}:{job.get('pipeline_id', '?')}"
        groups.setdefault(key, []).append(job)
    return groups


def _import_completed_pipeline(
    jobs: list[dict[str, Any]],
    db,
    *,
    report_format: str,
    dry_run: bool,
) -> bool:
    """Import results for a completed SLURM pipeline. Returns True on success."""
    experiment_id = jobs[0].get("experiment_id", "?")
    recipe_id = jobs[0].get("recipe_id", "?")
    experiment = db.get_experiment(experiment_id)
    backend = experiment.get("backend") if isinstance(experiment, dict) else None
    bundle_dir = jobs[0].get("bundle_dir")

    if not bundle_dir:
        # Try to derive from convention
        backend_dir = backend or "swe_lego"
        bundle_dir = f"outputs/{recipe_id}/{backend_dir}"

    bundle_path = Path(bundle_dir)
    if not bundle_path.exists():
        print(f"[sync]   Warning: bundle dir not found: {bundle_path}")
        return False

    print(f"[sync]   Importing results from {bundle_path}")
    if dry_run:
        print("[sync]   (dry-run) Would import results, run judge, and generate report.")
        return True

    # Reuse the import logic from cli/train.py
    try:
        from cli.train import _import_external_results

        import_args = argparse.Namespace(
            import_results=str(bundle_path),
            recipe_id=recipe_id,
            experiment_id=experiment_id,
            report_format=report_format,
            report_output=None,
            backend=backend,
        )
        _import_external_results(import_args)
        return True
    except Exception as exc:
        print(f"[sync]   Error importing results: {exc}")
        return False


def _handle_failed_pipeline(
    jobs: list[dict[str, Any]],
    db,
    *,
    dry_run: bool,
) -> None:
    """Mark experiment as failed and create debug tasks for failed SLURM jobs."""
    experiment_id = jobs[0].get("experiment_id", "?")
    recipe_id = jobs[0].get("recipe_id", "?")

    failed_stages = [
        j for j in jobs
        if j.get("status", "") in ("FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT", "OUT_OF_MEMORY")
    ]

    for job in failed_stages:
        stage = job.get("stage", "?")
        status = job.get("status", "?")
        job_id = job.get("job_id", "?")
        print(f"[sync]   FAILED: stage={stage}, job={job_id}, status={status}")

    if dry_run:
        print("[sync]   (dry-run) Would update experiment status to 'failed'.")
        return

    # Update experiment status
    experiment = db.get_experiment(experiment_id)
    if experiment is not None:
        experiment["status"] = "failed"
        first_failure = failed_stages[0] if failed_stages else {}
        experiment["error"] = (
            f"SLURM job {first_failure.get('job_id')} failed at stage "
            f"'{first_failure.get('stage')}' with status {first_failure.get('status')}"
        )
        db.insert_experiment(experiment)

    # Create debug task
    import hashlib
    import json

    task_raw = json.dumps({
        "recipe_id": recipe_id,
        "experiment_id": experiment_id,
        "kind": "debug_slurm_failure",
        "title": "SLURM failure",
    }, sort_keys=True)
    task_id = "task-" + hashlib.sha1(task_raw.encode()).hexdigest()[:12]

    db.upsert_task({
        "id": task_id,
        "recipe_id": recipe_id,
        "experiment_id": experiment_id,
        "kind": "debug_slurm_failure",
        "title": f"Investigate SLURM failure at stage '{failed_stages[0].get('stage')}'",
        "status": "pending",
        "priority": "high",
        "payload_json": {
            "failed_jobs": [
                {"job_id": j.get("job_id"), "stage": j.get("stage"), "status": j.get("status")}
                for j in failed_stages
            ],
            "bundle_dir": jobs[0].get("bundle_dir"),
        },
        "notes": f"Check SLURM logs in {jobs[0].get('bundle_dir', 'outputs/')}/slurm/",
    })


def _format_metric_summary(metrics: dict[str, Any]) -> str:
    """Render a concise metric summary for sync output."""
    if not isinstance(metrics, dict):
        return "-"
    preferred_keys = [
        "resolve_rate",
        "swe-bench-verified/resolve_rate",
        "swe-bench-lite/resolve_rate",
        "pass@1",
    ]
    parts: list[str] = []
    seen: set[str] = set()
    for key in preferred_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.4g}")
            seen.add(key)
    for key, value in sorted(metrics.items()):
        if key in seen or not isinstance(value, (int, float)):
            continue
        parts.append(f"{key}={value:.4g}")
        if len(parts) >= 4:
            break
    return ", ".join(parts) if parts else "-"


def _print_import_summary(db, experiment_id: str) -> None:
    """Print a concise post-import summary for the newly synced experiment."""
    experiment = db.get_experiment(experiment_id)
    if experiment is None:
        return

    verdict = db.get_latest_verdict(experiment_id)
    tasks = [
        task
        for task in db.get_tasks(
            recipe_id=experiment.get("recipe_id"),
            experiment_id=experiment_id,
        )
        if task.get("status") in {"pending", "blocked", "in_progress"}
    ]

    print(f"[sync]   Experiment     : {experiment_id}")
    print(f"[sync]   Recipe         : {experiment.get('recipe_id', '?')}")
    print(f"[sync]   Status         : {experiment.get('status', '?')}")
    print(f"[sync]   Metrics        : {_format_metric_summary(experiment.get('metrics_json', {}))}")
    if verdict:
        print(f"[sync]   Judge verdict  : {verdict.get('verdict', '?')}")
        reasoning = verdict.get("reasoning")
        if reasoning:
            print(f"[sync]   Reasoning      : {reasoning}")
    if tasks:
        print("[sync]   Open tasks:")
        for task in tasks[:5]:
            print(f"[sync]     - {task.get('kind', '?')}: {task.get('title', '?')}")


def run_sync(args: argparse.Namespace) -> None:
    """Poll SLURM jobs and auto-import completed results."""
    recipe_id = getattr(args, "recipe_id", None)
    dry_run = getattr(args, "dry_run", False)
    report_format = getattr(args, "report_format", "blog")

    print(f"[sync] Syncing SLURM jobs{f' for recipe: {recipe_id}' if recipe_id else ' (all active)'}")

    # Connect to DB
    try:
        from results.db import ResultDB
    except ImportError:
        print("[sync] Error: results.db module not available.")
        sys.exit(1)

    db = ResultDB()
    try:
        db.connect()
    except Exception as exc:
        print(f"[sync] Error connecting to results DB: {exc}")
        sys.exit(1)

    try:
        # Fetch jobs to check
        if recipe_id:
            jobs = db.get_slurm_jobs(recipe_id=recipe_id)
        else:
            jobs = db.get_active_slurm_jobs()

        if not jobs:
            print("[sync] No tracked SLURM jobs found — nothing to sync.")
            return

        active_count = sum(1 for j in jobs if j.get("status", "") not in _TERMINAL_STATES)
        print(f"[sync] Found {len(jobs)} tracked job(s), {active_count} still active.")

        # Poll sacct for live status
        if active_count > 0:
            print("[sync] Polling SLURM for live status...")
            jobs = _poll_jobs(jobs, db)

        # Print current state
        print("\n[sync] Current SLURM job status:")
        for job in jobs:
            elapsed = job.get("elapsed") or "-"
            print(f"[sync]   {job.get('stage', '?'):20s}  job {job.get('job_id', '?'):>10s}  {job.get('status', '?'):15s}  elapsed={elapsed}")

        # Group by pipeline and check for completion
        groups = _group_by_pipeline(jobs)
        imported = 0
        failed = 0
        still_active = 0

        for key, group_jobs in groups.items():
            exp_id = group_jobs[0].get("experiment_id", "?")
            all_terminal = all(j.get("status", "") in _TERMINAL_STATES for j in group_jobs)
            all_completed = all(j.get("status", "") == "COMPLETED" for j in group_jobs)

            if not all_terminal:
                still_active += 1
                print(f"\n[sync] Pipeline {key}: still running")
                continue

            if all_completed:
                print(f"\n[sync] Pipeline {key}: all jobs COMPLETED — importing results")
                success = _import_completed_pipeline(
                    group_jobs, db, report_format=report_format, dry_run=dry_run,
                )
                if success:
                    imported += 1
                    if not dry_run:
                        _print_import_summary(db, exp_id)
                else:
                    failed += 1
            else:
                print(f"\n[sync] Pipeline {key}: has failed jobs")
                _handle_failed_pipeline(group_jobs, db, dry_run=dry_run)
                failed += 1

        # Summary
        print(f"\n[sync] === Summary ===")
        print(f"[sync] Total pipelines : {len(groups)}")
        print(f"[sync] Still active    : {still_active}")
        print(f"[sync] Imported        : {imported}")
        print(f"[sync] Failed          : {failed}")
        if dry_run:
            print("[sync] Mode            : dry-run (no changes applied)")
        if still_active > 0:
            print(f"[sync] Run `act sync{f' --recipe-id {recipe_id}' if recipe_id else ''}` again later to check.")
        print("[sync] Done.")
    finally:
        db.close()
