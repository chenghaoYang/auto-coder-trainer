"""Status command — summarize tracked experiments, artifacts, and open tasks."""

from __future__ import annotations

import argparse
from pathlib import Path


def _render_status_report(
    *,
    recipe_id: str | None,
    experiments: list[dict],
    tasks: list[dict],
) -> str:
    lines = ["# Auto-Coder-Trainer Status", ""]
    if recipe_id:
        lines.append(f"- **Recipe Filter**: {recipe_id}")
    lines.append(f"- **Tracked Experiments**: {len(experiments)}")
    lines.append(f"- **Visible Tasks**: {len(tasks)}")
    lines.append("")

    if tasks:
        lines.extend(
            [
                "## Tasks",
                "| ID | Recipe | Experiment | Status | Priority | Kind | Title |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for task in tasks:
            lines.append(
                "| {id} | {recipe_id} | {experiment_id} | {status} | {priority} | {kind} | {title} |".format(
                    id=task.get("id", "?"),
                    recipe_id=task.get("recipe_id", "?"),
                    experiment_id=task.get("experiment_id") or "n/a",
                    status=task.get("status", "?"),
                    priority=task.get("priority", "?"),
                    kind=task.get("kind", "?"),
                    title=task.get("title", "?"),
                )
            )
        lines.append("")

    if experiments:
        lines.extend(
            [
                "## Experiments",
                "| ID | Recipe | Status | Trainer | Backend | Metrics |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for experiment in experiments:
            metrics = experiment.get("metrics_json", {})
            metric_text = ", ".join(
                f"{key}={value}"
                for key, value in sorted(metrics.items())
                if isinstance(value, (int, float))
            ) if isinstance(metrics, dict) else ""
            lines.append(
                "| {id} | {recipe_id} | {status} | {trainer_type} | {backend} | {metrics} |".format(
                    id=experiment.get("id", "?"),
                    recipe_id=experiment.get("recipe_id", "?"),
                    status=experiment.get("status", "?"),
                    trainer_type=experiment.get("trainer_type", "?"),
                    backend=experiment.get("backend", "?"),
                    metrics=metric_text or "-",
                )
            )
        lines.append("")

    if not tasks and not experiments:
        lines.append("_No tracked experiments or tasks yet._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_status(args: argparse.Namespace) -> None:
    """Print a project-wide status summary."""
    from results.db import ResultDB

    recipe_id = getattr(args, "recipe_id", None)
    open_only = getattr(args, "open_only", False)
    output = getattr(args, "output", None)

    db = ResultDB()
    try:
        db.connect()
    except Exception as exc:
        print(f"[status] Error connecting to results DB: {exc}")
        return

    try:
        experiments = db.list_experiments(recipe_id=recipe_id, limit=None)
        tasks = db.get_open_tasks(recipe_id=recipe_id) if open_only else db.get_tasks(recipe_id=recipe_id)
        report = _render_status_report(
            recipe_id=recipe_id,
            experiments=experiments,
            tasks=tasks,
        )
    finally:
        db.close()

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"[status] Report written to {output_path}")

    print(report)
