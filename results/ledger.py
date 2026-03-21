"""Task ledger helpers for human- and agent-readable experiment state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_task_ledger(
    *,
    recipe_id: str,
    experiment_id: str | None,
    experiment: dict[str, Any] | None,
    tasks: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    verdict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized task ledger payload."""
    completed = [task for task in tasks if task.get("status") == "completed"]
    open_tasks = [
        task
        for task in tasks
        if task.get("status") in {"pending", "blocked", "in_progress"}
    ]
    return {
        "recipe_id": recipe_id,
        "experiment_id": experiment_id,
        "status": experiment.get("status", "unknown") if experiment else "unknown",
        "latest_verdict": verdict.get("verdict") if verdict else None,
        "latest_reasoning": verdict.get("reasoning") if verdict else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "completed_tasks": len(completed),
            "open_tasks": len(open_tasks),
            "artifacts": len(artifacts),
        },
        "tasks": tasks,
        "artifacts": artifacts,
    }


def render_task_ledger_markdown(ledger: dict[str, Any]) -> str:
    """Render a compact Markdown ledger."""
    lines = [
        f"# Task Ledger: {ledger.get('recipe_id', 'unknown')}",
        "",
        f"- **Experiment ID**: {ledger.get('experiment_id') or 'n/a'}",
        f"- **Status**: {ledger.get('status', 'unknown')}",
        f"- **Latest Verdict**: {ledger.get('latest_verdict') or 'n/a'}",
        f"- **Updated**: {ledger.get('updated_at', 'n/a')}",
        "",
        "## Summary",
        f"- **Completed Tasks**: {ledger.get('summary', {}).get('completed_tasks', 0)}",
        f"- **Open Tasks**: {ledger.get('summary', {}).get('open_tasks', 0)}",
        f"- **Artifacts**: {ledger.get('summary', {}).get('artifacts', 0)}",
        "",
    ]

    tasks = ledger.get("tasks", [])
    if tasks:
        lines.extend(
            [
                "## Tasks",
                "| ID | Status | Priority | Kind | Title |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for task in tasks:
            lines.append(
                "| {id} | {status} | {priority} | {kind} | {title} |".format(
                    id=task.get("id", "?"),
                    status=task.get("status", "?"),
                    priority=task.get("priority", "?"),
                    kind=task.get("kind", "?"),
                    title=task.get("title", "?"),
                )
            )
        lines.append("")

    artifacts = ledger.get("artifacts", [])
    if artifacts:
        lines.extend(
            [
                "## Artifacts",
                "| Kind | Path |",
                "| --- | --- |",
            ]
        )
        for artifact in artifacts:
            lines.append(f"| {artifact.get('kind', '?')} | {artifact.get('path', '?')} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_task_ledger(ledger: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON and Markdown task ledgers to disk."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "task-ledger.json"
    md_path = root / "task-ledger.md"
    json_path.write_text(json.dumps(ledger, indent=2))
    md_path.write_text(render_task_ledger_markdown(ledger))
    return {
        "json": str(json_path),
        "markdown": str(md_path),
    }
