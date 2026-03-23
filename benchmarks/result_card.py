"""Result Card — structured experiment result representation and rendering.

Provides a ``ResultCard`` dataclass for capturing benchmark outcomes, plus
helpers for generating cards from the result database and rendering them
as Markdown or JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class ResultCard:
    """Structured result card for a benchmark run."""

    benchmark_id: str
    recipe_id: str
    model: str
    hardware: str
    timestamp: str
    metrics: dict[str, Any]
    cost: dict[str, Any]
    duration_seconds: float
    status: str
    reproducibility_info: dict[str, Any]

    # Optional extras
    experiment_id: str | None = None
    eval_runs: list[dict[str, Any]] = field(default_factory=list)
    verdict: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the card to a plain dict."""
        return asdict(self)


def generate_result_card(experiment_id: str, db: Any) -> ResultCard:
    """Pull experiment data from the ResultDB and build a ResultCard.

    Args:
        experiment_id: The experiment ID to look up.
        db: A connected ``results.db.ResultDB`` instance.

    Returns:
        A populated ``ResultCard``.

    Raises:
        ValueError: If the experiment is not found.
    """
    bundle = db.get_experiment_bundle(experiment_id)
    experiment = bundle["experiment"]
    if experiment is None:
        raise ValueError(f"Experiment {experiment_id!r} not found in database")

    metrics = experiment.get("metrics_json") or {}
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    budget = experiment.get("budget_json") or {}
    if isinstance(budget, str):
        budget = json.loads(budget)
    recipe = experiment.get("recipe_json") or {}
    if isinstance(recipe, str):
        recipe = json.loads(recipe)
    verdicts = bundle.get("verdicts", [])
    latest_verdict = verdicts[-1]["verdict"] if verdicts else None

    # Build reproducibility info from recipe + config hash
    reproducibility = {
        "config_hash": experiment.get("config_hash", ""),
        "backend": experiment.get("backend", ""),
        "trainer_type": experiment.get("trainer_type", ""),
        "recipe_version": recipe.get("version", ""),
    }
    seeds = recipe.get("eval", {}).get("seeds", [])
    if seeds:
        reproducibility["seeds"] = seeds

    # Cost info
    cost = {
        "max_gpu_hours": budget.get("max_gpu_hours"),
        "max_cost_usd": budget.get("max_cost_usd"),
        "gpu_type": budget.get("gpu_type"),
    }

    return ResultCard(
        benchmark_id=f"bench-{experiment_id}",
        recipe_id=experiment.get("recipe_id", ""),
        model=experiment.get("model_base", ""),
        hardware=budget.get("gpu_type", "unknown"),
        timestamp=experiment.get("timestamp", datetime.now(timezone.utc).isoformat()),
        metrics=metrics,
        cost=cost,
        duration_seconds=metrics.get("duration_seconds", 0.0),
        status=experiment.get("status", "unknown"),
        reproducibility_info=reproducibility,
        experiment_id=experiment_id,
        eval_runs=bundle.get("eval_runs", []),
        verdict=latest_verdict,
        error=experiment.get("error"),
    )


def render_result_card_markdown(card: ResultCard) -> str:
    """Render a ResultCard as a Markdown table / report.

    Args:
        card: The result card to render.

    Returns:
        A Markdown-formatted string.
    """
    lines: list[str] = []
    lines.append(f"# Result Card: {card.benchmark_id}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| **Benchmark ID** | {card.benchmark_id} |")
    lines.append(f"| **Recipe ID** | {card.recipe_id} |")
    lines.append(f"| **Model** | {card.model} |")
    lines.append(f"| **Hardware** | {card.hardware} |")
    lines.append(f"| **Status** | {card.status} |")
    lines.append(f"| **Duration** | {card.duration_seconds:.1f}s |")
    lines.append(f"| **Timestamp** | {card.timestamp} |")
    if card.verdict:
        lines.append(f"| **Verdict** | {card.verdict} |")
    if card.error:
        lines.append(f"| **Error** | {card.error} |")
    lines.append("")

    # Metrics
    if card.metrics:
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for key, value in card.metrics.items():
            if key == "duration_seconds":
                continue
            display = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| {key} | {display} |")
        lines.append("")

    # Cost
    if card.cost:
        lines.append("## Cost")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        for key, value in card.cost.items():
            if value is not None:
                lines.append(f"| {key} | {value} |")
        lines.append("")

    # Eval runs
    if card.eval_runs:
        lines.append("## Evaluation Runs")
        lines.append("")
        lines.append("| Benchmark | Seed | Metrics |")
        lines.append("| --- | --- | --- |")
        for run in card.eval_runs:
            bench = run.get("benchmark", "?")
            seed = run.get("seed", "?")
            run_metrics = run.get("metrics_json", {})
            if isinstance(run_metrics, str):
                try:
                    run_metrics = json.loads(run_metrics)
                except (json.JSONDecodeError, TypeError):
                    pass
            metrics_str = ", ".join(
                f"{k}={v}" for k, v in run_metrics.items()
            ) if isinstance(run_metrics, dict) else str(run_metrics)
            lines.append(f"| {bench} | {seed} | {metrics_str} |")
        lines.append("")

    # Reproducibility
    if card.reproducibility_info:
        lines.append("## Reproducibility")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        for key, value in card.reproducibility_info.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_result_card_json(card: ResultCard) -> str:
    """Render a ResultCard as pretty-printed JSON.

    Args:
        card: The result card to render.

    Returns:
        A JSON string.
    """
    return json.dumps(card.to_dict(), indent=2, default=str)


def compare_result_cards(cards: list[ResultCard]) -> str:
    """Generate a side-by-side Markdown comparison table for multiple cards.

    Args:
        cards: A list of ``ResultCard`` instances to compare.

    Returns:
        A Markdown table string.
    """
    if not cards:
        return "No result cards to compare.\n"

    if len(cards) == 1:
        return render_result_card_markdown(cards[0])

    # Collect all metric keys across all cards
    all_metric_keys: list[str] = []
    seen: set[str] = set()
    for card in cards:
        for key in card.metrics:
            if key not in seen and key != "duration_seconds":
                all_metric_keys.append(key)
                seen.add(key)

    lines: list[str] = []
    lines.append("# Benchmark Comparison")
    lines.append("")

    # Header row
    header = "| Field |"
    separator = "| --- |"
    for card in cards:
        header += f" {card.benchmark_id} |"
        separator += " --- |"
    lines.append(header)
    lines.append(separator)

    # Overview rows
    overview_fields = [
        ("Recipe ID", lambda c: c.recipe_id),
        ("Model", lambda c: c.model),
        ("Hardware", lambda c: c.hardware),
        ("Status", lambda c: c.status),
        ("Duration (s)", lambda c: f"{c.duration_seconds:.1f}"),
        ("Verdict", lambda c: c.verdict or "n/a"),
    ]
    for label, accessor in overview_fields:
        row = f"| **{label}** |"
        for card in cards:
            row += f" {accessor(card)} |"
        lines.append(row)

    # Metric rows
    if all_metric_keys:
        lines.append(f"| | {'| '.join([''] * len(cards))}|")
        for key in all_metric_keys:
            row = f"| **{key}** |"
            for card in cards:
                value = card.metrics.get(key)
                if value is None:
                    display = "n/a"
                elif isinstance(value, float):
                    display = f"{value:.4f}"
                else:
                    display = str(value)
                row += f" {display} |"
            lines.append(row)

    # Cost rows
    lines.append(f"| | {'| '.join([''] * len(cards))}|")
    cost_fields = ["max_gpu_hours", "max_cost_usd", "gpu_type"]
    for cf in cost_fields:
        row = f"| **{cf}** |"
        for card in cards:
            value = card.cost.get(cf)
            row += f" {value if value is not None else 'n/a'} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
