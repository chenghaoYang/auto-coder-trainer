"""Evaluation report formatting utilities."""

from typing import Any
from evaluators.base import BenchmarkResult


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format evaluation results as a Markdown table.

    Produces a Markdown table with columns for benchmark name, each
    metric, and the seed used.  Metrics are unioned across all results
    so every column appears even if only some benchmarks report it.

    Args:
        results: List of :class:`BenchmarkResult` instances.

    Returns:
        A Markdown-formatted table string.
    """
    if not results:
        return "_No results to display._"

    # Collect the superset of metric keys in stable order.
    metric_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for key in r.metrics:
            if key not in seen:
                metric_keys.append(key)
                seen.add(key)

    # Header
    headers = ["Benchmark"] + metric_keys + ["Seed"]
    separator = ["-" * max(len(h), 3) for h in headers]

    rows: list[list[str]] = [headers, separator]
    for r in results:
        row = [r.benchmark]
        for key in metric_keys:
            val = r.metrics.get(key)
            if val is None:
                row.append("—")
            elif isinstance(val, float) and not val.is_integer():
                row.append(f"{val:.2f}")
            else:
                row.append(str(int(val)) if isinstance(val, float) and val.is_integer() else str(val))
        row.append(str(r.seed))
        rows.append(row)

    # Render rows with pipe-delimited columns.
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    # Make the separator line use dashes with colons for alignment.
    lines[1] = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join(lines)


def compare_with_baseline(
    current: BenchmarkResult, baseline: BenchmarkResult
) -> str:
    """Compare *current* results with a *baseline* and return a Markdown table.

    For every metric present in either result the table shows the
    baseline value, current value, absolute delta, and a directional
    arrow indicating improvement or regression (higher is assumed
    better).

    Args:
        current: The result set being evaluated.
        baseline: The reference result set to compare against.

    Returns:
        A Markdown-formatted comparison table string.
    """
    all_keys: list[str] = []
    seen: set[str] = set()
    for key in list(baseline.metrics.keys()) + list(current.metrics.keys()):
        if key not in seen:
            all_keys.append(key)
            seen.add(key)

    headers = ["Metric", "Baseline", "Current", "Delta", ""]
    lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for key in all_keys:
        base_val = baseline.metrics.get(key)
        curr_val = current.metrics.get(key)

        base_str = f"{base_val:.2f}" if base_val is not None else "—"
        curr_str = f"{curr_val:.2f}" if curr_val is not None else "—"

        if base_val is not None and curr_val is not None:
            delta = curr_val - base_val
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f}"
            if delta > 0:
                arrow = "^"  # improvement
            elif delta < 0:
                arrow = "v"  # regression
            else:
                arrow = "="
        else:
            delta_str = "—"
            arrow = ""

        lines.append(
            f"| {key} | {base_str} | {curr_str} | {delta_str} | {arrow} |"
        )

    return "\n".join(lines)
