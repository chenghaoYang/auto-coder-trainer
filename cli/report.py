"""Report command — generate technical reports from experiment results.

Queries the result DB and generates Markdown or LaTeX reports
with method descriptions, results tables, ablation analysis, and conclusions.
"""

import argparse
import sys
from pathlib import Path


def run_report(args: argparse.Namespace) -> None:
    """Execute the report generation pipeline.

    Pipeline:
        1. Query result DB for experiment(s)
        2. Gather all related ablation and verdict data
        3. Generate formatted report (Markdown or LaTeX)
        4. Save to output directory
    """
    experiment_id = getattr(args, "experiment_id", None)
    recipe_id = getattr(args, "recipe_id", None)
    fmt = getattr(args, "format", "markdown")
    output_dir = Path(getattr(args, "output", "reports/"))

    if not experiment_id and not recipe_id:
        print("[report] Error: provide --experiment-id or --recipe-id")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Connect to results DB
    # ------------------------------------------------------------------
    try:
        from results.db import ResultDB
    except ImportError:
        print("[report] Error: results.db module not available.")
        sys.exit(1)

    db = ResultDB()
    try:
        db.connect()
    except Exception as exc:
        print(f"[report] Error connecting to results DB: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Query experiments
    # ------------------------------------------------------------------
    experiments = []
    if experiment_id:
        print(f"[report] Looking up experiment: {experiment_id}")
        exp = db.get_experiment(experiment_id)
        if exp:
            experiments.append(exp)
        else:
            print(f"[report] Experiment '{experiment_id}' not found.")
    elif recipe_id:
        print(f"[report] Looking up experiments for recipe: {recipe_id}")
        experiments = db.find_by_recipe(recipe_id)

    if not experiments:
        print("[report] No experiments found — nothing to report.")
        db.close()
        return

    print(f"[report] Found {len(experiments)} experiment(s)")

    experiment_ids = [e["id"] for e in experiments]

    # ------------------------------------------------------------------
    # 3. Generate report
    # ------------------------------------------------------------------
    try:
        from results.report_generator import ReportGenerator
    except ImportError:
        print("[report] Warning: ReportGenerator not available — generating basic report.")
        _generate_basic_report(experiments, fmt, output_dir)
        db.close()
        return

    generator = ReportGenerator(result_db=db)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if fmt == "latex":
            output_file = output_dir / "report.tex"
            content = generator.generate_latex(experiment_ids, output_file)
            print(f"[report] LaTeX report written to {output_file}")
        else:
            output_file = output_dir / "report.md"
            content = generator.generate_markdown(experiment_ids, output_file)
            print(f"[report] Markdown report written to {output_file}")

        # Also print to stdout for convenience
        if content:
            print("\n" + content)
    except NotImplementedError:
        print("[report] ReportGenerator methods not fully implemented — falling back to basic report.")
        _generate_basic_report(experiments, fmt, output_dir)
    except Exception as exc:
        print(f"[report] Error generating report: {exc}")
        _generate_basic_report(experiments, fmt, output_dir)

    db.close()
    print("[report] Done.")


def _generate_basic_report(
    experiments: list[dict],
    fmt: str,
    output_dir: Path,
) -> None:
    """Produce a minimal report when the full ReportGenerator is unavailable."""
    lines: list[str] = []

    if fmt == "latex":
        lines.append("\\documentclass{article}")
        lines.append("\\begin{document}")
        lines.append("\\section{Experiment Results}")
        for exp in experiments:
            lines.append(f"\\subsection{{{exp.get('id', '?')}}}")
            lines.append(f"Recipe: {exp.get('recipe_id', '?')} \\\\")
            lines.append(f"Status: {exp.get('status', '?')} \\\\")
            metrics = exp.get("metrics_json", {})
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    lines.append(f"{k}: {v} \\\\")
        lines.append("\\end{document}")
        ext = ".tex"
    else:
        lines.append("# Experiment Results\n")
        for exp in experiments:
            lines.append(f"## {exp.get('id', '?')}\n")
            lines.append(f"- **Recipe**: {exp.get('recipe_id', '?')}")
            lines.append(f"- **Status**: {exp.get('status', '?')}")
            lines.append(f"- **Trainer**: {exp.get('trainer_type', '?')} / {exp.get('backend', '?')}")
            lines.append(f"- **Model**: {exp.get('model_base', '?')}")
            metrics = exp.get("metrics_json", {})
            if isinstance(metrics, dict) and metrics:
                lines.append("- **Metrics**:")
                for k, v in metrics.items():
                    lines.append(f"  - {k}: {v}")
            lines.append("")
        ext = ".md"

    content = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"report{ext}"
    output_file.write_text(content)
    print(f"[report] Basic report written to {output_file}")
    print("\n" + content)
