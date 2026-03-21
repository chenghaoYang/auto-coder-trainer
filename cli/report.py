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
    detailed_experiments: list[dict] = []
    if experiment_id and hasattr(db, "get_experiment_bundle"):
        bundle = db.get_experiment_bundle(experiment_id)
        if bundle.get("experiment") is not None:
            detailed_experiments.append(bundle)
    elif recipe_id and hasattr(db, "find_by_recipe_with_details"):
        detailed_experiments = db.find_by_recipe_with_details(recipe_id)
    else:
        detailed_experiments = [{"experiment": exp, "ablations": [], "verdicts": []} for exp in experiments]

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
        _generate_basic_report(detailed_experiments, fmt, output_dir)
    except Exception as exc:
        print(f"[report] Error generating report: {exc}")
        _generate_basic_report(detailed_experiments, fmt, output_dir)

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
            record = exp.get("experiment", exp)
            lines.append(f"\\subsection{{{record.get('id', '?')}}}")
            lines.append(f"Recipe: {record.get('recipe_id', '?')} \\\\")
            lines.append(f"Status: {record.get('status', '?')} \\\\")
            verdicts = exp.get("verdicts", []) if isinstance(exp, dict) else []
            if verdicts:
                lines.append(f"Latest verdict: {verdicts[-1].get('verdict', '?')} \\\\")
            metrics = record.get("metrics_json", {})
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    lines.append(f"{k}: {v} \\\\")
            ablations = exp.get("ablations", []) if isinstance(exp, dict) else []
            for abl in ablations:
                lines.append(f"Ablation: {abl.get('variable', '?')}={abl.get('value', '?')} \\\\")
                abl_metrics = abl.get("metrics_json", {})
                if isinstance(abl_metrics, dict):
                    for k, v in abl_metrics.items():
                        lines.append(f"  {k}: {v} \\\\")
            tasks = exp.get("tasks", []) if isinstance(exp, dict) else []
            for task in tasks:
                lines.append(
                    f"Task: [{task.get('status', '?')}] {task.get('kind', '?')} - {task.get('title', '?')} \\\\"
                )
            artifacts = exp.get("artifacts", []) if isinstance(exp, dict) else []
            for artifact in artifacts:
                lines.append(
                    f"Artifact: {artifact.get('kind', '?')} -> {artifact.get('path', '?')} \\\\"
                )
        lines.append("\\end{document}")
        ext = ".tex"
    else:
        lines.append("# Experiment Results\n")
        for exp in experiments:
            record = exp.get("experiment", exp)
            lines.append(f"## {record.get('id', '?')}\n")
            lines.append(f"- **Recipe**: {record.get('recipe_id', '?')}")
            lines.append(f"- **Status**: {record.get('status', '?')}")
            lines.append(f"- **Trainer**: {record.get('trainer_type', '?')} / {record.get('backend', '?')}")
            lines.append(f"- **Model**: {record.get('model_base', '?')}")
            verdicts = exp.get("verdicts", []) if isinstance(exp, dict) else []
            if verdicts:
                latest = verdicts[-1]
                lines.append(f"- **Latest verdict**: {latest.get('verdict', '?')}")
                if latest.get("reasoning"):
                    lines.append(f"- **Reasoning**: {latest.get('reasoning')}")
            metrics = record.get("metrics_json", {})
            if isinstance(metrics, dict) and metrics:
                lines.append("- **Metrics**:")
                for k, v in metrics.items():
                    lines.append(f"  - {k}: {v}")
            ablations = exp.get("ablations", []) if isinstance(exp, dict) else []
            if ablations:
                lines.append("- **Ablations**:")
                for abl in ablations:
                    metric_parts = abl.get("metrics_json", {})
                    metric_text = ""
                    if isinstance(metric_parts, dict) and metric_parts:
                        metric_text = " | " + ", ".join(f"{k}: {v}" for k, v in metric_parts.items())
                    lines.append(
                        f"  - {abl.get('variable', '?')}={abl.get('value', '?')}{metric_text}"
                    )
            tasks = exp.get("tasks", []) if isinstance(exp, dict) else []
            if tasks:
                lines.append("- **Tasks**:")
                for task in tasks:
                    lines.append(
                        f"  - [{task.get('status', '?')}] {task.get('kind', '?')}: {task.get('title', '?')}"
                    )
            artifacts = exp.get("artifacts", []) if isinstance(exp, dict) else []
            if artifacts:
                lines.append("- **Artifacts**:")
                for artifact in artifacts:
                    lines.append(
                        f"  - {artifact.get('kind', '?')}: {artifact.get('path', '?')}"
                    )
            lines.append("")
        ext = ".md"

    content = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"report{ext}"
    output_file.write_text(content)
    print(f"[report] Basic report written to {output_file}")
    print("\n" + content)
