"""Auto-generate technical reports from experiment results."""

import json
import statistics
from pathlib import Path
from typing import Any


class ReportGenerator:
    """Generates Markdown/LaTeX technical reports from result DB data.

    Report structure:
    1. Method description (from recipe)
    2. Experimental setup (model, data, hyperparams)
    3. Results table (main + ablation)
    4. Analysis (vs baseline, failure attribution)
    5. Conclusions and next steps
    """

    def __init__(self, result_db: Any):
        self.result_db = result_db

    def _fetch_experiment_data(self, experiment_id: str) -> dict[str, Any]:
        """Fetch experiment, its ablations, and verdicts from the DB."""
        if hasattr(self.result_db, "get_experiment_bundle"):
            return self.result_db.get_experiment_bundle(experiment_id)

        exp = self.result_db.get_experiment(experiment_id)
        if exp is None:
            return {
                "experiment": None,
                "eval_runs": [],
                "ablations": [],
                "verdicts": [],
                "tasks": [],
                "artifacts": [],
            }

        conn = self.result_db._conn
        eval_runs = []
        ablations = []
        if conn is not None:
            cur = conn.execute(
                "SELECT * FROM eval_runs WHERE experiment_id = ? ORDER BY benchmark, seed, id",
                (experiment_id,),
            )
            eval_runs = [self.result_db._row_to_dict(r) for r in cur.fetchall()]
            cur = conn.execute(
                "SELECT * FROM ablations WHERE experiment_id = ? ORDER BY timestamp",
                (experiment_id,),
            )
            ablations = [self.result_db._row_to_dict(r) for r in cur.fetchall()]

        verdicts = []
        if conn is not None:
            cur = conn.execute(
                "SELECT * FROM verdicts WHERE experiment_id = ? ORDER BY timestamp",
                (experiment_id,),
            )
            verdicts = [self.result_db._row_to_dict(r) for r in cur.fetchall()]

        tasks = []
        artifacts = []
        if conn is not None:
            cur = conn.execute(
                "SELECT * FROM tasks WHERE experiment_id = ? ORDER BY updated_at DESC, id DESC",
                (experiment_id,),
            )
            tasks = [self.result_db._row_to_dict(r) for r in cur.fetchall()]
            cur = conn.execute(
                "SELECT * FROM artifacts WHERE experiment_id = ? ORDER BY timestamp, id",
                (experiment_id,),
            )
            artifacts = [self.result_db._row_to_dict(r) for r in cur.fetchall()]

        return {
            "experiment": exp,
            "eval_runs": eval_runs,
            "ablations": ablations,
            "verdicts": verdicts,
            "tasks": tasks,
            "artifacts": artifacts,
        }

    def _collect_results_rows(
        self, data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Build a flat list of result rows from the main experiment metrics.

        Each row has keys: benchmark, metric, value, seed.
        """
        rows: list[dict[str, Any]] = []
        exp = data["experiment"]
        if exp is None:
            return rows

        eval_runs = data.get("eval_runs", [])
        if eval_runs:
            for eval_run in eval_runs:
                metrics = eval_run.get("metrics_json", {})
                if isinstance(metrics, str):
                    try:
                        metrics = json.loads(metrics)
                    except json.JSONDecodeError:
                        metrics = {}
                if not isinstance(metrics, dict):
                    continue
                for metric_name, value in sorted(metrics.items()):
                    rows.append(
                        {
                            "benchmark": eval_run.get("benchmark", "main"),
                            "metric": metric_name,
                            "value": value,
                            "seed": eval_run.get("seed", "-"),
                        }
                    )
            return rows

        # Main experiment metrics
        metrics = exp.get("metrics_json")
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        if isinstance(metrics, dict):
            for metric_name, value in sorted(metrics.items()):
                rows.append(
                    {
                        "benchmark": "main",
                        "metric": metric_name,
                        "value": value,
                        "seed": "-",
                    }
                )

        return rows

    def _collect_task_rows(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Return tracked tasks for display."""
        rows: list[dict[str, Any]] = []
        for task in data.get("tasks", []):
            rows.append(
                {
                    "id": task.get("id", "?"),
                    "status": task.get("status", "?"),
                    "priority": task.get("priority", "?"),
                    "kind": task.get("kind", "?"),
                    "title": task.get("title", "?"),
                }
            )
        return rows

    def _collect_ablation_rows(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten ablation records into display rows."""
        rows: list[dict[str, Any]] = []
        for abl in data.get("ablations", []):
            abl_metrics = abl.get("metrics_json")
            if isinstance(abl_metrics, str):
                try:
                    abl_metrics = json.loads(abl_metrics)
                except json.JSONDecodeError:
                    abl_metrics = {}
            if isinstance(abl_metrics, dict):
                metric_text = ", ".join(
                    f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
                    for name, value in sorted(abl_metrics.items())
                )
            else:
                metric_text = "-"
            rows.append(
                {
                    "variable": abl.get("variable", "?"),
                    "value": abl.get("value", "?"),
                    "metrics": metric_text,
                }
            )
        return rows

    def _collect_verdict_rows(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten verdict records into display rows."""
        rows: list[dict[str, Any]] = []
        for verdict in data.get("verdicts", []):
            checks = verdict.get("checks_json", {})
            if isinstance(checks, dict):
                checks_text = ", ".join(
                    f"{name}={'yes' if ok else 'no'}" for name, ok in sorted(checks.items())
                )
            else:
                checks_text = "-"
            suggestions = verdict.get("suggestions_json", [])
            if isinstance(suggestions, list):
                suggestions_text = "; ".join(str(item) for item in suggestions) or "-"
            else:
                suggestions_text = str(suggestions) if suggestions else "-"
            rows.append(
                {
                    "verdict": verdict.get("verdict", "?"),
                    "reasoning": verdict.get("reasoning", ""),
                    "checks": checks_text,
                    "suggestions": suggestions_text,
                    "timestamp": verdict.get("timestamp", ""),
                }
            )
        return rows

    def _analyze_metrics(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute best, worst, and variance info across result rows."""
        if not rows:
            return {"best": None, "worst": None, "variance": {}}

        numeric = [(r["metric"], r["value"]) for r in rows if isinstance(r["value"], (int, float))]
        if not numeric:
            return {"best": None, "worst": None, "variance": {}}

        best = max(numeric, key=lambda x: x[1])
        worst = min(numeric, key=lambda x: x[1])

        # Group values by metric name for variance
        by_metric: dict[str, list[float]] = {}
        for name, val in numeric:
            by_metric.setdefault(name, []).append(val)

        variance: dict[str, float] = {}
        for name, vals in by_metric.items():
            if len(vals) >= 2:
                variance[name] = statistics.variance(vals)

        return {
            "best": {"metric": best[0], "value": best[1]},
            "worst": {"metric": worst[0], "value": worst[1]},
            "variance": variance,
        }

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def generate_markdown(self, experiment_ids: list[str], output_path: str | Path) -> str:
        """Generate a Markdown report for the given experiments."""
        parts: list[str] = []
        bundles = [self._fetch_experiment_data(exp_id) for exp_id in experiment_ids]

        if len(experiment_ids) > 1:
            parts.append("## Comparison\n")
            parts.append(self.generate_comparison_table(experiment_ids))
            parts.append("")

        for exp_id, data in zip(experiment_ids, bundles):
            exp = data["experiment"]
            if exp is None:
                parts.append(f"## Experiment {exp_id}\n\n_Not found._\n")
                continue

            recipe = exp.get("recipe_id", exp_id)
            parts.append(f"# Experiment Report: {recipe}\n")

            latest_verdict = data.get("verdicts", [])[-1] if data.get("verdicts") else None

            # Setup section
            parts.append("## Setup\n")
            parts.append("| Parameter | Value |")
            parts.append("| --- | --- |")
            parts.append(f"| Experiment ID | {exp_id} |")
            parts.append(f"| Model | {exp.get('model_base', 'N/A')} |")
            parts.append(f"| Trainer | {exp.get('trainer_type', 'N/A')} |")
            parts.append(f"| Backend | {exp.get('backend', 'N/A')} |")
            parts.append(f"| Config Hash | {exp.get('config_hash', 'N/A')} |")
            parts.append(f"| Status | {exp.get('status', 'unknown')} |")
            if latest_verdict:
                parts.append(f"| Latest Verdict | {latest_verdict.get('verdict', 'N/A')} |")

            metrics = exp.get("metrics_json")
            if isinstance(metrics, str):
                try:
                    metrics = json.loads(metrics)
                except json.JSONDecodeError:
                    metrics = {}
            if isinstance(metrics, dict) and metrics:
                hp_str = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
                parts.append(f"| Main Metrics | {hp_str} |")
            parts.append("")

            # Results table
            rows = self._collect_results_rows(data)
            if rows:
                parts.append("## Results\n")
                parts.append("| Benchmark | Metric | Value | Seed |")
                parts.append("| --- | --- | --- | --- |")
                for r in rows:
                    val = r["value"]
                    if isinstance(val, float):
                        val = f"{val:.4f}"
                    parts.append(f"| {r['benchmark']} | {r['metric']} | {val} | {r['seed']} |")
                parts.append("")

            # Ablation table
            ablation_rows = self._collect_ablation_rows(data)
            if ablation_rows:
                parts.append("## Ablations\n")
                parts.append("| Variable | Value | Metrics |")
                parts.append("| --- | --- | --- |")
                for row in ablation_rows:
                    parts.append(
                        f"| {row['variable']} | {row['value']} | {row['metrics']} |"
                    )
                parts.append("")

            # Verdict table
            verdict_rows = self._collect_verdict_rows(data)
            if verdict_rows:
                parts.append("## Verdicts\n")
                parts.append("| Verdict | Reasoning | Checks | Suggestions | Timestamp |")
                parts.append("| --- | --- | --- | --- | --- |")
                for row in verdict_rows:
                    parts.append(
                        f"| {row['verdict']} | {row['reasoning']} | {row['checks']} | "
                        f"{row['suggestions']} | {row['timestamp']} |"
                        )
                parts.append("")

            task_rows = self._collect_task_rows(data)
            if task_rows:
                parts.append("## Tasks\n")
                parts.append("| ID | Status | Priority | Kind | Title |")
                parts.append("| --- | --- | --- | --- | --- |")
                for row in task_rows:
                    parts.append(
                        f"| {row['id']} | {row['status']} | {row['priority']} | {row['kind']} | {row['title']} |"
                    )
                parts.append("")

            # Analysis
            analysis = self._analyze_metrics(rows)
            parts.append("## Analysis\n")
            if analysis["best"]:
                parts.append(
                    f"- **Best metric**: {analysis['best']['metric']} = "
                    f"{analysis['best']['value']:.4f}"
                )
            if analysis["worst"]:
                parts.append(
                    f"- **Worst metric**: {analysis['worst']['metric']} = "
                    f"{analysis['worst']['value']:.4f}"
                )
            if analysis["variance"]:
                var_lines = ", ".join(
                    f"{k}: {v:.6f}" for k, v in sorted(analysis["variance"].items())
                )
                parts.append(f"- **Variance across seeds**: {var_lines}")
            if not analysis["best"] and not analysis["worst"]:
                parts.append("_No numeric metrics available for analysis._")
            parts.append("")

            # Status / errors
            parts.append("## Status\n")
            parts.append(f"- **Status**: {exp.get('status', 'unknown')}")
            if exp.get("error"):
                parts.append(f"- **Error**: {exp['error']}")
            parts.append("")

        report = "\n".join(parts)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

        return report

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------

    def generate_latex(self, experiment_ids: list[str], output_path: str | Path) -> str:
        """Generate a LaTeX report (compatible with ARIS paper-writing workflow)."""
        parts: list[str] = []
        bundles = [self._fetch_experiment_data(exp_id) for exp_id in experiment_ids]

        parts.append(r"\documentclass{article}")
        parts.append(r"\usepackage{booktabs}")
        parts.append(r"\usepackage{geometry}")
        parts.append(r"\geometry{margin=1in}")
        parts.append(r"\begin{document}")
        parts.append("")

        if len(experiment_ids) > 1:
            parts.append(r"\section{Comparison}")
            parts.append(r"\begin{itemize}")
            for exp_id, data in zip(experiment_ids, bundles):
                exp = data["experiment"]
                if exp is None:
                    parts.append(rf"\item {_latex_escape(exp_id)}: experiment not found")
                    continue
                latest_verdict = data.get("verdicts", [])[-1] if data.get("verdicts") else None
                verdict_text = latest_verdict.get("verdict", "N/A") if latest_verdict else "N/A"
                parts.append(
                    rf"\item {_latex_escape(exp_id)}: status "
                    rf"{_latex_escape(exp.get('status', 'unknown'))}, verdict {_latex_escape(verdict_text)}"
                )
            parts.append(r"\end{itemize}")
            parts.append("")

        for exp_id, data in zip(experiment_ids, bundles):
            exp = data["experiment"]
            if exp is None:
                parts.append(rf"\section{{Experiment {_latex_escape(exp_id)}}}")
                parts.append("Experiment not found.")
                parts.append("")
                continue

            recipe = exp.get("recipe_id", exp_id)
            parts.append(rf"\section{{Experiment Report: {_latex_escape(recipe)}}}")
            parts.append("")

            latest_verdict = data.get("verdicts", [])[-1] if data.get("verdicts") else None

            # Setup
            parts.append(r"\subsection{Setup}")
            parts.append(r"\begin{tabular}{ll}")
            parts.append(r"\toprule")
            parts.append(r"Parameter & Value \\")
            parts.append(r"\midrule")
            parts.append(rf"Experiment ID & {_latex_escape(exp_id)} \\")
            parts.append(rf"Model & {_latex_escape(exp.get('model_base', 'N/A'))} \\")
            parts.append(rf"Trainer & {_latex_escape(exp.get('trainer_type', 'N/A'))} \\")
            parts.append(rf"Backend & {_latex_escape(exp.get('backend', 'N/A'))} \\")
            parts.append(rf"Config Hash & {_latex_escape(exp.get('config_hash', 'N/A'))} \\")
            parts.append(rf"Status & {_latex_escape(exp.get('status', 'unknown'))} \\")
            if latest_verdict:
                parts.append(rf"Latest Verdict & {_latex_escape(latest_verdict.get('verdict', 'N/A'))} \\")

            metrics = exp.get("metrics_json")
            if isinstance(metrics, str):
                try:
                    metrics = json.loads(metrics)
                except json.JSONDecodeError:
                    metrics = {}
            if isinstance(metrics, dict) and metrics:
                hp_str = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
                parts.append(rf"Main Metrics & {_latex_escape(hp_str)} \\")

            parts.append(r"\bottomrule")
            parts.append(r"\end{tabular}")
            parts.append("")

            # Results table
            rows = self._collect_results_rows(data)
            if rows:
                parts.append(r"\subsection{Results}")
                parts.append(r"\begin{tabular}{llrl}")
                parts.append(r"\toprule")
                parts.append(r"Benchmark & Metric & Value & Seed \\")
                parts.append(r"\midrule")
                for r in rows:
                    val = r["value"]
                    if isinstance(val, float):
                        val = f"{val:.4f}"
                    parts.append(
                        rf"{_latex_escape(str(r['benchmark']))} & "
                        rf"{_latex_escape(str(r['metric']))} & "
                        rf"{val} & {r['seed']} \\"
                    )
                parts.append(r"\bottomrule")
                parts.append(r"\end{tabular}")
                parts.append("")

            # Ablations
            ablation_rows = self._collect_ablation_rows(data)
            if ablation_rows:
                parts.append(r"\subsection{Ablations}")
                parts.append(r"\begin{tabular}{lll}")
                parts.append(r"\toprule")
                parts.append(r"Variable & Value & Metrics \\")
                parts.append(r"\midrule")
                for row in ablation_rows:
                    parts.append(
                        rf"{_latex_escape(str(row['variable']))} & "
                        rf"{_latex_escape(str(row['value']))} & "
                        rf"{_latex_escape(str(row['metrics']))} \\"
                    )
                parts.append(r"\bottomrule")
                parts.append(r"\end{tabular}")
                parts.append("")

            # Verdicts
            verdict_rows = self._collect_verdict_rows(data)
            if verdict_rows:
                parts.append(r"\subsection{Verdicts}")
                parts.append(r"\begin{tabular}{lllll}")
                parts.append(r"\toprule")
                parts.append(r"Verdict & Reasoning & Checks & Suggestions & Timestamp \\")
                parts.append(r"\midrule")
                for row in verdict_rows:
                    parts.append(
                        rf"{_latex_escape(str(row['verdict']))} & "
                        rf"{_latex_escape(str(row['reasoning']))} & "
                        rf"{_latex_escape(str(row['checks']))} & "
                        rf"{_latex_escape(str(row['suggestions']))} & "
                        rf"{_latex_escape(str(row['timestamp']))} \\"
                    )
                parts.append(r"\bottomrule")
                parts.append(r"\end{tabular}")
                parts.append("")

            task_rows = self._collect_task_rows(data)
            if task_rows:
                parts.append(r"\subsection{Tasks}")
                parts.append(r"\begin{tabular}{lllll}")
                parts.append(r"\toprule")
                parts.append(r"ID & Status & Priority & Kind & Title \\")
                parts.append(r"\midrule")
                for row in task_rows:
                    parts.append(
                        rf"{_latex_escape(str(row['id']))} & "
                        rf"{_latex_escape(str(row['status']))} & "
                        rf"{_latex_escape(str(row['priority']))} & "
                        rf"{_latex_escape(str(row['kind']))} & "
                        rf"{_latex_escape(str(row['title']))} \\"
                    )
                parts.append(r"\bottomrule")
                parts.append(r"\end{tabular}")
                parts.append("")

            # Analysis
            analysis = self._analyze_metrics(rows)
            parts.append(r"\subsection{Analysis}")
            parts.append(r"\begin{itemize}")
            if analysis["best"]:
                parts.append(
                    rf"\item \textbf{{Best metric}}: {_latex_escape(analysis['best']['metric'])} "
                    rf"= {analysis['best']['value']:.4f}"
                )
            if analysis["worst"]:
                parts.append(
                    rf"\item \textbf{{Worst metric}}: {_latex_escape(analysis['worst']['metric'])} "
                    rf"= {analysis['worst']['value']:.4f}"
                )
            if analysis["variance"]:
                var_lines = ", ".join(
                    f"{_latex_escape(k)}: {v:.6f}"
                    for k, v in sorted(analysis["variance"].items())
                )
                parts.append(rf"\item \textbf{{Variance across seeds}}: {var_lines}")
            if not analysis["best"] and not analysis["worst"]:
                parts.append(r"\item No numeric metrics available for analysis.")
            parts.append(r"\end{itemize}")
            parts.append("")

            # Status
            parts.append(r"\subsection{Status}")
            parts.append(rf"Status: {_latex_escape(exp.get('status', 'unknown'))}")
            if exp.get("error"):
                parts.append("")
                parts.append(rf"Error: {_latex_escape(exp['error'])}")
            parts.append("")

        parts.append(r"\end{document}")
        report = "\n".join(parts)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

        return report

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def generate_comparison_table(self, recipe_ids: list[str]) -> str:
        """Generate a comparison table across multiple recipes or experiments.

        Produces a Markdown table with experiment ID, recipe, and key
        metrics side by side, with the best value per metric bolded.
        """
        # Gather experiments for each identifier. Accept either experiment IDs
        # or recipe IDs so callers do not need to pre-normalize the input.
        experiments: list[dict[str, Any]] = []
        for rid in recipe_ids:
            exp = self.result_db.get_experiment(rid)
            if exp is not None:
                experiments.append(exp)
                continue
            exps = self.result_db.find_by_recipe(rid)
            experiments.extend(exps)

        if not experiments:
            return "_No experiments found for the given recipes._\n"

        # Collect the union of all metric keys
        all_metrics: set[str] = set()
        for exp in experiments:
            m = exp.get("metrics_json")
            if isinstance(m, str):
                m = json.loads(m)
            if isinstance(m, dict):
                all_metrics.update(m.keys())
        metric_names = sorted(all_metrics)

        if not metric_names:
            return "_No metrics available for comparison._\n"

        # Build header
        header_cols = ["Experiment ID", "Recipe", "Status", "Verdict"] + metric_names
        header = "| " + " | ".join(header_cols) + " |"
        sep = "| " + " | ".join("---" for _ in header_cols) + " |"

        # Find best value per metric (highest)
        best: dict[str, float] = {}
        for exp in experiments:
            m = exp.get("metrics_json")
            if isinstance(m, str):
                m = json.loads(m)
            if isinstance(m, dict):
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        if k not in best or v > best[k]:
                            best[k] = v

        # Build rows
        rows: list[str] = []
        for exp in experiments:
            m = exp.get("metrics_json")
            if isinstance(m, str):
                m = json.loads(m)
            if not isinstance(m, dict):
                m = {}

            verdict = "-"
            latest_verdict = self.result_db.get_latest_verdict(exp.get("id", "")) if exp.get("id") else None
            if latest_verdict is not None:
                verdict = latest_verdict.get("verdict", "-")

            cols = [
                exp.get("id", "?"),
                exp.get("recipe_id", "?"),
                exp.get("status", "?"),
                verdict,
            ]
            for mn in metric_names:
                val = m.get(mn)
                if val is None:
                    cols.append("-")
                elif isinstance(val, float):
                    cell = f"{val:.4f}"
                    if mn in best and val == best[mn]:
                        cell = f"**{cell}**"
                    cols.append(cell)
                else:
                    cell = str(val)
                    if mn in best and isinstance(val, (int, float)) and val == best[mn]:
                        cell = f"**{cell}**"
                    cols.append(cell)

            rows.append("| " + " | ".join(cols) + " |")

        return "\n".join([header, sep] + rows) + "\n"


    # ------------------------------------------------------------------
    # Blog-style report (LoRA Insights format)
    # ------------------------------------------------------------------

    def generate_blog_report(
        self,
        experiment_ids: list[str],
        output_path: str | Path,
        *,
        title: str | None = None,
    ) -> str:
        """Generate a blog-style experiment report inspired by Lightning AI's LoRA Insights.

        Structure (following Sebastian Raschka's LoRA Insights blog):
          1. Title + TL;DR (key takeaways)
          2. Introduction & Motivation
          3. Experimental Setup (model, dataset, hardware, benchmarks)
          4. Experiments — each as a sequential diary entry with:
             - Question / hypothesis
             - Setup description
             - Results table + interpretation
             - Key finding
          5. Ablation Studies
          6. Cost & Efficiency Analysis
          7. Practical Recommendations
          8. Reproducibility
          9. Conclusion
        """
        bundles = [self._fetch_experiment_data(exp_id) for exp_id in experiment_ids]

        # Gather all data
        all_recipes: list[dict] = []
        all_results: list[dict] = []
        all_ablations: list[dict] = []
        all_verdicts: list[dict] = []

        for data in bundles:
            exp = data["experiment"]
            if exp is None:
                continue
            recipe_json = exp.get("recipe_json")
            if isinstance(recipe_json, str):
                try:
                    recipe_json = json.loads(recipe_json)
                except json.JSONDecodeError:
                    recipe_json = {}
            all_recipes.append(recipe_json or {})
            all_results.append(data)
            all_ablations.extend(data.get("ablations", []))
            all_verdicts.extend(data.get("verdicts", []))

        parts: list[str] = []

        # === 1. Title + TL;DR ===
        report_title = title or self._infer_title(bundles)
        parts.append(f"# {report_title}\n")
        parts.append(self._generate_tldr(bundles, all_verdicts))

        # === 2. Introduction & Motivation ===
        parts.append("## Introduction\n")
        parts.append(self._generate_introduction(bundles, all_recipes))

        # === 3. Experimental Setup ===
        parts.append("## Experimental Setup\n")
        parts.append(self._generate_setup_section(bundles, all_recipes))

        # === 4. Experiments (sequential diary) ===
        parts.append("## Experiments\n")
        parts.append(self._generate_experiment_diary(bundles))

        # === 5. Ablation Studies ===
        if all_ablations:
            parts.append("## Ablation Studies\n")
            parts.append(self._generate_ablation_section(bundles, all_ablations))

        # === 6. Cost & Efficiency ===
        parts.append("## Cost & Efficiency Analysis\n")
        parts.append(self._generate_cost_section(bundles, all_recipes))

        # === 7. Practical Recommendations ===
        parts.append("## Practical Recommendations\n")
        parts.append(self._generate_recommendations(bundles, all_verdicts, all_ablations))

        # === 8. Reproducibility ===
        parts.append("## Reproducibility\n")
        parts.append(self._generate_reproducibility(bundles, all_recipes))

        # === 9. Conclusion ===
        parts.append("## Conclusion\n")
        parts.append(self._generate_conclusion(bundles, all_verdicts))

        report = "\n".join(parts)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

        return report

    # ------------------------------------------------------------------
    # Blog report helper methods
    # ------------------------------------------------------------------

    def _infer_title(self, bundles: list[dict]) -> str:
        """Infer a descriptive title from experiment data."""
        exp = bundles[0]["experiment"] if bundles else None
        if exp is None:
            return "Experiment Report"
        trainer_type = exp.get("trainer_type", "Training")
        model = exp.get("model_base", "LLM")
        backend = exp.get("backend", "")
        type_labels = {
            "sft": "Supervised Fine-Tuning",
            "rl": "Reinforcement Learning",
            "grpo": "GRPO Training",
            "distill": "Trajectory Distillation",
            "dpo": "Direct Preference Optimization",
        }
        method = type_labels.get(trainer_type, trainer_type.upper())
        model_short = model.split("/")[-1] if "/" in model else model
        n = len(bundles)
        suffix = f": Insights from {n} Experiment{'s' if n > 1 else ''}" if n > 1 else ""
        return f"{method} on {model_short}{suffix}"

    def _generate_tldr(self, bundles: list[dict], verdicts: list[dict]) -> str:
        """Generate a TL;DR / Key Takeaways section."""
        lines = ["> **TL;DR** — Key takeaways from this experiment series:\n"]

        # Summarize verdict outcomes
        verdict_counts: dict[str, int] = {}
        for v in verdicts:
            vv = v.get("verdict", "unknown")
            verdict_counts[vv] = verdict_counts.get(vv, 0) + 1

        # Find best metrics across all experiments
        best_metrics: dict[str, tuple[float, str]] = {}  # metric -> (value, exp_id)
        for data in bundles:
            exp = data["experiment"]
            if exp is None:
                continue
            rows = self._collect_results_rows(data)
            for r in rows:
                if isinstance(r["value"], (int, float)):
                    key = r["metric"]
                    if key not in best_metrics or r["value"] > best_metrics[key][0]:
                        best_metrics[key] = (r["value"], exp.get("id", "?"))

        if best_metrics:
            top_metrics = sorted(best_metrics.items(), key=lambda x: x[1][0], reverse=True)[:3]
            for metric, (value, _) in top_metrics:
                lines.append(f"> - **{metric}**: best result = {value:.4f}")

        accepted = verdict_counts.get("accept", 0)
        total = len(bundles)
        if total > 0:
            lines.append(f"> - {accepted}/{total} experiment(s) accepted by the judge")

        # Seed consistency
        seed_variances = self._compute_seed_variances(bundles)
        if seed_variances:
            avg_cv = sum(seed_variances.values()) / len(seed_variances)
            stability = "highly stable" if avg_cv < 0.02 else "stable" if avg_cv < 0.05 else "moderate" if avg_cv < 0.1 else "noisy"
            lines.append(f"> - Results are **{stability}** across seeds (avg CV = {avg_cv:.3f})")

        lines.append("")
        return "\n".join(lines)

    def _generate_introduction(self, bundles: list[dict], recipes: list[dict]) -> str:
        """Generate introduction with motivation and context."""
        lines = []

        if not bundles or bundles[0]["experiment"] is None:
            return "No experiment data available.\n"

        exp = bundles[0]["experiment"]
        trainer = exp.get("trainer_type", "unknown")
        model = exp.get("model_base", "unknown")
        n = len(bundles)

        type_descriptions = {
            "sft": "supervised fine-tuning (SFT) on curated trajectories",
            "rl": "reinforcement learning with environment feedback",
            "grpo": "Group Relative Policy Optimization (GRPO)",
            "distill": "trajectory distillation from a stronger teacher model",
            "dpo": "Direct Preference Optimization on paired trajectories",
        }
        method_desc = type_descriptions.get(trainer, f"{trainer}-based training")

        lines.append(
            f"This report documents {n} experiment{'s' if n > 1 else ''} "
            f"using {method_desc} to train coding agents based on "
            f"`{model}`. The experiments follow a structured recipe-driven "
            f"pipeline and are automatically evaluated and judged.\n"
        )

        # Source papers
        papers = set()
        for recipe in recipes:
            for p in recipe.get("source_papers", []):
                papers.add(str(p))
        if papers:
            lines.append(f"**Source papers**: {', '.join(sorted(papers))}\n")

        lines.append(
            "Each experiment is presented as a sequential diary entry, "
            "following the approach used in Raschka's "
            "[LoRA Insights](https://lightning.ai/blog/lora-insights) — "
            "we describe the question, setup, results, and key finding "
            "for each run before drawing overall conclusions.\n"
        )
        return "\n".join(lines)

    def _generate_setup_section(self, bundles: list[dict], recipes: list[dict]) -> str:
        """Generate the experimental setup section with model, data, and hardware info."""
        lines = []

        # Model table
        lines.append("### Model\n")
        lines.append("| Parameter | Value |")
        lines.append("| --- | --- |")
        models_seen = set()
        for data in bundles:
            exp = data["experiment"]
            if exp is None:
                continue
            model = exp.get("model_base", "N/A")
            if model in models_seen:
                continue
            models_seen.add(model)
            lines.append(f"| Base model | `{model}` |")

        for recipe in recipes:
            model_cfg = recipe.get("model", {})
            adapter = model_cfg.get("adapter", "N/A")
            lines.append(f"| Adapter | {adapter} |")
            if model_cfg.get("size"):
                lines.append(f"| Size | {model_cfg['size']} |")
            break
        lines.append("")

        # Dataset table
        lines.append("### Dataset\n")
        all_sources: list[dict] = []
        all_filters: list[dict] = []
        for recipe in recipes:
            ds = recipe.get("dataset", {})
            all_sources.extend(ds.get("sources", []))
            all_filters.extend(ds.get("filters", []))

        if all_sources:
            lines.append("| Dataset | Path | Weight |")
            lines.append("| --- | --- | --- |")
            seen = set()
            for src in all_sources:
                key = src.get("name", "")
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"| {src.get('name', '?')} | `{src.get('path', '?')}` | {src.get('mix_weight', 1.0)} |"
                )
            lines.append("")

        if all_filters:
            lines.append("**Data filters**: " + ", ".join(
                f"{f.get('type', '?')}({', '.join(f'{k}={v}' for k, v in f.get('params', {}).items())})"
                for f in all_filters
            ) + "\n")

        # Training config
        lines.append("### Training Configuration\n")
        lines.append("| Parameter | Value |")
        lines.append("| --- | --- |")
        for data in bundles[:1]:
            exp = data["experiment"]
            if exp is None:
                continue
            lines.append(f"| Trainer | {exp.get('trainer_type', '?')} |")
            lines.append(f"| Backend | {exp.get('backend', '?')} |")

        for recipe in recipes[:1]:
            trainer_cfg = recipe.get("trainer", {})
            params = trainer_cfg.get("params", {})
            for k, v in sorted(params.items()):
                lines.append(f"| {k} | {v} |")
            # Reward config
            reward = trainer_cfg.get("reward", {})
            if reward:
                lines.append(f"| Reward type | {reward.get('type', '?')} |")
                for comp in reward.get("components", []):
                    lines.append(
                        f"| Reward component | {comp.get('type', '?')} (weight={comp.get('weight', '?')}) |"
                    )
            break
        lines.append("")

        # Evaluation setup
        lines.append("### Evaluation\n")
        for recipe in recipes[:1]:
            eval_cfg = recipe.get("eval", recipe.get("evaluation", {}))
            benchmarks = eval_cfg.get("benchmarks", [])
            seeds = eval_cfg.get("seeds", [])
            metrics = eval_cfg.get("metrics", [])
            lines.append(f"- **Benchmarks**: {', '.join(benchmarks) or 'N/A'}")
            lines.append(f"- **Metrics**: {', '.join(metrics) or 'N/A'}")
            lines.append(f"- **Seeds**: {', '.join(str(s) for s in seeds) or 'N/A'}")
            lines.append(
                f"- Each experiment is evaluated {len(seeds)} time(s) with different "
                f"random seeds to assess reproducibility."
            )
            break
        lines.append("")

        # Hardware / budget
        lines.append("### Hardware & Budget\n")
        for recipe in recipes[:1]:
            budget = recipe.get("budget", {})
            if budget:
                lines.append(f"- **GPU**: {budget.get('gpu_type', 'N/A')}")
                lines.append(f"- **Max GPU hours**: {budget.get('max_gpu_hours', 'N/A')}")
                if budget.get("max_cost_usd"):
                    lines.append(f"- **Max cost**: ${budget['max_cost_usd']}")
            else:
                lines.append("- Budget not specified")
            break
        lines.append("")

        return "\n".join(lines)

    def _generate_experiment_diary(self, bundles: list[dict]) -> str:
        """Generate sequential experiment entries, each as a diary section."""
        lines = []

        for i, data in enumerate(bundles, 1):
            exp = data["experiment"]
            if exp is None:
                continue

            exp_id = exp.get("id", "?")
            recipe_id = exp.get("recipe_id", "?")
            status = exp.get("status", "unknown")
            trainer = exp.get("trainer_type", "?")
            backend = exp.get("backend", "?")

            # Section header
            lines.append(f"### Experiment {i}: {recipe_id}\n")

            # Question / hypothesis
            lines.append(f"**Question**: How does {trainer}/{backend} perform on this recipe configuration?\n")

            # Setup summary
            lines.append(f"**Setup**: `{exp.get('model_base', '?')}` · {trainer} · {backend} · status={status}\n")

            # Config hash for reproducibility
            config_hash = exp.get("config_hash", "")
            if config_hash:
                lines.append(f"**Config hash**: `{config_hash}`\n")

            # Results table
            rows = self._collect_results_rows(data)
            if rows:
                lines.append("**Results**:\n")
                lines.append("| Benchmark | Metric | Value | Seed |")
                lines.append("| --- | --- | --- | --- |")
                for r in rows:
                    val = r["value"]
                    if isinstance(val, float):
                        val = f"{val:.4f}"
                    lines.append(f"| {r['benchmark']} | {r['metric']} | {val} | {r['seed']} |")
                lines.append("")

                # Aggregate stats
                analysis = self._analyze_metrics(rows)
                if analysis["best"]:
                    lines.append(
                        f"**Best**: {analysis['best']['metric']} = {analysis['best']['value']:.4f}"
                    )
                if analysis["variance"]:
                    low_var = all(v < 0.001 for v in analysis["variance"].values())
                    if low_var:
                        lines.append(
                            "**Seed stability**: Excellent — variance across seeds is negligible."
                        )
                    else:
                        var_text = ", ".join(
                            f"{k}: {v:.6f}" for k, v in sorted(analysis["variance"].items())
                        )
                        lines.append(f"**Seed variance**: {var_text}")
                lines.append("")
            else:
                # No eval results — check for training metrics
                metrics = exp.get("metrics_json")
                if isinstance(metrics, str):
                    try:
                        metrics = json.loads(metrics)
                    except json.JSONDecodeError:
                        metrics = {}
                if isinstance(metrics, dict) and metrics:
                    lines.append("**Training metrics**:\n")
                    lines.append("| Metric | Value |")
                    lines.append("| --- | --- |")
                    for k, v in sorted(metrics.items()):
                        if isinstance(v, float):
                            lines.append(f"| {k} | {v:.4f} |")
                        else:
                            lines.append(f"| {k} | {v} |")
                    lines.append("")

            # Verdict
            verdicts = data.get("verdicts", [])
            if verdicts:
                latest = verdicts[-1]
                verdict_val = latest.get("verdict", "?")
                reasoning = latest.get("reasoning", "")
                emoji_map = {"accept": "PASS", "reject": "FAIL", "needs_rerun": "RERUN", "needs_ablation": "ABLATION"}
                label = emoji_map.get(verdict_val, verdict_val.upper())
                lines.append(f"**Judge verdict**: **{label}**")
                if reasoning:
                    lines.append(f"> {reasoning}")

                suggestions = latest.get("suggestions_json", [])
                if isinstance(suggestions, list) and suggestions:
                    lines.append("\n**Suggestions**:")
                    for s in suggestions:
                        lines.append(f"- {s}")
                lines.append("")

            # Error info
            if exp.get("error"):
                lines.append(f"**Error**: {exp['error']}\n")

            # Key finding (synthesized)
            lines.append(self._synthesize_finding(data))
            lines.append("---\n")

        return "\n".join(lines)

    def _generate_ablation_section(self, bundles: list[dict], ablations: list[dict]) -> str:
        """Generate ablation studies section with comparison tables."""
        lines = []

        lines.append(
            "Ablation studies isolate the effect of individual hyperparameters. "
            "Each row below shows one variable held at a different value while "
            "all other settings remain constant.\n"
        )

        # Group ablations by variable
        by_variable: dict[str, list[dict]] = {}
        for abl in ablations:
            var = abl.get("variable", "unknown")
            by_variable.setdefault(var, []).append(abl)

        for variable, entries in sorted(by_variable.items()):
            lines.append(f"### {variable}\n")
            lines.append("| Value | Metrics |")
            lines.append("| --- | --- |")

            best_value = None
            best_score = -float("inf")

            for entry in entries:
                abl_metrics = entry.get("metrics_json")
                if isinstance(abl_metrics, str):
                    try:
                        abl_metrics = json.loads(abl_metrics)
                    except json.JSONDecodeError:
                        abl_metrics = {}
                if isinstance(abl_metrics, dict):
                    metric_text = ", ".join(
                        f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
                        for name, value in sorted(abl_metrics.items())
                    )
                    # Track best
                    for v in abl_metrics.values():
                        if isinstance(v, (int, float)) and v > best_score:
                            best_score = v
                            best_value = entry.get("value", "?")
                else:
                    metric_text = "-"
                lines.append(f"| {entry.get('value', '?')} | {metric_text} |")

            if best_value is not None:
                lines.append(f"\n**Best setting**: `{variable}` = `{best_value}`\n")
            lines.append("")

        return "\n".join(lines)

    def _generate_cost_section(self, bundles: list[dict], recipes: list[dict]) -> str:
        """Generate cost and efficiency analysis."""
        lines = []

        budget_info: list[dict] = []
        for recipe in recipes:
            b = recipe.get("budget", {})
            if b:
                budget_info.append(b)

        if not budget_info:
            lines.append("No budget information was recorded for these experiments.\n")
            return "\n".join(lines)

        lines.append("| Parameter | Value |")
        lines.append("| --- | --- |")

        total_gpu_hours = sum(b.get("max_gpu_hours", 0) for b in budget_info)
        total_cost = sum(b.get("max_cost_usd", 0) for b in budget_info)

        for b in budget_info[:1]:
            lines.append(f"| GPU type | {b.get('gpu_type', 'N/A')} |")
        lines.append(f"| Total GPU hours (budget) | {total_gpu_hours} |")
        if total_cost:
            lines.append(f"| Total cost (budget) | ${total_cost:.2f} |")
        lines.append("")

        # Performance per GPU hour
        best_metrics: dict[str, float] = {}
        for data in bundles:
            rows = self._collect_results_rows(data)
            for r in rows:
                if isinstance(r["value"], (int, float)):
                    key = r["metric"]
                    if key not in best_metrics or r["value"] > best_metrics[key]:
                        best_metrics[key] = r["value"]

        if best_metrics and total_gpu_hours > 0:
            lines.append("**Performance per GPU-hour** (budget-normalized):\n")
            for metric, value in sorted(best_metrics.items()):
                per_hour = value / total_gpu_hours
                lines.append(f"- {metric}: {per_hour:.4f} per GPU-hour (best={value:.4f})")
            lines.append("")

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        bundles: list[dict],
        verdicts: list[dict],
        ablations: list[dict],
    ) -> str:
        """Synthesize practical recommendations from results and verdicts."""
        lines = []
        recommendations: list[str] = []

        # From verdicts
        for v in verdicts:
            suggestions = v.get("suggestions_json", [])
            if isinstance(suggestions, list):
                recommendations.extend(str(s) for s in suggestions)
            elif suggestions:
                recommendations.append(str(suggestions))

        # From ablations — recommend best settings
        by_variable: dict[str, list[tuple[Any, float]]] = {}
        for abl in ablations:
            var = abl.get("variable", "")
            abl_metrics = abl.get("metrics_json")
            if isinstance(abl_metrics, str):
                try:
                    abl_metrics = json.loads(abl_metrics)
                except json.JSONDecodeError:
                    continue
            if isinstance(abl_metrics, dict):
                for mv in abl_metrics.values():
                    if isinstance(mv, (int, float)):
                        by_variable.setdefault(var, []).append((abl.get("value"), mv))
                        break

        for var, entries in by_variable.items():
            if entries:
                best_val, best_score = max(entries, key=lambda x: x[1])
                recommendations.append(
                    f"For `{var}`, the best setting is `{best_val}` (score={best_score:.4f})."
                )

        # Seed stability advice
        seed_variances = self._compute_seed_variances(bundles)
        if seed_variances:
            high_var = {k: v for k, v in seed_variances.items() if v > 0.05}
            if high_var:
                recommendations.append(
                    "Some metrics show high variance across seeds: "
                    + ", ".join(f"{k} (CV={v:.3f})" for k, v in high_var.items())
                    + ". Consider running more seeds or investigating instability."
                )
            else:
                recommendations.append(
                    "Results are stable across seeds — 3 seeds provide sufficient confidence."
                )

        if not recommendations:
            lines.append("No specific recommendations could be derived from the current results.\n")
        else:
            seen = set()
            for i, rec in enumerate(recommendations, 1):
                if rec in seen:
                    continue
                seen.add(rec)
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def _generate_reproducibility(self, bundles: list[dict], recipes: list[dict]) -> str:
        """Generate reproducibility information."""
        lines = []

        lines.append("To reproduce these experiments:\n")
        lines.append("```bash")
        for data in bundles:
            exp = data["experiment"]
            if exp is None:
                continue
            recipe_id = exp.get("recipe_id", "?")
            lines.append(f"# Experiment: {exp.get('id', '?')}")
            lines.append(f"act train recipes/examples/{recipe_id}.recipe.json")
        lines.append("```\n")

        # Config hashes
        hashes = []
        for data in bundles:
            exp = data["experiment"]
            if exp and exp.get("config_hash"):
                hashes.append(f"- `{exp['id']}`: `{exp['config_hash']}`")
        if hashes:
            lines.append("**Config hashes** (for deduplication):\n")
            lines.extend(hashes)
            lines.append("")

        # Seeds
        all_seeds: set[int] = set()
        for recipe in recipes:
            eval_cfg = recipe.get("eval", recipe.get("evaluation", {}))
            for s in eval_cfg.get("seeds", []):
                all_seeds.add(s)
        if all_seeds:
            lines.append(f"**Seeds used**: {', '.join(str(s) for s in sorted(all_seeds))}\n")

        return "\n".join(lines)

    def _generate_conclusion(self, bundles: list[dict], verdicts: list[dict]) -> str:
        """Generate conclusion summarizing key outcomes."""
        lines = []

        if not bundles:
            return "No experiments to conclude on.\n"

        total = len(bundles)
        accepted = sum(
            1 for v in verdicts if v.get("verdict") == "accept"
        )
        rejected = sum(
            1 for v in verdicts if v.get("verdict") == "reject"
        )

        best_metrics: dict[str, tuple[float, str]] = {}
        for data in bundles:
            exp = data["experiment"]
            if exp is None:
                continue
            rows = self._collect_results_rows(data)
            for r in rows:
                if isinstance(r["value"], (int, float)):
                    key = r["metric"]
                    if key not in best_metrics or r["value"] > best_metrics[key][0]:
                        best_metrics[key] = (r["value"], exp.get("id", "?"))

        lines.append(
            f"Across {total} experiment(s), {accepted} were accepted and {rejected} rejected "
            f"by the automated judge.\n"
        )

        if best_metrics:
            lines.append("**Best results achieved**:\n")
            for metric, (value, exp_id) in sorted(best_metrics.items()):
                lines.append(f"- **{metric}**: {value:.4f} (experiment `{exp_id}`)")
            lines.append("")

        # Overall takeaway from verdicts
        if accepted == total and total > 0:
            lines.append(
                "All experiments passed the judge's quality checks, including baseline "
                "alignment, seed consistency, and ablation coverage. The results are "
                "ready for downstream deployment or further iteration.\n"
            )
        elif rejected == total and total > 0:
            lines.append(
                "All experiments were rejected. Review the judge's suggestions above "
                "and consider adjusting the recipe before re-running.\n"
            )
        elif total > 0:
            lines.append(
                "Mixed results suggest the recipe is partially effective. "
                "Focus future iterations on the configurations that passed "
                "and ablate the variables identified by the judge.\n"
            )

        return "\n".join(lines)

    def _synthesize_finding(self, data: dict) -> str:
        """Synthesize a key finding for a single experiment."""
        exp = data["experiment"]
        if exp is None:
            return "**Finding**: No data available.\n"

        rows = self._collect_results_rows(data)
        verdicts = data.get("verdicts", [])

        if not rows and not verdicts:
            status = exp.get("status", "unknown")
            return f"**Finding**: Experiment status is `{status}`. No evaluation results available yet.\n"

        # Best metric
        analysis = self._analyze_metrics(rows)
        best = analysis.get("best")

        verdict_text = ""
        if verdicts:
            latest = verdicts[-1]
            verdict_text = f" The judge's verdict is **{latest.get('verdict', '?')}**."

        if best:
            return (
                f"**Finding**: The best metric is {best['metric']} = {best['value']:.4f}."
                f"{verdict_text}\n"
            )
        return f"**Finding**: Experiment completed.{verdict_text}\n"

    def _compute_seed_variances(self, bundles: list[dict]) -> dict[str, float]:
        """Compute coefficient of variation for each metric across seeds."""
        metric_values: dict[str, list[float]] = {}
        for data in bundles:
            rows = self._collect_results_rows(data)
            for r in rows:
                if isinstance(r["value"], (int, float)):
                    metric_values.setdefault(r["metric"], []).append(float(r["value"]))

        result: dict[str, float] = {}
        for metric, values in metric_values.items():
            if len(values) >= 2:
                mean = sum(values) / len(values)
                if mean > 0:
                    std = statistics.stdev(values)
                    result[metric] = std / mean  # CV
        return result


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in *text*."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text
