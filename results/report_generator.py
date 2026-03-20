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
            return {"experiment": None, "ablations": [], "verdicts": []}

        conn = self.result_db._conn
        ablations = []
        if conn is not None:
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

        return {"experiment": exp, "ablations": ablations, "verdicts": verdicts}

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

        # Main experiment metrics
        metrics = exp.get("metrics_json")
        if isinstance(metrics, str):
            import json
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
        import json as _json

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
                m = _json.loads(m)
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
                m = _json.loads(m)
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
                m = _json.loads(m)
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
