"""Auto-generate technical reports from experiment results."""

import statistics
from typing import Any
from pathlib import Path


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
        """Build a flat list of result rows from experiment + ablation metrics.

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

        # Ablation metrics
        for abl in data.get("ablations", []):
            abl_metrics = abl.get("metrics_json")
            if isinstance(abl_metrics, str):
                import json
                abl_metrics = json.loads(abl_metrics)
            if isinstance(abl_metrics, dict):
                label = f"{abl.get('variable', '?')}={abl.get('value', '?')}"
                for metric_name, value in sorted(abl_metrics.items()):
                    rows.append(
                        {
                            "benchmark": label,
                            "metric": metric_name,
                            "value": value,
                            "seed": "-",
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

        for exp_id in experiment_ids:
            data = self._fetch_experiment_data(exp_id)
            exp = data["experiment"]
            if exp is None:
                parts.append(f"## Experiment {exp_id}\n\n_Not found._\n")
                continue

            recipe = exp.get("recipe_id", exp_id)
            parts.append(f"# Experiment Report: {recipe}\n")

            # Setup section
            parts.append("## Setup\n")
            parts.append(f"| Parameter | Value |")
            parts.append(f"| --- | --- |")
            parts.append(f"| Experiment ID | {exp_id} |")
            parts.append(f"| Model | {exp.get('model_base', 'N/A')} |")
            parts.append(f"| Trainer | {exp.get('trainer_type', 'N/A')} |")
            parts.append(f"| Backend | {exp.get('backend', 'N/A')} |")
            parts.append(f"| Config Hash | {exp.get('config_hash', 'N/A')} |")

            metrics = exp.get("metrics_json")
            if isinstance(metrics, str):
                import json
                metrics = json.loads(metrics)
            if isinstance(metrics, dict):
                hp_str = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
                parts.append(f"| Hyperparams / Metrics | {hp_str} |")
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
        parts.append(r"\documentclass{article}")
        parts.append(r"\usepackage{booktabs}")
        parts.append(r"\usepackage{geometry}")
        parts.append(r"\geometry{margin=1in}")
        parts.append(r"\begin{document}")
        parts.append("")

        for idx, exp_id in enumerate(experiment_ids):
            data = self._fetch_experiment_data(exp_id)
            exp = data["experiment"]
            if exp is None:
                parts.append(rf"\section{{Experiment {_latex_escape(exp_id)}}}")
                parts.append("Experiment not found.")
                parts.append("")
                continue

            recipe = exp.get("recipe_id", exp_id)
            parts.append(rf"\section{{Experiment Report: {_latex_escape(recipe)}}}")
            parts.append("")

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

            metrics = exp.get("metrics_json")
            if isinstance(metrics, str):
                import json
                metrics = json.loads(metrics)
            if isinstance(metrics, dict):
                hp_str = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
                parts.append(rf"Hyperparams / Metrics & {_latex_escape(hp_str)} \\")

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
        """Generate a comparison table across multiple recipes.

        Produces a Markdown table with experiment ID, recipe, and key
        metrics side by side, with the best value per metric bolded.
        """
        import json as _json

        # Gather experiments for each recipe
        experiments: list[dict[str, Any]] = []
        for rid in recipe_ids:
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
        header_cols = ["Experiment ID", "Recipe", "Status"] + metric_names
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

            cols = [
                exp.get("id", "?"),
                exp.get("recipe_id", "?"),
                exp.get("status", "?"),
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
