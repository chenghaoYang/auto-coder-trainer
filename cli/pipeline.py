"""Pipeline command — orchestrate the full agent team as a closed loop.

Usage:
    act pipeline --query "coding agent training" --model Qwen/Qwen2.5-Coder-7B-Instruct
    act pipeline --recipe recipes/examples/baseline-sft.recipe.json
    act pipeline --recipe recipe.json --max-iterations 3

The pipeline chains: collect → compose → train/observe → judge → report.
After the judge verdict, it auto-decides:
  - ACCEPT       → generate final blog-style report, done.
  - NEEDS_RERUN  → re-run missing seeds or full experiment.
  - NEEDS_ABLATION → run ablation variants.
  - REJECT       → log failure analysis, stop (or retry with suggestions).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_collect(query: str, output_dir: str, max_papers: int = 20, max_repos: int = 10) -> Path:
    """Phase 1: Collect papers and repos into method atoms."""
    from cli.collect import run_collect

    registry_dir = Path(output_dir) / "registry"
    run_collect(
        argparse.Namespace(
            query=query,
            max_papers=max_papers,
            max_repos=max_repos,
            output=str(registry_dir),
        )
    )
    return registry_dir / "method_atoms.json"


def _run_compose(
    atoms: str,
    model: str,
    registry_path: Path,
    output_dir: str,
) -> Path:
    """Phase 2: Compose method atoms into a recipe."""
    import cli.compose as compose_mod
    from cli.compose import run_compose

    original_registry = compose_mod.REGISTRY_PATH
    compose_mod.REGISTRY_PATH = registry_path

    recipe_path = Path(output_dir) / "composed.recipe.json"
    try:
        run_compose(
            argparse.Namespace(
                atoms=atoms,
                model=model,
                output=str(recipe_path),
            )
        )
    finally:
        compose_mod.REGISTRY_PATH = original_registry

    return recipe_path


def _run_train(recipe_path: Path, output_dir: str, *, dry_run: bool = False) -> str | None:
    """Phase 3: Train and evaluate. Returns experiment_id from the DB or None."""
    from cli.train import run_train

    run_train(
        argparse.Namespace(
            recipe=str(recipe_path),
            output_dir=output_dir,
            dry_run=dry_run,
        )
    )

    # Find the experiment that was just created for this recipe
    recipe = json.loads(recipe_path.read_text())
    recipe_id = recipe.get("id", "")
    try:
        from results.db import ResultDB
        db = ResultDB()
        db.connect()
        try:
            latest_experiment = _get_latest_experiment(recipe_id, db=db)
            if latest_experiment:
                return latest_experiment["id"]
        finally:
            db.close()
    except Exception:
        pass
    return None


def _run_report(
    recipe_id: str,
    experiment_id: str | None,
    output_dir: str,
    *,
    fmt: str = "blog",
) -> Path | None:
    """Phase 4: Generate report. Returns report path."""
    from cli.report import run_report

    run_report(
        argparse.Namespace(
            experiment_id=experiment_id,
            recipe_id=recipe_id,
            format=fmt,
            output=output_dir,
        )
    )

    ext_map = {"blog": ".md", "markdown": ".md", "latex": ".tex"}
    ext = ext_map.get(fmt, ".md")
    report_path = Path(output_dir) / f"report{ext}"
    return report_path if report_path.exists() else None


def _coerce_verdict_payload(verdict: dict[str, Any], *, experiment_id: str | None = None) -> dict[str, Any]:
    """Normalize verdict payloads read from the DB."""
    payload = dict(verdict)
    if (
        "research_suggestions" not in payload
        and "research_suggestions_json" in payload
    ):
        payload["research_suggestions"] = payload.get("research_suggestions_json")
    if experiment_id is not None:
        payload["experiment_id"] = experiment_id
    return payload


def _get_latest_experiment(recipe_id: str, *, db=None) -> dict[str, Any] | None:
    """Return the newest experiment tracked for a recipe."""
    owns_db = False
    if db is None:
        from results.db import ResultDB
        db = ResultDB()
        db.connect()
        owns_db = True
    try:
        experiments = db.find_by_recipe(recipe_id)
        return experiments[0] if experiments else None
    finally:
        if owns_db:
            db.close()


def _get_experiment_verdict(experiment_id: str | None, *, db=None) -> dict[str, Any] | None:
    """Return the newest verdict for a specific experiment."""
    if not experiment_id:
        return None
    owns_db = False
    if db is None:
        from results.db import ResultDB
        db = ResultDB()
        db.connect()
        owns_db = True
    try:
        verdict = db.get_latest_verdict(experiment_id)
        return _coerce_verdict_payload(verdict, experiment_id=experiment_id) if verdict else None
    finally:
        if owns_db:
            db.close()


def _get_latest_verdict(recipe_id: str) -> dict[str, Any] | None:
    """Query the DB for the newest available judge verdict on a recipe."""
    try:
        from results.db import ResultDB
        db = ResultDB()
        db.connect()
        try:
            experiments = db.find_by_recipe(recipe_id)
            if not experiments:
                return None
            for experiment in experiments:
                exp_id = experiment["id"]
                verdict = db.get_latest_verdict(exp_id)
                if verdict:
                    return _coerce_verdict_payload(verdict, experiment_id=exp_id)
        finally:
            db.close()
    except Exception:
        pass
    return None


def _is_waiting_on_external_execution(
    recipe_id: str,
    experiment_id: str | None,
) -> bool:
    """Return True when the latest run is still waiting on manual/external work."""
    if not experiment_id:
        return False
    try:
        from results.db import ResultDB

        db = ResultDB()
        db.connect()
        try:
            experiment = db.get_experiment(experiment_id)
            if experiment is None:
                return False
            if experiment.get("status") not in {"planned", "prepared", "running"}:
                return False
            tasks = db.get_tasks(recipe_id=recipe_id, experiment_id=experiment_id)
            return any(
                task.get("kind") == "execution_step"
                and task.get("status") in {"pending", "blocked", "in_progress"}
                for task in tasks
            )
        finally:
            db.close()
    except Exception:
        return False


def _run_rerun(recipe_id: str, *, dry_run: bool = False) -> None:
    """Dispatch pending tasks via the rerun command."""
    from cli.rerun import run_rerun
    run_rerun(
        argparse.Namespace(recipe_id=recipe_id, dry_run=dry_run)
    )


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _has_research_suggestions(verdict: dict[str, Any] | None) -> bool:
    """Check whether the verdict carries actionable research suggestions."""
    if verdict is None:
        return False
    suggestions = verdict.get("research_suggestions", verdict.get("research_suggestions_json", []))
    if not isinstance(suggestions, list):
        return False
    for entry in suggestions:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "research_queries" and entry.get("trigger_collection"):
            return True
    return False


def _extract_research_queries(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the query dicts out of research_suggestions."""
    queries: list[dict[str, Any]] = []
    suggestions = verdict.get("research_suggestions", verdict.get("research_suggestions_json", []))
    for entry in suggestions:
        if isinstance(entry, dict) and entry.get("type") == "research_queries":
            queries.extend(entry.get("queries", []))
    return queries


def _decide_next_action(verdict: dict[str, Any] | None) -> str:
    """Return one of: 'report', 'rerun', 'ablation', 'research_and_retry',
    'stop', 'report_and_stop'.
    """
    if verdict is None:
        return "report_and_stop"

    v = verdict.get("verdict", "").lower()
    if v == "accept":
        return "report"
    elif v == "needs_rerun":
        if _has_research_suggestions(verdict):
            return "research_and_retry"
        return "rerun"
    elif v == "needs_ablation":
        return "ablation"
    elif v == "reject":
        if _has_research_suggestions(verdict):
            return "research_and_retry"
        return "report_and_stop"
    return "report_and_stop"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Orchestrate the full collect → compose → train → judge → report loop.

    Supports two entry modes:
      1. From scratch: --query + --atoms → collect, compose, then train.
      2. From recipe:  --recipe → skip collect/compose, go straight to train.

    After training, the pipeline inspects the judge verdict and auto-decides
    the next action. It iterates up to --max-iterations times before stopping.
    """
    query = getattr(args, "query", None)
    atoms = getattr(args, "atoms", None)
    recipe_path_arg = getattr(args, "recipe", None)
    model = getattr(args, "model", "Qwen/Qwen2.5-Coder-7B-Instruct")
    output_dir = getattr(args, "output_dir", "outputs/")
    report_dir = getattr(args, "report_dir", "reports/")
    dry_run = getattr(args, "dry_run", False)
    max_iterations = getattr(args, "max_iterations", 3)
    report_format = getattr(args, "report_format", "blog")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1 & 2: Collect + Compose (only if starting from scratch)
    # ------------------------------------------------------------------
    if recipe_path_arg:
        recipe_path = Path(recipe_path_arg)
        if not recipe_path.exists():
            print(f"[pipeline] Error: recipe file not found: {recipe_path}")
            sys.exit(1)
        print(f"[pipeline] === Using existing recipe: {recipe_path} ===")
    else:
        if not query:
            print("[pipeline] Error: provide --query (for full pipeline) or --recipe (to skip collect/compose)")
            sys.exit(1)

        print(f"\n[pipeline] {'='*60}")
        print(f"[pipeline] === Phase 1: COLLECT ===")
        print(f"[pipeline] {'='*60}")
        registry_path = _run_collect(query, output_dir)

        if not registry_path.exists():
            print("[pipeline] Error: collect phase produced no registry file.")
            sys.exit(1)

        atom_names = atoms
        if not atom_names:
            # Auto-select: use all collected atoms
            registry = json.loads(registry_path.read_text())
            all_atoms = [a["name"] for a in registry.get("atoms", [])]
            atom_names = ",".join(all_atoms) if all_atoms else ""
            if not atom_names:
                print("[pipeline] Error: no atoms collected. Try a different query.")
                sys.exit(1)
            print(f"[pipeline] Auto-selected atoms: {atom_names}")

        print(f"\n[pipeline] {'='*60}")
        print(f"[pipeline] === Phase 2: COMPOSE ===")
        print(f"[pipeline] {'='*60}")
        recipe_path = _run_compose(atom_names, model, registry_path, output_dir)

        if not recipe_path.exists():
            print("[pipeline] Error: compose phase produced no recipe.")
            sys.exit(1)

    # Read recipe for metadata
    recipe = json.loads(recipe_path.read_text())
    recipe_id = recipe.get("id", "unknown")

    # ------------------------------------------------------------------
    # Phase 3+: Train / Observe → Judge → Decide loop
    # ------------------------------------------------------------------
    pending_train = True
    experiment_id: str | None = None
    for iteration in range(1, max_iterations + 1):
        if pending_train:
            print(f"\n[pipeline] {'='*60}")
            print(f"[pipeline] === Phase 3: TRAIN (iteration {iteration}/{max_iterations}) ===")
            print(f"[pipeline] {'='*60}")
            experiment_id = _run_train(recipe_path, output_dir, dry_run=dry_run)
        else:
            print(f"\n[pipeline] {'='*60}")
            print(f"[pipeline] === Phase 3: OBSERVE (iteration {iteration}/{max_iterations}) ===")
            print(f"[pipeline] {'='*60}")
            print("[pipeline] Inspecting the latest dispatched experiment state.")

        latest_experiment = _get_latest_experiment(recipe_id)
        if latest_experiment is not None:
            experiment_id = latest_experiment["id"]

        # Get judge verdict for the newest experiment only
        verdict = _get_experiment_verdict(experiment_id)
        verdict_value = verdict.get("verdict", "unknown") if verdict else "no_verdict"
        print(f"\n[pipeline] Judge verdict: {verdict_value}")

        if verdict is None and _is_waiting_on_external_execution(recipe_id, experiment_id):
            print(f"\n[pipeline] {'='*60}")
            print("[pipeline] SLURM pipeline submitted — training is running asynchronously.")
            print(f"[pipeline] {'='*60}")
            # Show tracked SLURM jobs if available
            try:
                from results.db import ResultDB
                _db = ResultDB()
                _db.connect()
                try:
                    slurm_jobs = _db.get_slurm_jobs(recipe_id=recipe_id)
                    if slurm_jobs:
                        for sj in slurm_jobs:
                            print(f"[pipeline]   {sj.get('stage', '?'):20s}  "
                                  f"job {sj.get('job_id', '?'):>10s}  {sj.get('status', '?')}")
                finally:
                    _db.close()
            except Exception:
                pass
            print(f"\n[pipeline] Next steps:")
            print(f"[pipeline]   act status --recipe-id {recipe_id} --slurm   # live SLURM status")
            print(f"[pipeline]   act sync --recipe-id {recipe_id}             # import results when done")
            return
        if verdict is None and not pending_train:
            print("[pipeline] No verdict is available yet for the latest experiment.")
            print("[pipeline] Stop here and use `act status` / `act sync` to continue the loop.")
            return

        # Decide next action
        action = _decide_next_action(verdict)
        print(f"[pipeline] Decision: {action}")

        if action == "report":
            print(f"\n[pipeline] {'='*60}")
            print(f"[pipeline] === Phase 4: REPORT (final — accepted) ===")
            print(f"[pipeline] {'='*60}")
            report_path = _run_report(recipe_id, experiment_id, report_dir, fmt=report_format)
            if report_path:
                print(f"[pipeline] Report written to {report_path}")
            print(f"\n[pipeline] Pipeline complete. Experiment accepted after {iteration} iteration(s).")
            return

        elif action == "rerun":
            print(f"\n[pipeline] {'='*60}")
            print(f"[pipeline] === RERUN (dispatching pending tasks) ===")
            print(f"[pipeline] {'='*60}")
            _run_rerun(recipe_id, dry_run=dry_run)
            pending_train = False
            continue

        elif action == "research_and_retry":
            print(f"\n[pipeline] {'='*60}")
            print(f"[pipeline] === RESEARCH & RETRY (feedback-driven collection) ===")
            print(f"[pipeline] {'='*60}")

            research_queries = _extract_research_queries(verdict) if verdict else []
            if research_queries:
                # Sort by priority (lower number = higher priority)
                research_queries.sort(key=lambda q: q.get("priority", 99))
                for rq in research_queries:
                    print(f"[pipeline]   query: {rq.get('query', '')} "
                          f"(priority={rq.get('priority', '?')}, "
                          f"category={rq.get('target_category', '?')})")

                # Phase A: Run collect with suggested queries
                combined_query = " OR ".join(
                    rq["query"] for rq in research_queries if rq.get("query")
                )
                print(f"\n[pipeline] Collecting research for: {combined_query[:120]}...")
                registry_path = _run_collect(combined_query, output_dir)

                if registry_path.exists():
                    # Phase B: Compose new recipe variants from fresh atoms
                    registry = json.loads(registry_path.read_text())
                    all_atoms = [a["name"] for a in registry.get("atoms", [])]
                    atom_names = ",".join(all_atoms) if all_atoms else ""

                    if atom_names:
                        print(f"[pipeline] Composing new recipe from atoms: {atom_names}")
                        recipe_path = _run_compose(
                            atom_names, model, registry_path, output_dir,
                        )
                        if recipe_path.exists():
                            recipe = json.loads(recipe_path.read_text())
                            recipe_id = recipe.get("id", recipe_id)
                            print(f"[pipeline] New recipe composed: {recipe_id}")
                        else:
                            print("[pipeline] Warning: compose produced no recipe, "
                                  "continuing with existing recipe")
                    else:
                        print("[pipeline] Warning: no new atoms collected, "
                              "continuing with existing recipe")
                else:
                    print("[pipeline] Warning: collect produced no registry, "
                          "continuing with existing recipe")
            else:
                print("[pipeline] No research queries available, falling back to rerun")
                _run_rerun(recipe_id, dry_run=dry_run)
                pending_train = False
                continue
            pending_train = True
            continue

        elif action == "ablation":
            print(f"\n[pipeline] {'='*60}")
            print(f"[pipeline] === ABLATION (running missing variants) ===")
            print(f"[pipeline] {'='*60}")
            _run_rerun(recipe_id, dry_run=dry_run)
            pending_train = False
            continue

        elif action in ("stop", "report_and_stop"):
            print(f"\n[pipeline] {'='*60}")
            print(f"[pipeline] === Phase 4: REPORT (final — {verdict_value}) ===")
            print(f"[pipeline] {'='*60}")
            report_path = _run_report(recipe_id, experiment_id, report_dir, fmt=report_format)
            if report_path:
                print(f"[pipeline] Report written to {report_path}")
            if verdict_value == "reject":
                reasoning = verdict.get("reasoning", "") if verdict else ""
                print(f"[pipeline] Experiment rejected: {reasoning}")
                suggestions = verdict.get("suggestions_json", []) if verdict else []
                if suggestions:
                    print("[pipeline] Suggestions for next iteration:")
                    for s in (suggestions if isinstance(suggestions, list) else [suggestions]):
                        print(f"[pipeline]   - {s}")
            print(f"[pipeline] Pipeline stopped after {iteration} iteration(s).")
            return

    # ------------------------------------------------------------------
    # Max iterations reached
    # ------------------------------------------------------------------
    experiment_id = locals().get("experiment_id")  # safe if loop never ran
    print(f"\n[pipeline] {'='*60}")
    print(f"[pipeline] Max iterations ({max_iterations}) reached. Generating final report.")
    print(f"[pipeline] {'='*60}")
    report_path = _run_report(recipe_id, experiment_id, report_dir, fmt=report_format)
    if report_path:
        print(f"[pipeline] Report written to {report_path}")

    # Print final summary
    print(f"\n[pipeline] === PIPELINE SUMMARY ===")
    print(f"[pipeline] Recipe     : {recipe_id}")
    print(f"[pipeline] Iterations : {max_iterations}")
    verdict = _get_latest_verdict(recipe_id)
    verdict_value = verdict.get("verdict", "unknown") if verdict else "no_verdict"
    print(f"[pipeline] Final verdict: {verdict_value}")
    print("[pipeline] Done.")
