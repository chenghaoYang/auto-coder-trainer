"""Results bridge for SWE-Lego training and evaluation.

Parses LLaMA-Factory training logs and SWE-bench evaluation results,
then imports them into auto-coder-trainer's result format for storage
and judging.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SWE_LEGO_ROOT = Path(__file__).parent / "SWE-Lego"


def parse_training_logs(log_dir: str | Path) -> dict[str, Any]:
    """Parse LLaMA-Factory training logs for metrics.

    Looks for ``trainer_state.json`` in *log_dir* (the training output_dir).

    Returns dict with keys like ``loss``, ``learning_rate``, ``epoch``, etc.
    """
    log_dir = Path(log_dir)
    state_path = log_dir / "trainer_state.json"

    if not state_path.exists():
        logger.warning("trainer_state.json not found in %s", log_dir)
        return {"status": "not_found", "log_dir": str(log_dir)}

    state = json.loads(state_path.read_text())

    log_history = state.get("log_history", [])
    if not log_history:
        return {
            "status": "empty",
            "epoch": state.get("epoch", 0),
            "global_step": state.get("global_step", 0),
        }

    # Extract final metrics from last log entry
    final = log_history[-1]
    # Also collect the best training loss across the run
    train_losses = [
        entry["loss"] for entry in log_history if "loss" in entry
    ]

    result: dict[str, Any] = {
        "status": "ok",
        "epoch": final.get("epoch", state.get("epoch", 0)),
        "global_step": final.get("step", state.get("global_step", 0)),
        "final_loss": final.get("loss") or final.get("train_loss"),
        "learning_rate": final.get("learning_rate"),
        "best_loss": min(train_losses) if train_losses else None,
        "num_log_entries": len(log_history),
    }

    # Include eval metrics if present
    eval_entries = [e for e in log_history if "eval_loss" in e]
    if eval_entries:
        last_eval = eval_entries[-1]
        result["eval_loss"] = last_eval.get("eval_loss")
        result["eval_epoch"] = last_eval.get("epoch")

    return result


def parse_swebench_results(
    results_dir: str | Path,
    run_id: str = "openhands",
) -> dict[str, Any]:
    """Parse SWE-bench evaluation results.

    Looks for the report JSON in *results_dir*.

    Returns dict with ``resolved_count``, ``total_count``, ``resolve_rate``,
    and ``per_instance`` details.
    """
    results_dir = Path(results_dir)

    # SWE-bench writes results as <run_id>.<dataset_slug>.json
    report_files = list(results_dir.glob(f"{run_id}*.json"))
    if not report_files:
        # Fallback: look for any JSON report
        report_files = list(results_dir.glob("*.json"))

    if not report_files:
        logger.warning("No SWE-bench report files found in %s", results_dir)
        return {"status": "not_found", "results_dir": str(results_dir)}

    # Use the most recent report
    report_path = max(report_files, key=lambda p: p.stat().st_mtime)
    report = json.loads(report_path.read_text())

    resolved = report.get("resolved", report.get("resolved_ids", []))
    total = report.get("total", report.get("total_instances", 0))

    if isinstance(resolved, list):
        resolved_count = len(resolved)
    else:
        resolved_count = int(resolved)

    if total == 0:
        # Try to infer from other fields
        applied = report.get("applied", report.get("applied_ids", []))
        if isinstance(applied, list):
            total = len(applied)

    resolve_rate = (resolved_count / total * 100.0) if total > 0 else 0.0

    per_instance: list[dict[str, Any]] = []
    for instance_id in report.get("resolved_ids", report.get("resolved", [])):
        if isinstance(instance_id, str):
            per_instance.append({"instance_id": instance_id, "resolved": True})

    return {
        "status": "ok",
        "report_path": str(report_path),
        "resolved_count": resolved_count,
        "total_count": total,
        "resolve_rate": resolve_rate,
        "per_instance": per_instance,
    }


def import_results(
    bundle_dir: str | Path,
    recipe_id: str,
    experiment_id: str,
    expected_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Import SWE-Lego training and eval results into auto-coder-trainer format.

    Returns dict with ``train_result`` (TrainResult-like dict) and
    ``eval_results`` (list of EvalResult-like dicts), suitable for feeding
    into ResultDB and ExperimentJudge.
    """
    bundle_dir = Path(bundle_dir)

    # Parse training logs — look in saves/ or output_dir
    train_metrics: dict[str, Any] = {"status": "not_found"}
    for candidate in [bundle_dir / "saves", bundle_dir / "output", bundle_dir]:
        if candidate.is_dir():
            for subdir in sorted(candidate.iterdir()):
                if (subdir / "trainer_state.json").exists():
                    train_metrics = parse_training_logs(subdir)
                    break
            if train_metrics.get("status") == "ok":
                break

    train_status = "success" if train_metrics.get("status") == "ok" else "failed"
    train_result = {
        "recipe_id": recipe_id,
        "trainer_type": "swe_lego",
        "backend": "llama_factory",
        "status": train_status,
        "metrics": {
            k: v for k, v in train_metrics.items()
            if isinstance(v, (int, float)) and v is not None
        },
        "checkpoint_path": None,
        "error": None if train_status == "success" else "Training logs not found or incomplete",
    }

    # Find checkpoint path
    for candidate in [bundle_dir / "saves", bundle_dir / "output", bundle_dir]:
        if candidate.is_dir():
            for subdir in sorted(candidate.iterdir(), reverse=True):
                if (subdir / "config.json").exists():
                    train_result["checkpoint_path"] = str(subdir)
                    break

    # Parse SWE-bench evaluation results
    eval_results: list[dict[str, Any]] = []
    swebench_results_dir = SWE_LEGO_ROOT / "SWE-bench-4.0.4" / "results"
    if swebench_results_dir.is_dir():
        seeds = [int(seed) for seed in (expected_seeds or [42])]
        seen_runs = 0
        for seed in seeds:
            run_id = f"openhands_seed_{seed}"
            swebench = parse_swebench_results(swebench_results_dir, run_id=run_id)
            if swebench.get("status") != "ok":
                continue
            eval_results.append({
                "recipe_id": recipe_id,
                "benchmark": "swe-bench-verified",
                "metrics": {
                    "resolved_count": swebench["resolved_count"],
                    "total_count": swebench["total_count"],
                    "resolve_rate": swebench["resolve_rate"],
                },
                "seed": seed,
                "details": {
                    "report_path": swebench.get("report_path"),
                    "per_instance": swebench.get("per_instance", []),
                    "run_id": run_id,
                },
            })
            seen_runs += 1

        if seen_runs == 0:
            swebench = parse_swebench_results(swebench_results_dir, run_id="openhands")
            if swebench.get("status") == "ok":
                fallback_seed = seeds[0] if seeds else 42
                eval_results.append({
                    "recipe_id": recipe_id,
                    "benchmark": "swe-bench-verified",
                    "metrics": {
                        "resolved_count": swebench["resolved_count"],
                        "total_count": swebench["total_count"],
                        "resolve_rate": swebench["resolve_rate"],
                    },
                    "seed": fallback_seed,
                    "details": {
                        "report_path": swebench.get("report_path"),
                        "per_instance": swebench.get("per_instance", []),
                        "run_id": "openhands",
                    },
                })

    return {
        "experiment_id": experiment_id,
        "recipe_id": recipe_id,
        "train_result": train_result,
        "eval_results": eval_results,
    }


def _simple_verdict(results: dict[str, Any]) -> str:
    """Fallback verdict based on resolve_rate thresholds."""
    if results["eval_results"]:
        best_rate = max(
            er["metrics"].get("resolve_rate", 0) for er in results["eval_results"]
        )
        if best_rate >= 30.0:
            return "strong"
        elif best_rate >= 15.0:
            return "moderate"
        elif best_rate > 0:
            return "weak"
        else:
            return "failed"
    elif results["train_result"]["status"] != "success":
        return "train_failed"
    return "unknown"


def import_and_judge(
    bundle_dir: str | Path,
    recipe_id: str,
    experiment_id: str | None = None,
    result_db: Any = None,
) -> dict[str, Any]:
    """Full import pipeline: parse results, run judge, generate report.

    When *result_db* is provided, uses :class:`ExperimentJudge` for a
    rigorous verdict and stores results in the DB.  Otherwise falls back
    to simple resolve-rate thresholds.

    Returns dict with ``experiment_id``, ``verdict``, and ``report_path``.
    """
    import uuid

    bundle_dir = Path(bundle_dir)
    if experiment_id is None:
        experiment_id = f"{recipe_id}-{uuid.uuid4().hex[:8]}"

    results = import_results(bundle_dir, recipe_id, experiment_id)

    # Try real ExperimentJudge first, fall back to simple thresholds
    verdict = _simple_verdict(results)
    judge_result = None
    try:
        from judge.judge import ExperimentJudge

        judge = ExperimentJudge(result_db=result_db)
        judge_result = judge.judge(recipe_id, results)
        verdict = judge_result.verdict.value
        logger.info(
            "ExperimentJudge verdict for %s: %s (%s)",
            experiment_id, verdict, judge_result.reasoning,
        )
    except Exception as exc:
        logger.warning(
            "ExperimentJudge unavailable, using simple verdict: %s", exc,
        )

    # Write report
    report_dir = bundle_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{experiment_id}_report.json"

    report: dict[str, Any] = {
        "experiment_id": experiment_id,
        "recipe_id": recipe_id,
        "verdict": verdict,
        "train_result": results["train_result"],
        "eval_results": results["eval_results"],
    }
    if judge_result is not None:
        report["judge_checks"] = judge_result.checks
        report["judge_reasoning"] = judge_result.reasoning
        report["judge_suggestions"] = judge_result.suggestions
    report_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info(
        "Experiment %s: verdict=%s, report=%s",
        experiment_id, verdict, report_path,
    )

    return {
        "experiment_id": experiment_id,
        "verdict": verdict,
        "report_path": str(report_path),
    }
