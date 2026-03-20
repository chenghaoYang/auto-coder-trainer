"""Experiment Judge — the core arbiter for training experiments.

Enforces scientific rigor: baseline alignment, seed consistency,
minimal ablation, result deduplication, and failure attribution.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from judge.baseline import find_baseline, compute_delta
from judge.ablation import validate_ablation_coverage
from judge.attribution import attribute_failure as _attribute_failure
from judge.dedup import compute_config_hash, find_duplicates
from trainers.base import TrainResult, EvalResult


class Verdict(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_ABLATION = "needs_ablation"
    NEEDS_RERUN = "needs_rerun"


@dataclass
class JudgementResult:
    """Result of experiment judgement."""
    verdict: Verdict
    recipe_id: str
    reasoning: str
    checks: dict[str, bool] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_metrics(experiment: dict[str, Any]) -> dict[str, float]:
    """Extract metrics dict from an experiment record."""
    raw = experiment.get("metrics", experiment.get("metrics_json", {}))
    if isinstance(raw, str):
        return json.loads(raw)
    return raw or {}


# Seed variance threshold: coefficient of variation must be below this
_MAX_SEED_CV = 0.10  # 10 %


class ExperimentJudge:
    """Judges experiment validity and decides whether to accept results.

    Judgement pipeline:
        1. check_baseline() — verify baseline run exists and is comparable
        2. check_seeds() — verify seed consistency across runs
        3. check_ablation() — verify minimal ablation coverage
        4. check_dedup() — check for duplicate experiments in result DB
        5. attribute_failure() — if below baseline, analyze why
        6. judge() — produce final verdict
    """

    def __init__(self, result_db: Any = None):
        self.result_db = result_db

    # ---- individual checks ------------------------------------------------

    def check_baseline(self, recipe_id: str, results: dict[str, Any]) -> bool:
        """Verify that a corresponding baseline experiment exists and that
        current results meet or exceed it.

        Returns True if baseline exists and current metrics are not worse
        (within a 5 % tolerance). Also returns True if no baseline exists
        yet (first run is the baseline).
        """
        if self.result_db is None:
            return False

        recipe = {"recipe_id": recipe_id, **(results.get("recipe", {}))}
        baseline = find_baseline(recipe, self.result_db)
        if baseline is None:
            # No prior baseline — this run *becomes* the baseline.
            return True

        baseline_metrics = _parse_metrics(baseline)
        current_metrics = results.get("metrics", {})
        if not baseline_metrics or not current_metrics:
            return True

        deltas = compute_delta(current_metrics, baseline_metrics)
        # Check that no metric regressed more than 5 %
        for key, val in deltas.items():
            if key.endswith("_relative_pct") and val < -5.0:
                return False
        return True

    def check_seeds(
        self,
        results: list[dict[str, Any]],
        expected_seeds: list[int],
    ) -> bool:
        """Verify that all expected seeds were run and variance is low.

        Returns True when every expected seed has a result AND the
        coefficient of variation for each metric is below the threshold.
        """
        if not expected_seeds:
            return True

        observed_seeds = {r.get("seed") for r in results}
        if not set(expected_seeds).issubset(observed_seeds):
            return False

        # Check variance across seeds
        metric_values: dict[str, list[float]] = {}
        for r in results:
            for k, v in r.get("metrics", {}).items():
                metric_values.setdefault(k, []).append(v)

        for key, vals in metric_values.items():
            if len(vals) < 2:
                continue
            mean = statistics.mean(vals)
            if mean == 0:
                continue
            cv = statistics.stdev(vals) / abs(mean)
            if cv > _MAX_SEED_CV:
                return False

        return True

    def check_ablation(
        self,
        recipe_id: str,
        ablation_config: list[dict],
    ) -> bool:
        """Verify that minimal ablation experiments were conducted.

        Returns True if every ablation defined in the config has a
        corresponding record in the result DB.
        """
        if not ablation_config:
            # No ablations required
            return True
        if self.result_db is None:
            return False

        recipe = {"recipe_id": recipe_id, "ablation": ablation_config}
        missing = validate_ablation_coverage(recipe, self.result_db)
        return len(missing) == 0

    def check_dedup(self, recipe_id: str, results: dict[str, Any]) -> bool:
        """Check if an equivalent experiment already exists in the result DB.

        Returns True if NO duplicates are found (i.e., experiment is novel).
        Returns False if duplicates exist.
        """
        if self.result_db is None:
            return True

        recipe = results.get("recipe", {})
        config_hash = compute_config_hash(recipe)
        duplicates = find_duplicates(config_hash, self.result_db)
        # Exclude the current experiment itself if it is already stored
        current_id = results.get("experiment_id")
        duplicates = [d for d in duplicates if d.get("id") != current_id]
        return len(duplicates) == 0

    def attribute_failure(
        self,
        recipe_id: str,
        results: dict[str, Any],
        baseline: dict[str, Any],
    ) -> str:
        """Analyze why an experiment performed below baseline.

        Delegates to :func:`attribution.attribute_failure` and returns
        a human-readable summary string.
        """
        train_result = results.get("train_result")
        if train_result is None:
            # Build a minimal TrainResult from raw dict
            train_result = TrainResult(
                recipe_id=recipe_id,
                trainer_type=results.get("trainer_type", "unknown"),
                backend=results.get("backend", "unknown"),
                status=results.get("status", "failed"),
                metrics=results.get("train_metrics", {}),
                checkpoint_path=results.get("checkpoint_path"),
                error=results.get("error"),
            )

        eval_results: list[EvalResult] = results.get("eval_results", [])
        if eval_results and isinstance(eval_results[0], dict):
            eval_results = [
                EvalResult(
                    recipe_id=recipe_id,
                    benchmark=e.get("benchmark", ""),
                    metrics=e.get("metrics", {}),
                    seed=e.get("seed", 42),
                )
                for e in eval_results
            ]

        baseline_metrics = _parse_metrics(baseline)
        report = _attribute_failure(train_result, eval_results, baseline_metrics)
        cause = report.get("likely_cause", "unknown")
        evidence = "; ".join(report.get("evidence", []))
        return f"Likely cause: {cause}. Evidence: {evidence}"

    # ---- orchestrator ------------------------------------------------------

    def judge(self, recipe_id: str, results: dict[str, Any]) -> JudgementResult:
        """Run full judgement pipeline and return verdict.

        Executes all checks in sequence and determines the appropriate
        verdict:
        - REJECT if training failed or metrics regressed significantly
        - NEEDS_RERUN if seed coverage is incomplete or duplicates found
        - NEEDS_ABLATION if ablation coverage is incomplete
        - ACCEPT if all checks pass
        """
        checks: dict[str, bool] = {}
        suggestions: list[str] = []
        reasoning_parts: list[str] = []

        # 1. Baseline check
        checks["baseline"] = self.check_baseline(recipe_id, results)
        if not checks["baseline"]:
            reasoning_parts.append("Metrics regressed compared to baseline")

        # 2. Seed check
        eval_results_raw = results.get("eval_results", [])
        expected_seeds = results.get("expected_seeds", results.get("seeds", []))
        checks["seeds"] = self.check_seeds(eval_results_raw, expected_seeds)
        if not checks["seeds"]:
            reasoning_parts.append("Seed coverage incomplete or high variance")
            suggestions.append("Re-run evaluation with all required seeds")

        # 3. Ablation check
        ablation_config = results.get("ablation", [])
        checks["ablation"] = self.check_ablation(recipe_id, ablation_config)
        if not checks["ablation"]:
            reasoning_parts.append("Missing ablation experiments")
            suggestions.append("Run missing ablation experiments before accepting")

        # 4. Dedup check
        checks["dedup"] = self.check_dedup(recipe_id, results)
        if not checks["dedup"]:
            reasoning_parts.append("Duplicate experiment already exists in DB")
            suggestions.append("Review existing experiment before re-running")

        # 5. Failure attribution (only if baseline failed)
        if not checks["baseline"]:
            recipe = {"recipe_id": recipe_id, **(results.get("recipe", {}))}
            baseline = find_baseline(recipe, self.result_db) if self.result_db else None
            if baseline is not None:
                attribution_msg = self.attribute_failure(
                    recipe_id, results, baseline,
                )
                reasoning_parts.append(attribution_msg)

        # 6. Determine verdict
        status = results.get("status", results.get("train_status", "success"))
        if status in ("failed", "timeout"):
            verdict = Verdict.REJECT
            reasoning_parts.insert(0, f"Training status: {status}")
        elif not checks["baseline"]:
            verdict = Verdict.REJECT
        elif not checks["seeds"]:
            verdict = Verdict.NEEDS_RERUN
        elif not checks["ablation"]:
            verdict = Verdict.NEEDS_ABLATION
        elif not checks["dedup"]:
            verdict = Verdict.NEEDS_RERUN
        else:
            verdict = Verdict.ACCEPT
            reasoning_parts.append("All checks passed")

        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "All checks passed"

        return JudgementResult(
            verdict=verdict,
            recipe_id=recipe_id,
            reasoning=reasoning,
            checks=checks,
            suggestions=suggestions,
        )
