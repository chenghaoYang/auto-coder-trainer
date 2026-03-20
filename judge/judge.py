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


def _as_dict(value: Any) -> dict[str, Any]:
    """Best-effort conversion of a dataclass or mapping-like object to dict."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _coerce_metrics(value: Any) -> dict[str, float]:
    """Normalise a metrics payload into a plain dict."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if isinstance(value, dict):
        return value
    return {}


def _coerce_train_result(recipe_id: str, results: dict[str, Any]) -> TrainResult:
    """Build a TrainResult from the supported result payload shapes."""
    source = results.get("train_result", results.get("train", {}))
    source_dict = _as_dict(source)

    train_metrics = _coerce_metrics(
        source_dict.get("metrics", source_dict.get("metrics_json", results.get("train_metrics", {})))
    )
    if not train_metrics:
        train_metrics = _coerce_metrics(results.get("metrics", {}))

    return TrainResult(
        recipe_id=source_dict.get("recipe_id", recipe_id),
        trainer_type=source_dict.get("trainer_type", results.get("trainer_type", "unknown")),
        backend=source_dict.get("backend", results.get("backend", "unknown")),
        status=source_dict.get(
            "status",
            results.get("status", results.get("train_status", "failed")),
        ),
        metrics=train_metrics,
        checkpoint_path=source_dict.get("checkpoint_path", results.get("checkpoint_path")),
        error=source_dict.get("error", results.get("error")),
    )


def _coerce_eval_results(recipe_id: str, results: dict[str, Any]) -> list[EvalResult]:
    """Build EvalResult objects from supported payload shapes."""
    raw_eval = results.get("eval_results", results.get("eval", []))
    if isinstance(raw_eval, dict):
        raw_eval = [raw_eval]
    if not isinstance(raw_eval, list):
        return []

    eval_results: list[EvalResult] = []
    for item in raw_eval:
        item_dict = _as_dict(item)
        if not item_dict:
            continue
        eval_results.append(
            EvalResult(
                recipe_id=item_dict.get("recipe_id", recipe_id),
                benchmark=item_dict.get("benchmark", item_dict.get("name", "")),
                metrics=_coerce_metrics(item_dict.get("metrics", item_dict.get("metrics_json", {}))),
                seed=item_dict.get("seed", 42),
                details={
                    k: v
                    for k, v in item_dict.items()
                    if k not in {"recipe_id", "benchmark", "name", "metrics", "metrics_json", "seed"}
                },
            )
        )
    return eval_results


def _coerce_recipe_payload(results: dict[str, Any]) -> dict[str, Any]:
    """Extract recipe-like payload from supported result shapes."""
    for key in ("recipe", "compiled_recipe", "config", "training_config"):
        candidate = results.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def _coerce_ablation_config(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract ablation config from either the result payload or embedded recipe."""
    if isinstance(results.get("ablation"), list):
        return results["ablation"]
    recipe = _coerce_recipe_payload(results)
    if isinstance(recipe.get("ablation"), list):
        return recipe["ablation"]
    return []


def _coerce_expected_seeds(results: dict[str, Any]) -> list[int]:
    """Extract expected seeds from the result payload, falling back to recipe config."""
    for key in ("expected_seeds", "seeds"):
        seeds = results.get(key)
        if isinstance(seeds, list):
            return seeds

    recipe = _coerce_recipe_payload(results)
    eval_section = recipe.get("eval", {})
    if isinstance(eval_section, dict) and isinstance(eval_section.get("seeds"), list):
        return eval_section["seeds"]

    eval_config = results.get("eval_config", {})
    if isinstance(eval_config, dict) and isinstance(eval_config.get("seeds"), list):
        return eval_config["seeds"]

    return []


def _aggregate_eval_metrics(eval_results: list[EvalResult]) -> dict[str, float]:
    """Compute mean metrics across evaluation seeds."""
    aggregated: dict[str, list[float]] = {}
    for result in eval_results:
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                aggregated.setdefault(key, []).append(float(value))

    return {
        key: sum(values) / len(values)
        for key, values in aggregated.items()
        if values
    }


def _current_metrics(recipe_id: str, results: dict[str, Any]) -> dict[str, float]:
    """Extract the best available current metrics for comparison."""
    metrics = _coerce_metrics(results.get("metrics", {}))
    if metrics:
        return metrics

    eval_results = _coerce_eval_results(recipe_id, results)
    if eval_results:
        eval_metrics = _aggregate_eval_metrics(eval_results)
        if eval_metrics:
            return eval_metrics

    train_result = _coerce_train_result(recipe_id, results)
    if train_result.metrics:
        return train_result.metrics

    return _coerce_metrics(results.get("train_metrics", {}))


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
            return True

        recipe = {"recipe_id": recipe_id, **_coerce_recipe_payload(results)}
        baseline = find_baseline(recipe, self.result_db)
        if baseline is None:
            # No prior baseline — this run *becomes* the baseline.
            return True

        baseline_metrics = _parse_metrics(baseline)
        current_metrics = _current_metrics(recipe_id, results)
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

        observed_seeds = {
            r.seed if isinstance(r, EvalResult) else r.get("seed")
            for r in results
        }
        if not set(expected_seeds).issubset(observed_seeds):
            return False

        # Check variance across seeds
        metric_values: dict[str, list[float]] = {}
        for r in results:
            metrics = r.metrics if isinstance(r, EvalResult) else r.get("metrics", {})
            for k, v in metrics.items():
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
            return True

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

        recipe = _coerce_recipe_payload(results)
        if not recipe:
            return True

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
        train_result = _coerce_train_result(recipe_id, results)
        eval_results = _coerce_eval_results(recipe_id, results)
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
        recipe_payload = _coerce_recipe_payload(results)
        eval_results_raw = _coerce_eval_results(recipe_id, results)
        expected_seeds = _coerce_expected_seeds(results)

        # 1. Baseline check
        checks["baseline"] = self.check_baseline(recipe_id, results)
        if not checks["baseline"]:
            reasoning_parts.append("Metrics regressed compared to baseline")
        elif self.result_db is None:
            reasoning_parts.append("Baseline check skipped (no result DB)")

        # 2. Seed check
        checks["seeds"] = self.check_seeds(eval_results_raw, expected_seeds)
        if not checks["seeds"]:
            reasoning_parts.append("Seed coverage incomplete or high variance")
            suggestions.append("Re-run evaluation with all required seeds")

        # 3. Ablation check
        ablation_config = _coerce_ablation_config(results)
        checks["ablation"] = self.check_ablation(recipe_id, ablation_config)
        if not checks["ablation"]:
            reasoning_parts.append("Missing ablation experiments")
            suggestions.append("Run missing ablation experiments before accepting")
        elif self.result_db is None:
            reasoning_parts.append("Ablation and dedup checks skipped (no result DB)")

        # 4. Dedup check
        checks["dedup"] = self.check_dedup(recipe_id, results)
        if not checks["dedup"]:
            reasoning_parts.append("Duplicate experiment already exists in DB")
            suggestions.append("Review existing experiment before re-running")

        # 5. Failure attribution (only if baseline failed)
        if not checks["baseline"]:
            recipe = {"recipe_id": recipe_id, **recipe_payload}
            baseline = find_baseline(recipe, self.result_db) if self.result_db else None
            if baseline is not None:
                attribution_msg = self.attribute_failure(
                    recipe_id, results, baseline,
                )
                reasoning_parts.append(attribution_msg)

        # 6. Determine verdict
        status = _coerce_train_result(recipe_id, results).status
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
