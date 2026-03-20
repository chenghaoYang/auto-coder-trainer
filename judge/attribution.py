"""Failure attribution analysis for experiment judge."""

import json
from typing import Any

from trainers.base import TrainResult, EvalResult


def attribute_failure(
    train_result: TrainResult,
    eval_results: list[EvalResult],
    baseline_metrics: dict[str, float],
) -> dict[str, Any]:
    """Analyze why an experiment failed or underperformed.

    Inspects training status, eval metrics, and comparison with baseline
    to attribute the likely cause of failure.

    Returns attribution report with:
    - likely_cause: str
    - evidence: list[str]
    - suggested_fixes: list[str]
    """
    evidence: list[str] = []
    suggested_fixes: list[str] = []

    # ---- Case 1: Training itself failed ----
    if train_result.status == "failed":
        evidence.append(f"Training failed with error: {train_result.error or 'unknown'}")
        error_msg = (train_result.error or "").lower()

        if "oom" in error_msg or "out of memory" in error_msg or "cuda" in error_msg:
            return {
                "likely_cause": "out_of_memory",
                "evidence": evidence,
                "suggested_fixes": [
                    "Reduce batch size or sequence length",
                    "Enable gradient checkpointing",
                    "Use a smaller model or quantised training (LoRA/QLoRA)",
                ],
            }
        if "timeout" in error_msg or train_result.status == "timeout":
            return {
                "likely_cause": "timeout",
                "evidence": evidence,
                "suggested_fixes": [
                    "Increase training time budget",
                    "Reduce dataset size or number of epochs",
                ],
            }
        return {
            "likely_cause": "training_error",
            "evidence": evidence,
            "suggested_fixes": [
                "Check training logs for stack trace",
                "Verify data format matches expected schema",
                "Try with default hyperparameters to isolate the issue",
            ],
        }

    if train_result.status == "timeout":
        evidence.append("Training timed out before completing")
        return {
            "likely_cause": "timeout",
            "evidence": evidence,
            "suggested_fixes": [
                "Increase training time budget",
                "Reduce dataset size or number of epochs",
            ],
        }

    # ---- Case 2: Training succeeded but no eval results ----
    if not eval_results:
        evidence.append("Training succeeded but no evaluation results were produced")
        return {
            "likely_cause": "missing_eval",
            "evidence": evidence,
            "suggested_fixes": [
                "Verify checkpoint was saved correctly",
                "Check that evaluation benchmarks are configured",
            ],
        }

    # ---- Case 3: Metric regression — compare eval metrics with baseline ----
    # Aggregate eval metrics across seeds (mean)
    aggregated: dict[str, list[float]] = {}
    for er in eval_results:
        for k, v in er.metrics.items():
            aggregated.setdefault(k, []).append(v)

    mean_metrics = {k: sum(vs) / len(vs) for k, vs in aggregated.items()}

    degraded_metrics: list[str] = []
    for key in sorted(set(mean_metrics) & set(baseline_metrics)):
        cur = mean_metrics[key]
        base = baseline_metrics[key]
        if base > 0 and cur < base * 0.95:  # >5% regression
            pct = ((base - cur) / base) * 100
            degraded_metrics.append(key)
            evidence.append(
                f"Metric '{key}' regressed: {cur:.4f} vs baseline {base:.4f} "
                f"({pct:.1f}% drop)"
            )

    if not degraded_metrics:
        # Metrics are fine — no real failure to attribute
        return {
            "likely_cause": "none",
            "evidence": ["All metrics are within 5% of baseline"],
            "suggested_fixes": [],
        }

    # Heuristic: attempt to classify likely cause of metric degradation
    train_metrics = train_result.metrics or {}
    train_loss = train_metrics.get("loss", train_metrics.get("train_loss"))

    if train_loss is not None and train_loss > 2.0:
        likely = "underfitting"
        suggested_fixes = [
            "Increase number of training epochs",
            "Increase learning rate",
            "Verify training data quality and coverage",
        ]
    elif train_loss is not None and train_loss < 0.01:
        likely = "overfitting"
        suggested_fixes = [
            "Reduce number of training epochs",
            "Add or increase regularisation (dropout, weight decay)",
            "Increase dataset size or diversity",
        ]
    elif "reward" in " ".join(degraded_metrics).lower():
        likely = "reward_design"
        suggested_fixes = [
            "Review reward function for misalignment with target metric",
            "Check reward scaling and clipping parameters",
        ]
    else:
        likely = "hyperparameter_mismatch"
        suggested_fixes = [
            "Run ablation study over learning rate and batch size",
            "Compare hyperparameters with baseline configuration",
            "Check data preprocessing pipeline for differences",
        ]

    return {
        "likely_cause": likely,
        "evidence": evidence,
        "suggested_fixes": suggested_fixes,
    }
