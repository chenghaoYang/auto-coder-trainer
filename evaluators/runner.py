"""Unified evaluation runner for benchmark dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluators.pass_at_k import compute_pass_at_k
from evaluators.swe_bench import SWEBenchEvaluator, VALID_VARIANTS


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _candidate_eval_files(root: Path, benchmark: str) -> list[Path]:
    candidates = [
        root / "eval" / f"{benchmark}.json",
        root / "eval" / f"{benchmark}.jsonl",
        root / f"{benchmark}.json",
        root / f"{benchmark}.jsonl",
        root / f"metrics.{benchmark}.json",
    ]
    return [candidate for candidate in candidates if candidate.exists()]


def _coerce_metrics_from_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if isinstance(payload.get("metrics"), dict):
            return payload["metrics"]
        if any(isinstance(v, (int, float)) for v in payload.values()):
            return {
                key: value
                for key, value in payload.items()
                if isinstance(value, (int, float))
            }
    return {}


def _coerce_rows_to_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    if all("n_samples" in row and "n_correct" in row for row in rows):
        metrics = compute_pass_at_k(rows, k_values=(1, 5, 10))
        metrics["num_problems"] = len(rows)
        return metrics

    if all("passed" in row for row in rows):
        total = len(rows)
        passed = sum(1 for row in rows if row.get("passed"))
        return {
            "pass@1": passed / total if total else 0.0,
            "passed": passed,
            "total": total,
        }

    return {}


def _resolve_local_metrics(checkpoint_path: str, benchmark: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(checkpoint_path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = _candidate_eval_files(root, benchmark)
    else:
        files = []

    for path in files:
        if path.suffix == ".json":
            payload = _load_json(path)
            metrics = _coerce_metrics_from_payload(payload)
            if metrics:
                return metrics, {"source_file": str(path)}
        if path.suffix == ".jsonl":
            rows = _load_jsonl(path)
            metrics = _coerce_rows_to_metrics(rows)
            if metrics:
                return metrics, {"source_file": str(path), "num_rows": len(rows)}

    raise RuntimeError(
        f"Could not resolve benchmark '{benchmark}' from checkpoint path '{checkpoint_path}'. "
        "Provide a predictions JSONL or benchmark metrics JSON file."
    )


def _resolve_swe_predictions_path(checkpoint_path: str, benchmark: str) -> str:
    path = Path(checkpoint_path)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    candidates = [
        path / f"{benchmark}.jsonl",
        path / "predictions.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    jsonl_files = sorted(path.rglob("*.jsonl"))
    if jsonl_files:
        return str(jsonl_files[0])
    raise FileNotFoundError(
        f"No predictions JSONL found for SWE-bench benchmark '{benchmark}' in {checkpoint_path}"
    )


def run_evaluation(
    *,
    checkpoint_path: str,
    benchmark: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Dispatch benchmark evaluation and return a normalized payload."""
    if benchmark in VALID_VARIANTS:
        evaluator = SWEBenchEvaluator(variant=benchmark)
        predictions_path = _resolve_swe_predictions_path(checkpoint_path, benchmark)
        result = evaluator.evaluate(predictions_path, seed=seed)
        return {
            "metrics": result.metrics,
            "details": {
                "num_samples": result.num_samples,
                "details": result.details,
                "predictions_path": predictions_path,
            },
        }

    metrics, details = _resolve_local_metrics(checkpoint_path, benchmark)
    return {"metrics": metrics, "details": details}
