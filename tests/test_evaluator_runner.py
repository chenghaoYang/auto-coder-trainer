"""Unit tests for evaluators.runner — evaluation dispatch and metric resolution."""

import json
from pathlib import Path

import pytest

from evaluators.runner import (
    _coerce_metrics_from_payload,
    _coerce_rows_to_metrics,
    _normalize_metrics,
    _resolve_local_metrics,
    run_evaluation,
)


# ---------------------------------------------------------------------------
# _coerce_metrics_from_payload
# ---------------------------------------------------------------------------

class TestCoerceMetricsFromPayload:
    def test_extracts_nested_metrics_dict(self):
        payload = {"metrics": {"pass@1": 0.75, "resolve_rate": 0.42}, "extra": "ignored"}
        assert _coerce_metrics_from_payload(payload) == {"pass@1": 0.75, "resolve_rate": 0.42}

    def test_extracts_flat_numeric_values(self):
        payload = {"accuracy": 0.9, "total": 100, "name": "test"}
        result = _coerce_metrics_from_payload(payload)
        assert result == {"accuracy": 0.9, "total": 100}

    def test_returns_empty_for_non_dict(self):
        assert _coerce_metrics_from_payload([1, 2, 3]) == {}
        assert _coerce_metrics_from_payload("string") == {}

    def test_returns_empty_when_no_numeric_values(self):
        assert _coerce_metrics_from_payload({"a": "b", "c": "d"}) == {}


# ---------------------------------------------------------------------------
# _coerce_rows_to_metrics
# ---------------------------------------------------------------------------

class TestCoerceRowsToMetrics:
    def test_pass_at_k_from_n_samples_n_correct(self):
        rows = [
            {"problem_id": "p1", "n_samples": 10, "n_correct": 5},
            {"problem_id": "p2", "n_samples": 10, "n_correct": 10},
        ]
        result = _coerce_rows_to_metrics(rows)
        assert "pass@1" in result
        assert "num_problems" in result
        assert result["num_problems"] == 2
        assert 0.0 <= result["pass@1"] <= 1.0

    def test_binary_passed_field(self):
        rows = [
            {"id": "1", "passed": True},
            {"id": "2", "passed": False},
            {"id": "3", "passed": True},
        ]
        result = _coerce_rows_to_metrics(rows)
        assert result["pass@1"] == pytest.approx(2 / 3)
        assert result["passed"] == 2
        assert result["total"] == 3

    def test_empty_rows_returns_empty(self):
        assert _coerce_rows_to_metrics([]) == {}

    def test_unrecognised_schema_returns_empty(self):
        rows = [{"foo": "bar"}, {"baz": 42}]
        assert _coerce_rows_to_metrics(rows) == {}


class TestNormalizeMetrics:
    def test_normalizes_metric_keys_and_numeric_values(self):
        metrics = {
            "pass_at_1": 0.2,
            "PASS_10": 0.8,
            "resolve_rate_percent": 73.1,
            "name": "ignored",
        }
        assert _normalize_metrics(metrics) == {
            "pass@1": 0.2,
            "pass@10": 0.8,
            "resolve_rate": 73.1,
        }


# ---------------------------------------------------------------------------
# _resolve_local_metrics
# ---------------------------------------------------------------------------

class TestResolveLocalMetrics:
    def test_resolves_json_with_nested_metrics(self, tmp_path: Path):
        metrics_file = tmp_path / "eval" / "humaneval.json"
        metrics_file.parent.mkdir()
        metrics_file.write_text(json.dumps({"metrics": {"pass@1": 0.85}}))

        metrics, details = _resolve_local_metrics(str(tmp_path), "humaneval")
        assert metrics == {"pass@1": 0.85}
        assert "source_file" in details

    def test_resolves_jsonl_with_pass_at_k(self, tmp_path: Path):
        rows = [
            {"problem_id": "p1", "n_samples": 10, "n_correct": 8},
            {"problem_id": "p2", "n_samples": 10, "n_correct": 3},
        ]
        metrics_file = tmp_path / "eval" / "humaneval.jsonl"
        metrics_file.parent.mkdir()
        metrics_file.write_text("\n".join(json.dumps(r) for r in rows))

        metrics, details = _resolve_local_metrics(str(tmp_path), "humaneval")
        assert "pass@1" in metrics
        assert details["num_rows"] == 2

    def test_resolves_flat_json_metrics(self, tmp_path: Path):
        metrics_file = tmp_path / "eval" / "bench.json"
        metrics_file.parent.mkdir()
        metrics_file.write_text(json.dumps({"accuracy": 0.92, "f1": 0.88}))

        metrics, details = _resolve_local_metrics(str(tmp_path), "bench")
        assert metrics == {"accuracy": 0.92, "f1": 0.88}

    def test_raises_when_no_files_found(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="Could not resolve"):
            _resolve_local_metrics(str(tmp_path), "nonexistent")

    def test_resolves_direct_file_path(self, tmp_path: Path):
        metrics_file = tmp_path / "results.json"
        metrics_file.write_text(json.dumps({"metrics": {"score": 0.5}}))

        metrics, details = _resolve_local_metrics(str(metrics_file), "any")
        assert metrics == {"score": 0.5}


# ---------------------------------------------------------------------------
# run_evaluation (non-SWE-bench path)
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    def test_dispatches_generic_benchmark(self, tmp_path: Path):
        metrics_file = tmp_path / "eval" / "humaneval.json"
        metrics_file.parent.mkdir()
        metrics_file.write_text(json.dumps({"metrics": {"pass@1": 0.7}}))

        result = run_evaluation(
            checkpoint_path=str(tmp_path),
            benchmark="humaneval",
            seed=42,
        )
        assert result["metrics"] == {"pass@1": 0.7}
        assert "details" in result
        assert result["details"]["schema_version"] == "eval.v1"

    def test_raises_for_missing_benchmark(self, tmp_path: Path):
        with pytest.raises(RuntimeError):
            run_evaluation(
                checkpoint_path=str(tmp_path),
                benchmark="nonexistent",
                seed=42,
            )
