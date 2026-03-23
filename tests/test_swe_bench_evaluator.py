import json
import subprocess
from pathlib import Path

import pytest

from evaluators.runner import run_evaluation
from evaluators.swe_bench import (
    SWEBenchEvaluator,
    SWEBenchHarnessProcessError,
    SWEBenchHarnessTimeoutError,
)


def test_parse_report_prefers_report_schema(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "noise.json").write_text(json.dumps({"foo": "bar"}))
    (log_dir / "results.json").write_text(json.dumps({"resolved": ["i1"], "total": ["i1", "i2"]}))

    report = SWEBenchEvaluator._parse_report(str(log_dir), run_id="eval")
    assert report["resolved"] == ["i1"]


def test_run_harness_retries_and_fails(monkeypatch: pytest.MonkeyPatch):
    evaluator = SWEBenchEvaluator(retries=1)

    class DummyResult:
        returncode = 1
        stdout = "out"
        stderr = "err"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: DummyResult())

    with pytest.raises(SWEBenchHarnessProcessError):
        evaluator._run_harness("predictions.jsonl", "eval", "/tmp")


def test_run_harness_timeout_error(monkeypatch: pytest.MonkeyPatch):
    evaluator = SWEBenchEvaluator(retries=0, timeout=1)

    def _raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=1)

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(SWEBenchHarnessTimeoutError):
        evaluator._run_harness("predictions.jsonl", "eval", "/tmp")


def test_run_evaluation_swe_includes_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"instance_id":"i1","model_patch":"diff --git"}\n')

    monkeypatch.setattr(SWEBenchEvaluator, "_check_swebench_available", lambda self: None)

    def _fake_run_harness(self, predictions_path: str, run_id: str, log_dir: str):
        self.last_run_audit = {
            "attempt": 1,
            "attempts": 1,
            "command": "python -m swebench.harness.run_evaluation",
            "dataset": self.dataset,
            "run_id": run_id,
            "log_dir": log_dir,
            "returncode": 0,
        }
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        report = {"resolved": ["i1"], "total": ["i1", "i2"]}
        (Path(log_dir) / f"{run_id}.json").write_text(json.dumps(report))
        return subprocess.CompletedProcess(args=["python"], returncode=0)

    monkeypatch.setattr(SWEBenchEvaluator, "_run_harness", _fake_run_harness)

    result = run_evaluation(
        checkpoint_path=str(predictions),
        benchmark="swe-bench-lite",
        seed=42,
    )
    assert result["metrics"]["resolve_rate"] == 50.0
    assert result["details"]["audit"]["returncode"] == 0
    assert result["details"]["schema_version"] == "eval.v1"
