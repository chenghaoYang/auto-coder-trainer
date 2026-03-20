"""SWE-bench / SWE-rebench evaluator.

Wraps the SWE-bench evaluation harness CLI for standardized evaluation.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from evaluators.base import BaseEvaluator, BenchmarkResult

logger = logging.getLogger(__name__)

# Maps variant names to the dataset identifiers used by the swebench harness.
VARIANT_DATASET_MAP: dict[str, str] = {
    "swe-bench-lite": "princeton-nlp/SWE-bench_Lite",
    "swe-bench-verified": "princeton-nlp/SWE-bench_Verified",
    "swe-rebench": "princeton-nlp/SWE-rebench",
}

VALID_VARIANTS = tuple(VARIANT_DATASET_MAP.keys())


class SWEBenchEvaluator(BaseEvaluator):
    """Evaluator for SWE-bench family benchmarks.

    Supports: swe-bench-lite, swe-bench-verified, swe-rebench.
    """

    def __init__(
        self,
        variant: str = "swe-bench-lite",
        max_workers: int = 4,
        timeout: int = 1800,
    ):
        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"Unknown SWE-bench variant '{variant}'. "
                f"Valid variants: {', '.join(VALID_VARIANTS)}"
            )
        self.variant = variant
        self.dataset = VARIANT_DATASET_MAP[variant]
        self.max_workers = max_workers
        self.timeout = timeout

    def get_benchmark_name(self) -> str:
        return self.variant

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_swebench_available() -> None:
        """Raise a helpful error when the swebench package is missing."""
        try:
            import swebench  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "The 'swebench' package is not installed.  "
                "Install it with:  pip install swebench"
            )

    def _run_harness(
        self, predictions_path: str, run_id: str, log_dir: str
    ) -> subprocess.CompletedProcess:
        """Invoke the SWE-bench evaluation harness via subprocess."""
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self.dataset,
            "--predictions_path", predictions_path,
            "--max_workers", str(self.max_workers),
            "--run_id", run_id,
            "--log_dir", log_dir,
        ]
        logger.info("Running SWE-bench harness: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SWE-bench harness exited with code {result.returncode}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result

    @staticmethod
    def _parse_report(log_dir: str, run_id: str) -> dict[str, Any]:
        """Parse the JSON report produced by the harness.

        The harness writes a results JSON file into *log_dir*.  We look
        for the conventional location and fall back to scanning the
        directory.
        """
        log_path = Path(log_dir)
        # The harness typically writes <run_id>.json or results.json
        candidates = [
            log_path / f"{run_id}.json",
            log_path / "results.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return json.loads(candidate.read_text())

        # Fall back: find the first JSON file in the directory tree.
        json_files = sorted(log_path.rglob("*.json"))
        if json_files:
            return json.loads(json_files[0].read_text())

        raise FileNotFoundError(
            f"Could not locate evaluation results in {log_dir}"
        )

    def _build_result(
        self, report: dict[str, Any], seed: int
    ) -> BenchmarkResult:
        """Convert a raw harness report dict into a BenchmarkResult."""
        resolved = report.get("resolved", [])
        total = report.get("total", [])
        num_resolved = len(resolved) if isinstance(resolved, list) else int(resolved)
        num_total = len(total) if isinstance(total, list) else int(total)

        resolve_rate = (num_resolved / num_total * 100.0) if num_total else 0.0

        metrics: dict[str, float] = {
            "resolved": float(num_resolved),
            "total": float(num_total),
            "resolve_rate": round(resolve_rate, 2),
        }

        # Some variants expose extra metrics – propagate them.
        for key in ("applied", "error", "failed"):
            if key in report:
                val = report[key]
                metrics[key] = float(len(val) if isinstance(val, list) else val)

        # Build per-instance details.
        details: list[dict[str, Any]] = []
        if isinstance(resolved, list):
            for instance_id in resolved:
                details.append({"instance_id": instance_id, "resolved": True})

        return BenchmarkResult(
            benchmark=self.variant,
            metrics=metrics,
            seed=seed,
            num_samples=num_total,
            details=details,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self, model_path: str, seed: int = 42, **kwargs: Any
    ) -> BenchmarkResult:
        """Run SWE-bench evaluation on *model_path*.

        Args:
            model_path: Path to a predictions JSONL file (or a directory
                containing one) produced by the model under evaluation.
            seed: Random seed recorded in the result metadata.
            **kwargs: Forwarded options.  Recognised keys:
                ``run_id`` – identifier for the evaluation run
                    (default: ``"eval"``).
                ``log_dir`` – directory for harness logs.  A temporary
                    directory is used when not provided.

        Returns:
            A :class:`BenchmarkResult` with SWE-bench metrics.
        """
        self._check_swebench_available()

        predictions_path = model_path
        # If a directory was supplied, look for the conventional file name.
        pred_dir = Path(predictions_path)
        if pred_dir.is_dir():
            jsonl_files = list(pred_dir.glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(
                    f"No .jsonl prediction files found in {predictions_path}"
                )
            predictions_path = str(jsonl_files[0])

        run_id: str = kwargs.get("run_id", "eval")
        user_log_dir: str | None = kwargs.get("log_dir")

        if user_log_dir:
            log_dir = user_log_dir
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._run_harness(predictions_path, run_id, log_dir)
            report = self._parse_report(log_dir, run_id)
            return self._build_result(report, seed)

        # Use a temporary directory so we always clean up.
        with tempfile.TemporaryDirectory(prefix="swebench_eval_") as tmp_dir:
            log_dir = tmp_dir
            self._run_harness(predictions_path, run_id, log_dir)
            report = self._parse_report(log_dir, run_id)
            return self._build_result(report, seed)
