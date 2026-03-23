"""Run a standardized benchmark for an auto-coder-trainer recipe.

Usage:
    python3 -m benchmarks.run_benchmark recipes/examples/baseline-sft.recipe.json
    python3 -m benchmarks.run_benchmark recipes/examples/baseline-sft.recipe.json --dry-run
    python3 -m benchmarks.run_benchmark recipes/examples/baseline-sft.recipe.json --gpu-type A100-80GB

Pipeline steps: validate recipe -> train (or dry-run) -> evaluate -> judge -> report
Outputs a structured "result card" JSON to the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _generate_benchmark_id(recipe_id: str, timestamp: str) -> str:
    """Generate a short deterministic benchmark ID."""
    digest = hashlib.sha256(f"{recipe_id}-{timestamp}".encode()).hexdigest()[:8]
    return f"bench-{recipe_id}-{digest}"


def _get_peak_memory_mb() -> float:
    """Return peak resident memory in MB (best-effort)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS
        if platform.system() == "Darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except Exception:
        return 0.0


def _collect_system_info(gpu_type: str | None) -> dict[str, Any]:
    """Gather hardware / environment info for reproducibility."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu_type": gpu_type or os.environ.get("GPU_TYPE", "unknown"),
    }
    # Try to detect GPU via nvidia-smi
    if info["gpu_type"] == "unknown":
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                info["gpu_type"] = result.stdout.strip().split("\n")[0]
        except Exception:
            pass
    return info


def _step_validate(recipe_path: Path) -> dict[str, Any]:
    """Step 1: Load and validate the recipe. Returns recipe dict."""
    logger.info("Step 1/5: Validating recipe %s", recipe_path)
    with open(recipe_path) as f:
        recipe = json.load(f)

    from recipes.compiler import load_schema, validate_recipe, normalize_recipe
    recipe = normalize_recipe(recipe)
    schema = load_schema()
    errors = validate_recipe(recipe, schema)
    if errors:
        raise ValueError(f"Recipe validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    logger.info("  Recipe %r validated successfully", recipe.get("id", "?"))
    return recipe


def _step_train(recipe: dict[str, Any], output_dir: Path, dry_run: bool) -> dict[str, Any]:
    """Step 2: Run training (or dry-run). Returns train metrics dict."""
    logger.info("Step 2/5: %s", "Dry-run training" if dry_run else "Training")

    from recipes.compiler import compile_recipe
    config = compile_recipe(recipe)

    if dry_run:
        logger.info("  Dry-run mode: skipping actual training")
        return {
            "trainer_type": config.trainer_type,
            "backend": config.backend,
            "model_base": config.model_config["base"],
            "dry_run": True,
            "train_loss": None,
        }

    # Attempt real training via the CLI train path
    try:
        from trainers.registry import get_trainer_class
        trainer_cls = get_trainer_class(config.trainer_type, config.backend)
        if trainer_cls is None:
            raise ImportError(
                f"No trainer registered for ({config.trainer_type!r}, {config.backend!r})"
            )
        trainer = trainer_cls(config, output_dir=str(output_dir))
        result = trainer.train()
        return {
            "trainer_type": config.trainer_type,
            "backend": config.backend,
            "model_base": config.model_config["base"],
            "dry_run": False,
            **(result if isinstance(result, dict) else {}),
        }
    except ImportError as exc:
        logger.warning("  Training backend not available: %s", exc)
        return {
            "trainer_type": config.trainer_type,
            "backend": config.backend,
            "model_base": config.model_config["base"],
            "dry_run": False,
            "skipped_reason": str(exc),
        }


def _step_evaluate(
    recipe: dict[str, Any], train_result: dict[str, Any], dry_run: bool,
) -> dict[str, Any]:
    """Step 3: Run evaluation. Returns eval metrics dict."""
    logger.info("Step 3/5: %s", "Dry-run evaluation" if dry_run else "Evaluating")

    eval_section = recipe.get("eval", {})
    benchmarks = eval_section.get("benchmarks", [])
    metrics_names = eval_section.get("metrics", [])
    seeds = eval_section.get("seeds", [42])

    if dry_run:
        logger.info("  Dry-run mode: generating placeholder metrics")
        placeholder: dict[str, Any] = {}
        for m in metrics_names:
            placeholder[m] = 0.0
        return {
            "benchmarks": benchmarks,
            "seeds": seeds,
            "metrics": placeholder,
            "dry_run": True,
        }

    # Attempt real evaluation
    try:
        from evaluators.run_eval import run_eval
        eval_results = run_eval(
            model_path=train_result.get("checkpoint_path", ""),
            benchmarks=benchmarks,
            seeds=seeds,
        )
        return eval_results
    except ImportError:
        logger.warning("  Evaluator not available; returning empty metrics")
        placeholder = {}
        for m in metrics_names:
            placeholder[m] = 0.0
        return {
            "benchmarks": benchmarks,
            "seeds": seeds,
            "metrics": placeholder,
            "eval_skipped": True,
        }


def _step_judge(
    recipe: dict[str, Any], eval_result: dict[str, Any], dry_run: bool,
) -> dict[str, Any]:
    """Step 4: Run the judge. Returns verdict dict."""
    logger.info("Step 4/5: %s", "Dry-run judging" if dry_run else "Judging results")

    if dry_run:
        return {"verdict": "dry_run", "reasoning": "Dry-run mode; no real judgment."}

    try:
        from judge.judge import judge_experiment
        verdict = judge_experiment(recipe, eval_result)
        return verdict if isinstance(verdict, dict) else {"verdict": str(verdict)}
    except ImportError:
        logger.warning("  Judge module not available; returning placeholder verdict")
        return {"verdict": "skipped", "reasoning": "Judge module not installed."}


def _step_report(
    benchmark_id: str,
    recipe: dict[str, Any],
    train_result: dict[str, Any],
    eval_result: dict[str, Any],
    judge_result: dict[str, Any],
    hw_info: dict[str, Any],
    duration_seconds: float,
    peak_memory_mb: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Step 5: Build and write the result card."""
    logger.info("Step 5/5: Generating result card")

    from benchmarks.result_card import ResultCard, render_result_card_json, render_result_card_markdown

    budget = recipe.get("budget", {})
    metrics = eval_result.get("metrics", {})
    metrics["peak_memory_mb"] = peak_memory_mb
    metrics["duration_seconds"] = duration_seconds

    card = ResultCard(
        benchmark_id=benchmark_id,
        recipe_id=recipe.get("id", "unknown"),
        model=train_result.get("model_base", recipe.get("model", {}).get("base", "unknown")),
        hardware=hw_info.get("gpu_type", "unknown"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        metrics=metrics,
        cost={
            "max_gpu_hours": budget.get("max_gpu_hours"),
            "max_cost_usd": budget.get("max_cost_usd"),
            "gpu_type": budget.get("gpu_type"),
        },
        duration_seconds=duration_seconds,
        status="success" if judge_result.get("verdict") != "fail" else "failed",
        reproducibility_info={
            "platform": hw_info.get("platform", ""),
            "python_version": hw_info.get("python_version", ""),
            "gpu_type": hw_info.get("gpu_type", ""),
            "trainer_type": train_result.get("trainer_type", ""),
            "backend": train_result.get("backend", ""),
            "dry_run": train_result.get("dry_run", False),
        },
        experiment_id=None,
        eval_runs=[],
        verdict=judge_result.get("verdict"),
        error=judge_result.get("error"),
    )

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    card_json = render_result_card_json(card)
    card_md = render_result_card_markdown(card)

    json_path = output_dir / f"{benchmark_id}.json"
    md_path = output_dir / f"{benchmark_id}.md"
    json_path.write_text(card_json)
    md_path.write_text(card_md)

    logger.info("  Result card written to %s", json_path)
    logger.info("  Markdown report written to %s", md_path)

    return card.to_dict()


def run_benchmark(
    recipe_path: str | Path,
    output_dir: str | Path = "outputs/benchmarks",
    dry_run: bool = False,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    """Run the full benchmark pipeline for a recipe.

    Args:
        recipe_path: Path to the recipe JSON file.
        output_dir: Directory for benchmark outputs.
        dry_run: If True, skip actual training and eval.
        gpu_type: Hardware tag to record in the result card.

    Returns:
        The result card as a dict.
    """
    recipe_path = Path(recipe_path)
    output_dir = Path(output_dir)
    start_time = time.monotonic()
    hw_info = _collect_system_info(gpu_type)
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Step 1: Validate
        recipe = _step_validate(recipe_path)

        benchmark_id = _generate_benchmark_id(recipe.get("id", "unknown"), timestamp)

        # Step 2: Train
        train_result = _step_train(recipe, output_dir / "train", dry_run)

        # Step 3: Evaluate
        eval_result = _step_evaluate(recipe, train_result, dry_run)

        # Step 4: Judge
        judge_result = _step_judge(recipe, eval_result, dry_run)

        # Step 5: Report
        duration = time.monotonic() - start_time
        peak_mem = _get_peak_memory_mb()

        card = _step_report(
            benchmark_id=benchmark_id,
            recipe=recipe,
            train_result=train_result,
            eval_result=eval_result,
            judge_result=judge_result,
            hw_info=hw_info,
            duration_seconds=duration,
            peak_memory_mb=peak_mem,
            output_dir=output_dir,
        )

        return card

    except Exception as exc:
        duration = time.monotonic() - start_time
        logger.error("Benchmark failed: %s", exc)
        error_card = {
            "benchmark_id": f"bench-error-{int(time.time())}",
            "recipe_id": recipe_path.stem,
            "model": "unknown",
            "hardware": hw_info.get("gpu_type", "unknown"),
            "timestamp": timestamp,
            "metrics": {},
            "cost": {},
            "duration_seconds": duration,
            "status": "error",
            "reproducibility_info": hw_info,
            "error": f"{type(exc).__name__}: {exc}",
        }
        # Write error card
        output_dir.mkdir(parents=True, exist_ok=True)
        err_path = output_dir / f"error-{int(time.time())}.json"
        err_path.write_text(json.dumps(error_card, indent=2))
        return error_card


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Run a standardized benchmark suite for an auto-coder-trainer recipe",
    )
    parser.add_argument("recipe", type=str, help="Path to recipe JSON file")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/benchmarks",
        help="Output directory for benchmark results (default: outputs/benchmarks)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and simulate without actual training/evaluation",
    )
    parser.add_argument(
        "--gpu-type", type=str, default=None,
        help="Hardware tag for result card (e.g., A100-80GB)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = run_benchmark(
        recipe_path=args.recipe,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        gpu_type=args.gpu_type,
    )

    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("status") != "error" else 1)


if __name__ == "__main__":
    main()
