"""Pre-defined benchmark suites for auto-coder-trainer.

Provides "quick", "standard", and "full" suites, each consisting of a list
of recipe configurations to run through the benchmark pipeline.

Usage:
    python3 -m benchmarks.suite --suite quick --dry-run
    python3 -m benchmarks.suite --suite standard --output-dir outputs/benchmarks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base directory for built-in benchmark recipes
SUITE_DIR = Path(__file__).parent / "recipes"

# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------
# Each suite entry has:
#   - recipe_file: filename in benchmarks/recipes/   OR
#   - recipe_inline: full recipe dict to write to a temp file
#   - dry_run: bool  (default for this entry; can be overridden globally)
#   - description: human-readable label

_QUICK_SUITE: list[dict[str, Any]] = [
    {
        "recipe_file": "quick_sft.recipe.json",
        "dry_run": True,
        "description": "Validate SFT recipe compilation and pipeline wiring",
    },
]

_STANDARD_SUITE: list[dict[str, Any]] = [
    {
        "recipe_file": "quick_sft.recipe.json",
        "dry_run": False,
        "description": "SFT baseline on small model with HumanEval",
    },
]

_FULL_SUITE: list[dict[str, Any]] = [
    {
        "recipe_file": "quick_sft.recipe.json",
        "dry_run": False,
        "description": "SFT baseline",
    },
    {
        "recipe_inline": {
            "id": "recipe-benchmark-rl-grpo",
            "name": "GRPO RL Benchmark",
            "version": "1.0",
            "source_papers": [],
            "model": {"base": "Qwen/Qwen2.5-Coder-0.5B", "size": "0.5B", "adapter": "lora"},
            "dataset": {
                "sources": [
                    {
                        "name": "swe-bench-trajectories",
                        "path": "bigcode/swe-bench-trajectories",
                        "mix_weight": 1.0,
                    }
                ],
                "filters": [],
                "total_samples": 50,
            },
            "trainer": {
                "type": "grpo",
                "backend": "trl",
                "params": {"lr": 1e-5, "ppo_epochs": 1, "rollout_batch_size": 4},
                "reward": {"type": "entropy_aware", "entropy_coeff": 0.01},
            },
            "eval": {"benchmarks": ["humaneval"], "metrics": ["pass@1"], "seeds": [42]},
            "ablation": [],
            "budget": {"max_gpu_hours": 2, "gpu_type": "any", "max_cost_usd": 4},
        },
        "dry_run": False,
        "description": "GRPO RL with entropy-aware reward",
    },
    {
        "recipe_inline": {
            "id": "recipe-benchmark-distill",
            "name": "Distillation Benchmark",
            "version": "1.0",
            "source_papers": [],
            "model": {"base": "Qwen/Qwen2.5-Coder-0.5B", "size": "0.5B", "adapter": "lora"},
            "dataset": {
                "sources": [
                    {
                        "name": "swe-bench-trajectories",
                        "path": "bigcode/swe-bench-trajectories",
                        "mix_weight": 1.0,
                    }
                ],
                "filters": [],
                "total_samples": 50,
            },
            "trainer": {
                "type": "distill",
                "backend": "trl",
                "params": {"lr": 2e-5, "epochs": 1, "batch_size": 2},
            },
            "eval": {"benchmarks": ["humaneval"], "metrics": ["pass@1"], "seeds": [42]},
            "ablation": [],
            "budget": {"max_gpu_hours": 2, "gpu_type": "any", "max_cost_usd": 4},
        },
        "dry_run": False,
        "description": "Trajectory distillation baseline",
    },
]

SUITES: dict[str, list[dict[str, Any]]] = {
    "quick": _QUICK_SUITE,
    "standard": _STANDARD_SUITE,
    "full": _FULL_SUITE,
}


def list_suites() -> list[str]:
    """Return available suite names."""
    return list(SUITES.keys())


def get_suite(name: str) -> list[dict[str, Any]]:
    """Get a suite definition by name.

    Args:
        name: Suite name ("quick", "standard", or "full").

    Returns:
        List of recipe run configs.

    Raises:
        KeyError: If the suite name is unknown.
    """
    if name not in SUITES:
        raise KeyError(
            f"Unknown suite {name!r}. Available: {', '.join(SUITES.keys())}"
        )
    return SUITES[name]


def _resolve_recipe(entry: dict[str, Any], output_dir: Path) -> Path:
    """Resolve a suite entry to a recipe file path."""
    if "recipe_file" in entry:
        return SUITE_DIR / entry["recipe_file"]

    if "recipe_inline" in entry:
        recipe = entry["recipe_inline"]
        recipe_path = output_dir / f"{recipe['id']}.recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2))
        return recipe_path

    raise ValueError(f"Suite entry must have 'recipe_file' or 'recipe_inline': {entry}")


def run_suite(
    suite_name: str,
    output_dir: str | Path = "outputs/benchmarks",
    *,
    dry_run: bool = False,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    """Run all benchmarks in a named suite.

    Args:
        suite_name: One of "quick", "standard", "full".
        output_dir: Base output directory for all benchmark results.
        dry_run: If True, force dry-run for all entries in the suite.
        gpu_type: Hardware tag for result cards.

    Returns:
        A summary dict with suite metadata and per-run results.
    """
    from benchmarks.run_benchmark import run_benchmark

    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite {suite_name!r}. Available: {list(SUITES)}")

    entries = SUITES[suite_name]
    output_path = Path(output_dir)
    suite_output = output_path / suite_name
    suite_output.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    total_start = time.monotonic()

    logger.info("Running benchmark suite %r (%d entries)", suite_name, len(entries))

    for i, entry in enumerate(entries, 1):
        desc = entry.get("description", f"Entry {i}")
        entry_dry_run = dry_run or entry.get("dry_run", False)

        logger.info(
            "Suite %s [%d/%d]: %s (dry_run=%s)",
            suite_name, i, len(entries), desc, entry_dry_run,
        )

        recipe_path = _resolve_recipe(entry, suite_output)
        run_output = suite_output / f"entry-{i}"

        try:
            card = run_benchmark(
                recipe_path=str(recipe_path),
                output_dir=str(run_output),
                dry_run=entry_dry_run,
                gpu_type=gpu_type,
            )
            card["suite_label"] = desc
            results.append(card)
        except Exception as exc:
            logger.error("Suite entry %d failed: %s", i, exc)
            results.append({
                "suite_label": desc,
                "recipe": str(recipe_path),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "dry_run": entry_dry_run,
            })

    total_duration = time.monotonic() - total_start

    summary: dict[str, Any] = {
        "suite_name": suite_name,
        "timestamp": timestamp,
        "total_duration_seconds": total_duration,
        "total_runs": len(results),
        "successful": sum(1 for r in results if r.get("status") not in ("error", "failed")),
        "failed": sum(1 for r in results if r.get("status") in ("error", "failed")),
        "dry_run": dry_run,
        "results": results,
    }

    # Write suite summary
    summary_path = suite_output / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "Suite %r complete: %d/%d succeeded in %.1fs. Summary: %s",
        suite_name, summary["successful"], summary["total_runs"],
        total_duration, summary_path,
    )

    return summary


def main():
    """CLI entry point for running benchmark suites."""
    parser = argparse.ArgumentParser(
        prog="benchmark_suite",
        description="Run a pre-defined benchmark suite for auto-coder-trainer",
    )
    parser.add_argument(
        "--suite", type=str, choices=list(SUITES.keys()),
        help="Suite to run: quick, standard, or full",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/benchmarks",
        help="Base output directory (default: outputs/benchmarks)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Force dry-run for all suite entries",
    )
    parser.add_argument(
        "--gpu-type", type=str, default=None,
        help="Hardware tag for result cards",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_suites",
        help="List available suites and exit",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_suites:
        for name in list_suites():
            suite_entries = SUITES[name]
            print(f"  {name}: {len(suite_entries)} recipe(s)")
            for entry in suite_entries:
                print(f"    - {entry.get('description', '?')}")
        sys.exit(0)

    summary = run_suite(
        suite_name=args.suite,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        gpu_type=args.gpu_type,
    )

    print(json.dumps(summary, indent=2, default=str))
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
