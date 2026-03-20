"""RL data loading and environment setup utilities.

Provides helpers to:
1. Load coding-task prompts for RL rollouts from HuggingFace datasets or local JSONL.
2. Apply configurable filters (issue quality, trajectory length, etc.).
3. Set up sandboxed rollout environments for code execution during training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_rl_prompts(
    sources: list[dict[str, Any]],
    filters: list[dict[str, Any]] | None = None,
    total_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load coding task prompts for RL rollouts.

    Each source dict is expected to contain:
        - name (str): human-readable identifier
        - path (str): HuggingFace dataset ID or local file path
        - mix_weight (float): sampling weight when mixing multiple sources

    Each returned prompt is a dict with at least ``prompt`` and ``metadata`` keys.

    Args:
        sources: Dataset source specifications from the recipe.
        filters: Optional filter specs ``[{type, params}]``.
        total_samples: Cap on the total number of prompts to return.

    Returns:
        List of prompt dicts ready for rollout.
    """
    if not sources:
        raise ValueError("At least one data source is required")

    all_prompts: list[dict[str, Any]] = []

    for source in sources:
        name = source.get("name", "unnamed")
        path = source["path"]
        weight = source.get("mix_weight", 1.0)

        logger.info("Loading RL prompts from %s (path=%s, weight=%.2f)", name, path, weight)
        raw = _load_from_path(path)
        logger.info("  loaded %d raw examples from %s", len(raw), name)

        # Normalise each example into a standard prompt dict
        normalised = [_normalise_prompt(ex, source_name=name) for ex in raw]
        all_prompts.extend(normalised)

    # Apply filters
    if filters:
        all_prompts = _apply_filters(all_prompts, filters)
        logger.info("After filtering: %d prompts", len(all_prompts))

    # Cap total samples
    if total_samples is not None and len(all_prompts) > total_samples:
        all_prompts = all_prompts[:total_samples]
        logger.info("Capped to %d prompts", total_samples)

    return all_prompts


def _load_from_path(path: str) -> list[dict[str, Any]]:
    """Load raw examples from a HuggingFace dataset or local file."""
    local = Path(path)

    # Local JSONL file
    if local.exists() and local.suffix in (".jsonl", ".json"):
        return _load_local(local)

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(path, split="train")
        return [dict(row) for row in ds]
    except Exception as exc:
        raise RuntimeError(
            f"Could not load dataset from '{path}'. "
            f"Ensure the path is a valid HuggingFace dataset ID or a local JSONL file. "
            f"Original error: {exc}"
        ) from exc


def _load_local(path: Path) -> list[dict[str, Any]]:
    """Load examples from a local JSON/JSONL file."""
    data: list[dict[str, Any]] = []
    with open(path) as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            content = json.load(f)
            data = content if isinstance(content, list) else [content]
    return data


def _normalise_prompt(example: dict[str, Any], source_name: str = "") -> dict[str, Any]:
    """Normalise a raw dataset example into a standard prompt dict.

    Looks for common field names used across trajectory datasets and maps them
    to a uniform schema.
    """
    # Try to find the prompt text
    prompt = (
        example.get("prompt")
        or example.get("instruction")
        or example.get("problem_statement")
        or example.get("query")
        or ""
    )

    # Try to find test specifications
    tests = (
        example.get("tests")
        or example.get("test_cases")
        or example.get("test")
        or []
    )

    return {
        "prompt": prompt,
        "tests": tests,
        "metadata": {
            "source": source_name,
            "instance_id": example.get("instance_id", example.get("id", "")),
            "repo": example.get("repo", ""),
            "original_fields": list(example.keys()),
        },
    }


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

_FILTER_REGISTRY: dict[str, Any] = {}


def _register_filter(name: str):
    """Decorator to register a filter function."""
    def decorator(fn):
        _FILTER_REGISTRY[name] = fn
        return fn
    return decorator


def _apply_filters(
    prompts: list[dict[str, Any]],
    filters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply a chain of filters to the prompt list."""
    for fspec in filters:
        filter_type = fspec["type"]
        params = fspec.get("params", {})
        fn = _FILTER_REGISTRY.get(filter_type)
        if fn is None:
            logger.warning("Unknown filter type %r — skipping", filter_type)
            continue
        prompts = fn(prompts, **params)
    return prompts


@_register_filter("issue_free")
def _filter_issue_free(prompts: list[dict[str, Any]], **_kwargs) -> list[dict[str, Any]]:
    """Keep only prompts that have a non-empty prompt string."""
    return [p for p in prompts if p.get("prompt")]


@_register_filter("length")
def _filter_length(
    prompts: list[dict[str, Any]],
    max_turns: int = 30,
    max_prompt_chars: int | None = None,
    **_kwargs,
) -> list[dict[str, Any]]:
    """Filter by prompt/trajectory length."""
    filtered = []
    for p in prompts:
        # max_turns applies if the prompt carries turn info
        turns = p.get("metadata", {}).get("turns", 0)
        if turns and turns > max_turns:
            continue
        if max_prompt_chars and len(p.get("prompt", "")) > max_prompt_chars:
            continue
        filtered.append(p)
    return filtered


@_register_filter("quality_score")
def _filter_quality_score(
    prompts: list[dict[str, Any]],
    min_score: float = 0.5,
    **_kwargs,
) -> list[dict[str, Any]]:
    """Keep only prompts whose metadata quality_score >= min_score."""
    return [
        p for p in prompts
        if p.get("metadata", {}).get("quality_score", 1.0) >= min_score
    ]


# ---------------------------------------------------------------------------
# Rollout environment
# ---------------------------------------------------------------------------

def setup_rollout_env(env_config: dict[str, Any]) -> dict[str, Any]:
    """Set up a sandboxed code execution environment for RL rollouts.

    Depending on the backend, this may launch Docker containers, set up
    a local venv, or configure a remote sandbox service.

    Args:
        env_config: Environment configuration, e.g.::
            {
                "type": "docker",               # or "local", "remote"
                "image": "python:3.11-slim",
                "timeout": 60,
                "memory_limit": "4g",
                "network": false
            }

    Returns:
        A dict describing the initialised environment, including:
            - env_type (str)
            - ready (bool)
            - execute_fn (callable): function(code: str) -> dict with
              ``stdout``, ``stderr``, ``exit_code``, ``tests_passed``, ``tests_total``
    """
    env_type = env_config.get("type", "local")
    timeout = env_config.get("timeout", 60)

    if env_type == "docker":
        return _setup_docker_env(env_config, timeout)
    elif env_type == "remote":
        return _setup_remote_env(env_config, timeout)
    else:
        return _setup_local_env(env_config, timeout)


def _setup_local_env(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Set up a local subprocess-based execution environment."""
    import subprocess
    import tempfile

    workdir = tempfile.mkdtemp(prefix="rl_rollout_")

    def execute_fn(code: str, test_code: str = "") -> dict[str, Any]:
        full_code = code
        if test_code:
            full_code += "\n\n" + test_code
        try:
            result = subprocess.run(
                ["python", "-c", full_code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
            )
            # Parse test results from output if available
            tests_passed, tests_total = _parse_test_output(result.stdout + result.stderr)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Timeout after {timeout}s",
                "exit_code": -1,
                "tests_passed": 0,
                "tests_total": 0,
            }

    return {"env_type": "local", "ready": True, "workdir": workdir, "execute_fn": execute_fn}


def _setup_docker_env(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Set up a Docker-based sandboxed execution environment."""
    image = env_config.get("image", "python:3.11-slim")
    memory_limit = env_config.get("memory_limit", "4g")
    network_disabled = not env_config.get("network", False)

    try:
        import subprocess
        # Verify Docker is available
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("Docker not available, falling back to local env: %s", exc)
        return _setup_local_env(env_config, timeout)

    def execute_fn(code: str, test_code: str = "") -> dict[str, Any]:
        import subprocess as sp
        full_code = code + ("\n\n" + test_code if test_code else "")
        cmd = [
            "docker", "run", "--rm",
            f"--memory={memory_limit}",
            *(["--network=none"] if network_disabled else []),
            image,
            "python", "-c", full_code,
        ]
        try:
            result = sp.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
            tests_passed, tests_total = _parse_test_output(result.stdout + result.stderr)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
            }
        except sp.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Docker timeout after {timeout}s",
                "exit_code": -1,
                "tests_passed": 0,
                "tests_total": 0,
            }

    return {"env_type": "docker", "ready": True, "image": image, "execute_fn": execute_fn}


def _setup_remote_env(env_config: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Placeholder for remote sandbox (e.g. Modal, E2B, etc.)."""
    logger.warning("Remote environment not yet fully implemented, returning stub")

    def execute_fn(code: str, test_code: str = "") -> dict[str, Any]:
        raise NotImplementedError(
            "Remote rollout environment requires configuring a sandbox service. "
            "Set env_config['endpoint'] and env_config['api_key']."
        )

    return {"env_type": "remote", "ready": False, "execute_fn": execute_fn}


def _parse_test_output(output: str) -> tuple[int, int]:
    """Best-effort extraction of test pass/total counts from pytest-style output.

    Looks for patterns like:
        ``5 passed, 2 failed`` (pytest)
        ``Ran 7 tests`` / ``OK`` (unittest)
    """
    import re

    # pytest style: "5 passed, 2 failed, 1 error"
    pytest_passed = re.search(r"(\d+)\s+passed", output)
    pytest_failed = re.search(r"(\d+)\s+failed", output)
    pytest_error = re.search(r"(\d+)\s+error", output)
    if pytest_passed or pytest_failed:
        passed = int(pytest_passed.group(1)) if pytest_passed else 0
        failed = int(pytest_failed.group(1)) if pytest_failed else 0
        errors = int(pytest_error.group(1)) if pytest_error else 0
        total = passed + failed + errors
        return passed, total

    # unittest style: "Ran N tests" + "OK" or "FAILED"
    unittest_ran = re.search(r"Ran\s+(\d+)\s+test", output)
    if unittest_ran:
        total = int(unittest_ran.group(1))
        if "OK" in output:
            return total, total
        # Try to extract failure count
        unittest_fail = re.search(r"failures=(\d+)", output)
        unittest_err = re.search(r"errors=(\d+)", output)
        fails = int(unittest_fail.group(1)) if unittest_fail else 0
        errs = int(unittest_err.group(1)) if unittest_err else 0
        return total - fails - errs, total

    return 0, 0
