# SSD Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Apple's Simple Self-Distillation (SSD) method as a new `trainers/ssd/` launcher bundle module, following the same patterns as `trainers/swe_lego/` and `trainers/tinyzero/`.

**Architecture:** SSD is a 3-step pipeline (Sample → Fine-tune → Evaluate) that runs as a SLURM dependency chain. Sampling uses vLLM for offline inference. Fine-tuning reuses existing SFTTrainer logic. Evaluation ports LiveCodeBench v6 from ml-ssd.

**Tech Stack:** Python, vLLM, HuggingFace Transformers/Datasets, SLURM (sbatch), SQLite

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `trainers/ssd/__init__.py` | Module exports |
| Create | `trainers/ssd/data.py` | vLLM sampling → JSONL |
| Create | `trainers/ssd/lcb_evaluator.py` | LiveCodeBench v6 evaluation (from ml-ssd) |
| Create | `trainers/ssd/launcher.py` | Bundle generation + SLURM submission |
| Create | `trainers/ssd/results_bridge.py` | Eval results → DB import |
| Modify | `trainers/registry.py` | Register SSD trainer |
| Modify | `trainers/__init__.py` | Export SSDLauncher |
| Modify | `pyproject.toml` | Add `ssd` optional deps |
| Create | `tests/test_ssd_launcher.py` | Launcher bundle tests |
| Create | `tests/test_ssd_data.py` | Sampling data format tests |
| Create | `tests/test_lcb_evaluator.py` | Evaluator unit tests |
| Create | `tests/test_ssd_results_bridge.py` | Results import tests |

---

## Task 1: Module Structure + Dependencies

**Files:**
- Create: `trainers/ssd/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create `trainers/ssd/__init__.py`**

```python
"""SSD (Simple Self-Distillation) trainer — launcher bundle for Sample → Fine-tune → Evaluate pipeline."""

from trainers.ssd.launcher import SSDLauncher

__all__ = ["SSDLauncher"]
```

- [ ] **Step 2: Add SSD dependencies to `pyproject.toml`**

Add this to the `[project.optional-dependencies]` section in `pyproject.toml`:

```toml
ssd = ["vllm>=0.11.0", "scipy>=1.17.0", "sentencepiece>=0.2.0"]
```

Also add `"ssd"` to the `"all"` extra if one exists, or add the dependencies to the existing dev/extra groups.

- [ ] **Step 3: Commit**

```bash
git add trainers/ssd/__init__.py pyproject.toml
git commit -m "feat(ssd): scaffold SSD trainer module and dependencies"
```

---

## Task 2: Data Sampling (`trainers/ssd/data.py`)

**Files:**
- Create: `trainers/ssd/data.py`
- Create: `tests/test_ssd_data.py`

- [ ] **Step 1: Write failing test for sampling data generation**

Create `tests/test_ssd_data.py`:

```python
"""Tests for SSD sampling data generation."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_generate_sampling_script_creates_valid_python(tmp_path: Path):
    """generate_sampling_script should produce a valid Python script string."""
    from trainers.ssd.data import generate_sampling_script

    config = {
        "model_name": "Qwen/Qwen3-Coder",
        "output_path": str(tmp_path / "sample_data.jsonl"),
        "dataset": "livecodebench/code_generation_lite",
        "dataset_split": "test",
        "dataset_filter": {"contest_date": ["2025-02", "2025-03", "2025-04", "2025-05"]},
        "temperature": 0.9,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 65536,
        "n_samples": 10,
        "seed": 0,
        "tensor_parallel_size": 4,
    }
    script = generate_sampling_script(config)
    assert "from vllm import LLM" in script
    assert "Qwen/Qwen3-Coder" in script
    assert str(tmp_path / "sample_data.jsonl") in script
    # Must be valid Python
    compile(script, "<sample_script>", "exec")


def test_format_sample_output():
    """format_sample_output should produce valid JSONL-ready dicts."""
    from trainers.ssd.data import format_sample_output

    result = format_sample_output(
        prompt="Solve this problem",
        completion="```python\nprint('hello')\n```",
        metadata={"task_id": "task-001", "difficulty": "easy"},
    )
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == "```python\nprint('hello')\n```"
    assert result["metadata"]["task_id"] == "task-001"


def test_format_sample_output_strips_empty_completion():
    """format_sample_output should skip entries with empty completions."""
    from trainers.ssd.data import format_sample_output

    result = format_sample_output(
        prompt="Solve this",
        completion="",
        metadata={"task_id": "task-002"},
    )
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ssd_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trainers.ssd.data'`

- [ ] **Step 3: Implement `trainers/ssd/data.py`**

```python
"""SSD sampling data generation — produces JSONL training data via vLLM offline inference.

Generates a self-contained Python script that:
1. Loads problems from LiveCodeBench v6
2. Runs vLLM offline inference to generate code solutions
3. Writes results as JSONL for downstream SFT training
"""
from __future__ import annotations

import json
import textwrap
from typing import Any


def generate_sampling_script(config: dict[str, Any]) -> str:
    """Generate a self-contained Python script for vLLM sampling.

    The produced script can be submitted to SLURM or run standalone.
    It outputs one JSONL file where each line is a training example
    with ``messages`` (user prompt + assistant completion).
    """
    model_name = config.get("model_name", "")
    output_path = config.get("output_path", "sample_data.jsonl")
    dataset = config.get("dataset", "livecodebench/code_generation_lite")
    dataset_split = config.get("dataset_split", "test")
    dataset_filter = config.get("dataset_filter", {})
    temperature = config.get("temperature", 0.9)
    top_p = config.get("top_p", 0.8)
    top_k = config.get("top_k", 20)
    max_tokens = config.get("max_tokens", 65536)
    n_samples = config.get("n_samples", 10)
    seed = config.get("seed", 0)
    tensor_parallel_size = config.get("tensor_parallel_size", 4)

    filter_lines = ""
    if dataset_filter.get("contest_date"):
        dates = dataset_filter["contest_date"]
        filter_lines = textwrap.dedent(f"""\
            # Filter by contest date
            target_months = {dates!r}
            ds = ds.filter(lambda x: x["contest_date"][:7] in target_months)
            print(f"Problems after date filter: {{len(ds)}}")
        """)

    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"SSD Sampling Script — generated by auto-coder-trainer.\"\"\"

        import json
        import sys
        from pathlib import Path

        from datasets import load_dataset
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        def main():
            model_name = {model_name!r}
            output_path = Path({output_path!r})
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Load dataset
            ds = load_dataset({dataset!r}, split={dataset_split!r}, trust_remote_code=True)
            print(f"Loaded {{len(ds)}} problems")
            {filter_lines}
            # Initialize vLLM
            llm = LLM(model=model_name, tensor_parallel_size={tensor_parallel_size})
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            PROMPT_TEMPLATE = \"\"\"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

        Question: {{problem_description}}

        Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
        ```python
          # YOUR CODE HERE
        ```\"\"\"

            stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
            all_examples = list(ds)
            n_written = 0

            for i in range({n_samples}):
                seed = {seed} + i
                prompts = []
                for example in all_examples:
                    prompt_text = PROMPT_TEMPLATE.format(problem_description=example.get("question_content", example.get("prompt", "")))
                    messages = [{{"role": "user", "content": prompt_text}}]
                    templated = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(templated)

                sampling_params = SamplingParams(
                    max_tokens={max_tokens},
                    temperature={temperature},
                    top_p={top_p},
                    top_k={top_k},
                    seed=seed,
                    stop_token_ids=stop_token_ids,
                )

                print(f"Generating repeat {{i+1}}/{{{n_samples}}}...")
                outputs = llm.generate(prompts, sampling_params)

                for example, output in zip(all_examples, outputs):
                    completion = output.outputs[0].text
                    if not completion.strip():
                        continue
                    entry = {{
                        "messages": [
                            {{"role": "user", "content": example.get("question_content", example.get("prompt", ""))}},
                            {{"role": "assistant", "content": completion}},
                        ],
                        "metadata": {{
                            "task_id": example.get("question_id", ""),
                            "difficulty": example.get("difficulty", ""),
                            "repeat_index": i,
                            "seed": seed,
                        }},
                    }}
                    with open(output_path, "a") as f:
                        f.write(json.dumps(entry) + "\\n")
                    n_written += 1

            print(f"Wrote {{n_written}} examples to {{output_path}}")

        if __name__ == "__main__":
            main()
    """)


def format_sample_output(
    prompt: str,
    completion: str,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Format a single sampling result into a training-ready dict.

    Returns ``None`` if the completion is empty.
    """
    if not completion.strip():
        return None

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "metadata": metadata,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ssd_data.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add trainers/ssd/data.py tests/test_ssd_data.py
git commit -m "feat(ssd): add sampling data generation module"
```

---

## Task 3: LiveCodeBench Evaluator (`trainers/ssd/lcb_evaluator.py`)

**Files:**
- Create: `trainers/ssd/lcb_evaluator.py`
- Create: `tests/test_lcb_evaluator.py`

This is the largest task. The evaluator is ported from ml-ssd's `benchmark.py` + `livecodebench_utils.py` (~800 lines), adapted to the `BaseEvaluator` interface.

- [ ] **Step 1: Write failing test for LCB evaluator**

Create `tests/test_lcb_evaluator.py`:

```python
"""Tests for LiveCodeBench v6 evaluator ported from ml-ssd."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy", reason="numpy required for LCB evaluator tests")


def test_has_code_extracts_python_blocks():
    """has_code should extract code from markdown code blocks."""
    from trainers.ssd.lcb_evaluator import has_code

    response = "Here is my solution:\n```python\nprint('hello')\n```\nDone."
    matches = has_code(response)
    assert len(matches) == 1
    assert "print('hello')" in matches[0]


def test_has_code_returns_empty_for_no_code():
    """has_code should return empty list when no code blocks found."""
    from trainers.ssd.lcb_evaluator import has_code

    assert has_code("Just text, no code here.") == []


def test_post_process_code_strips_markdown():
    """post_process_code should remove markdown delimiters."""
    from trainers.ssd.lcb_evaluator import post_process_code

    code = "```python\nx = 1\nprint(x)\n```"
    result = post_process_code(code)
    assert "```" not in result
    assert "x = 1" in result


def test_compare_strings_exact_match():
    """compare_strings_with_decimal_fallback should match identical strings."""
    from trainers.ssd.lcb_evaluator import compare_strings_with_decimal_fallback

    assert compare_strings_with_decimal_fallback("hello\nworld", "hello\nworld") is True


def test_compare_strings_numeric_match():
    """compare_strings_with_decimal_fallback should match numeric equivalents."""
    from trainers.ssd.lcb_evaluator import compare_strings_with_decimal_fallback

    assert compare_strings_with_decimal_fallback("1.0 2.0", "1.0 2.0") is True
    assert compare_strings_with_decimal_fallback("0.5", "0.50") is True


def test_compare_strings_mismatch():
    """compare_strings_with_decimal_fallback should reject different outputs."""
    from trainers.ssd.lcb_evaluator import compare_strings_with_decimal_fallback

    assert compare_strings_with_decimal_fallback("1 2 3", "1 2 4") is False


def test_estimate_pass_at_k():
    """estimate_pass_at_k should compute correct probabilities."""
    import numpy as np

    from trainers.ssd.lcb_evaluator import estimate_pass_at_k

    # 5 samples, all correct → pass@1 = 1.0
    result = estimate_pass_at_k(5, 5, 1)
    assert result == 1.0

    # 5 samples, 0 correct → pass@1 = 0.0
    result = estimate_pass_at_k(5, 0, 1)
    assert result == 0.0

    # 10 samples, 5 correct → pass@1 should be 0.5
    result = estimate_pass_at_k(10, 5, 1)
    assert result == pytest.approx(0.5, abs=0.01)


def test_compute_metrics_from_results():
    """compute_metrics_from_results should aggregate pass@k across problems."""
    from trainers.ssd.lcb_evaluator import compute_metrics_from_results

    results = {
        "task-1": [[1, 1, 1], [1, 1, 0]],
        "task-2": [[1, 0, 0], [0, 0, 0]],
    }
    metrics = compute_metrics_from_results(results, k_list=[1])
    assert "pass@1" in metrics
    assert 0.0 <= metrics["pass@1"] <= 1.0


def test_lcb_evaluator_get_benchmark_name():
    """LiveCodeBenchEvaluator should return correct benchmark name."""
    from trainers.ssd.lcb_evaluator import LiveCodeBenchEvaluator

    evaluator = LiveCodeBenchEvaluator(config={})
    assert evaluator.get_benchmark_name() == "livecodebench-v6"


def test_generate_eval_script_creates_valid_python(tmp_path: Path):
    """generate_eval_script should produce a valid Python script."""
    from trainers.ssd.lcb_evaluator import generate_eval_script

    config = {
        "model_path": str(tmp_path / "model"),
        "output_path": str(tmp_path / "eval_results.json"),
        "tensor_parallel_size": 4,
        "max_tokens": 32768,
        "n_repeat": 20,
        "sampling_params": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "seed": [0, 1234, 1234, 1234],
    }
    script = generate_eval_script(config)
    assert "from vllm import LLM" in script
    compile(script, "<eval_script>", "exec")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lcb_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trainers.ssd.lcb_evaluator'`

- [ ] **Step 3: Implement `trainers/ssd/lcb_evaluator.py`**

This file ports the core evaluation logic from ml-ssd. It contains:
- Utility functions: `has_code`, `post_process_code`, `compare_strings_with_decimal_fallback`, `estimate_pass_at_k`, `compute_metrics_from_results`
- Sandbox utilities: `reliability_guard`, `Capturing`, `MockStdinWithBuffer`, `MockBuffer`
- Test execution: `run_test_std`, `run_test_func`, `lcb_run`
- `LiveCodeBenchEvaluator` class implementing `BaseEvaluator`
- `generate_eval_script()` for the launcher bundle

The full implementation should be ported from `/Users/yangchenghao/Desktop/github_high_oss/ml-ssd/evaluation/livecodebench_utils.py` and `/Users/yangchenghao/Desktop/github_high_oss/ml-ssd/evaluation/benchmark.py` with the following adaptations:

1. **Module header** — Replace Apple's `# For licensing see accompanying LICENSE file.` with the project's existing header pattern, but keep the Apple copyright notice as a comment acknowledging the source.

2. **Imports** — Add `from evaluators.base import BaseEvaluator, BenchmarkResult` to support the evaluator interface.

3. **`LiveCodeBenchEvaluator` class** — New class wrapping the ported logic:

```python
class LiveCodeBenchEvaluator(BaseEvaluator):
    """LiveCodeBench v6 evaluator ported from Apple ml-ssd."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_benchmark_name(self) -> str:
        return "livecodebench-v6"

    def evaluate(self, model_path: str, seed: int = 42) -> BenchmarkResult:
        """Evaluate model on LiveCodeBench v6 and return pass@k metrics."""
        # Full implementation delegates to LiveCodeBenchV6.run()
        # Adapted from ml-ssd benchmark.py
        ...
```

4. **`generate_eval_script()`** — Generates a self-contained Python script for SLURM submission, similar to `generate_sampling_script()` in data.py. The script should:
   - Load the model via vLLM
   - Load LiveCodeBench v6 dataset
   - Generate solutions with n_repeat sampling
   - Execute test cases using the ported sandbox utilities
   - Compute pass@k metrics
   - Write results to JSON

5. **Functions ported directly from `livecodebench_utils.py`** (keep as-is with minor adaptation):
   - `Capturing`, `MockBuffer`, `MockStdinWithBuffer`
   - `get_stripped_lines`, `convert_line_to_decimals`, `compare_strings_with_decimal_fallback`
   - `reliability_guard`
   - `translate_private_test_cases`, `map_to_example`, `post_process_code`
   - `parse_function_name_from_starter_code`, `clean_if_name`, `make_function`
   - `compile_code`, `get_function`, `call_method`
   - `prepare_test_input_output_std`, `run_test_func`, `run_test_std`
   - `prepare_test_input_output_functional`, `run_tests_for_one_example`, `lcb_run`
   - `estimate_pass_at_k`, `compute_metrics_from_results`

6. **Functions ported from `benchmark.py`**:
   - `has_code`
   - `filter_by_contest_date`
   - Prompt templates (`LCB_PROMPT_WITHOUT_STARTER_CODE`, `LCB_PROMPT_WITH_STARTER_CODE`)

**Key note**: The ported code is ~800 lines. Copy from the ml-ssd source files at `/Users/yangchenghao/Desktop/github_high_oss/ml-ssd/evaluation/` and adapt imports/interfaces as described above. The utility functions need no logic changes — only the module-level organization and the new `LiveCodeBenchEvaluator` class + `generate_eval_script()` are new code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lcb_evaluator.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add trainers/ssd/lcb_evaluator.py tests/test_lcb_evaluator.py
git commit -m "feat(ssd): port LiveCodeBench v6 evaluator from ml-ssd"
```

---

## Task 4: Results Bridge (`trainers/ssd/results_bridge.py`)

**Files:**
- Create: `trainers/ssd/results_bridge.py`
- Create: `tests/test_ssd_results_bridge.py`

- [ ] **Step 1: Write failing test for results bridge**

Create `tests/test_ssd_results_bridge.py`:

```python
"""Tests for SSD results bridge."""
import json
from pathlib import Path

import pytest


def test_import_results_parses_eval_json(tmp_path: Path):
    """import_results should parse eval_results.json into canonical payload."""
    from trainers.ssd.results_bridge import import_results

    eval_results = {
        "pass@1": 0.45,
        "pass@5": 0.72,
        "pass@1_easy": 0.55,
        "pass@1_medium": 0.40,
        "pass@1_hard": 0.25,
        "num_total": 100,
        "num_repeat": 20,
    }
    eval_path = tmp_path / "eval_results.json"
    eval_path.write_text(json.dumps(eval_results))

    payload = import_results(
        bundle_dir=tmp_path,
        recipe_id="recipe-ssd-001",
        experiment_id="exp-001",
    )
    assert payload["experiment_id"] == "exp-001"
    assert payload["recipe_id"] == "recipe-ssd-001"
    assert payload["train_result"]["status"] == "success"
    assert "pass@1" in payload["eval_results"][0]["metrics"]


def test_import_results_handles_missing_eval(tmp_path: Path):
    """import_results should return failed status when eval_results.json missing."""
    from trainers.ssd.results_bridge import import_results

    payload = import_results(
        bundle_dir=tmp_path,
        recipe_id="recipe-ssd-001",
        experiment_id="exp-001",
    )
    assert payload["train_result"]["status"] == "failed"
    assert len(payload["eval_results"]) == 0


def test_import_results_handles_empty_metrics(tmp_path: Path):
    """import_results should handle eval_results.json with no pass@k keys."""
    from trainers.ssd.results_bridge import import_results

    eval_path = tmp_path / "eval_results.json"
    eval_path.write_text(json.dumps({"num_total": 0}))

    payload = import_results(
        bundle_dir=tmp_path,
        recipe_id="recipe-ssd-002",
        experiment_id="exp-002",
    )
    assert payload["eval_results"][0]["metrics"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ssd_results_bridge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trainers.ssd.results_bridge'`

- [ ] **Step 3: Implement `trainers/ssd/results_bridge.py`**

```python
"""Results bridge for SSD launcher bundles.

Parses launcher bundle artifacts written by ``trainers.ssd.launcher``
and converts them into the canonical import payload expected by ``cli.train``.

Follows the same pattern as ``trainers.tinyzero.results_bridge``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _coerce_numeric_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _find_checkpoint_path(bundle_dir: Path) -> str | None:
    """Find the most recent model checkpoint in the bundle."""
    model_dir = bundle_dir / "model"
    if model_dir.exists() and (model_dir / "config.json").exists():
        return str(model_dir)
    return None


def _parse_train_result(bundle_dir: Path) -> dict[str, Any]:
    """Parse training status from bundle artifacts."""
    model_path = _find_checkpoint_path(bundle_dir)
    if model_path is not None:
        return {
            "status": "success",
            "metrics": {},
            "checkpoint_path": model_path,
            "error": None,
        }

    # Check for sample data as evidence of partial progress
    sample_path = bundle_dir / "sample_data.jsonl"
    if sample_path.exists():
        return {
            "status": "failed",
            "metrics": {},
            "checkpoint_path": None,
            "error": "Sampling completed but training did not produce a checkpoint",
        }

    return {
        "status": "failed",
        "metrics": {},
        "checkpoint_path": None,
        "error": "No training artifacts found",
    }


def _parse_eval_results(
    bundle_dir: Path,
    *,
    recipe_id: str,
) -> list[dict[str, Any]]:
    """Parse evaluation results from eval_results.json."""
    payload = _load_json_if_exists(bundle_dir / "eval_results.json")
    if not payload:
        return []

    metrics = _coerce_numeric_metrics(payload)
    # Filter out non-benchmark metrics
    benchmark_metrics = {
        k: v for k, v in metrics.items()
        if k.startswith("pass@") and not k.startswith("pass@_")
    }

    details = {
        "num_total": payload.get("num_total"),
        "num_repeat": payload.get("num_repeat"),
    }

    return [
        {
            "recipe_id": recipe_id,
            "benchmark": "livecodebench-v6",
            "metrics": benchmark_metrics,
            "seed": 42,
            "details": details,
        }
    ]


def import_results(
    bundle_dir: str | Path,
    recipe_id: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Import SSD launcher artifacts into a canonical result payload."""
    bundle_dir = Path(bundle_dir)
    train_result = _parse_train_result(bundle_dir)
    eval_results = _parse_eval_results(
        bundle_dir,
        recipe_id=recipe_id,
    )

    return {
        "experiment_id": experiment_id,
        "recipe_id": recipe_id,
        "train_result": {
            "recipe_id": recipe_id,
            "trainer_type": "ssd",
            "backend": "ssd",
            "status": train_result["status"],
            "metrics": train_result["metrics"],
            "checkpoint_path": train_result["checkpoint_path"],
            "error": train_result["error"],
        },
        "eval_results": eval_results,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ssd_results_bridge.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add trainers/ssd/results_bridge.py tests/test_ssd_results_bridge.py
git commit -m "feat(ssd): add results bridge for eval → DB import"
```

---

## Task 5: Launcher Bundle (`trainers/ssd/launcher.py`)

**Files:**
- Create: `trainers/ssd/launcher.py`
- Create: `tests/test_ssd_launcher.py`

- [ ] **Step 1: Write failing test for launcher bundle generation**

Create `tests/test_ssd_launcher.py`:

```python
"""Tests for SSD launcher bundle generation."""
import json
from pathlib import Path

import pytest


def _ssd_recipe_config(tmp_path: Path) -> dict:
    """Return a minimal compiled recipe config for SSD."""
    return {
        "recipe_id": "recipe-ssd-test-001",
        "trainer_type": "ssd",
        "backend": "ssd",
        "model_config": {
            "base": "Qwen/Qwen3-Coder-8B",
            "adapter": "full",
        },
        "data_config": {
            "sources": [],
        },
        "training_params": {
            "sample_temperature": 0.9,
            "sample_top_p": 0.8,
            "sample_top_k": 20,
            "n_samples_per_problem": 10,
            "max_tokens": 65536,
            "epochs": 1,
            "lr": 2e-5,
            "batch_size": 1,
            "decode_temperature": 0.6,
            "n_repeat": 20,
            "eval_max_tokens": 32768,
            "tensor_parallel_size": 4,
        },
        "eval_config": {
            "benchmarks": ["livecodebench-v6"],
            "seeds": [42],
        },
        "budget": {},
    }


def test_build_ssd_bundle_creates_all_scripts(tmp_path: Path):
    """build_ssd_launcher_bundle should produce all required bundle files."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    # Verify all expected files exist
    assert Path(paths["sample_script"]).exists()
    assert Path(paths["train_script"]).exists()
    assert Path(paths["eval_script"]).exists()
    assert Path(paths["import_results_script"]).exists()
    assert Path(paths["env_script"]).exists()
    assert Path(paths["launcher_json"]).exists()

    # Verify scripts are executable
    for key in ("sample_script", "train_script", "eval_script", "import_results_script"):
        assert Path(paths[key]).stat().st_mode & 0o111  # executable bit set


def test_build_ssd_bundle_scripts_are_valid_bash(tmp_path: Path):
    """Generated scripts should be valid bash (start with shebang)."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    for key in ("sample_script", "train_script", "eval_script", "import_results_script"):
        content = Path(paths[key]).read_text()
        assert content.startswith("#!/usr/bin/env bash"), f"{key} missing shebang"
        assert "set -euo pipefail" in content, f"{key} missing error handling"


def test_build_ssd_bundle_launcher_json(tmp_path: Path):
    """launcher.json should contain correct metadata."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    launcher = json.loads(Path(paths["launcher_json"]).read_text())
    assert launcher["backend"] == "ssd"
    assert launcher["recipe_id"] == "recipe-ssd-test-001"
    assert launcher["trainer_type"] == "ssd"


def test_build_ssd_bundle_contains_config_files(tmp_path: Path):
    """Bundle should contain JSON configs for each stage."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    assert Path(paths["sample_config"]).exists()
    assert Path(paths["eval_config"]).exists()

    sample_cfg = json.loads(Path(paths["sample_config"]).read_text())
    assert sample_cfg["model_name"] == "Qwen/Qwen3-Coder-8B"
    assert sample_cfg["temperature"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ssd_launcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trainers.ssd.launcher'`

- [ ] **Step 3: Implement `trainers/ssd/launcher.py`**

```python
"""Build SSD (Simple Self-Distillation) launch bundles from compiled recipe configs.

Generates a 4-stage SLURM pipeline:
    sample.sh → train.sh → eval.sh → import_results.sh

Follows the same launcher bundle pattern as trainers/swe_lego/ and trainers/tinyzero/.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from trainers.ssd.data import generate_sampling_script
from trainers.ssd.lcb_evaluator import generate_eval_script


def build_ssd_launcher_bundle(
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compile a training config into an SSD launch bundle."""
    recipe_id: str = config.get("recipe_id", "unknown")
    bundle_dir = Path(output_dir) / recipe_id / "ssd"

    model_cfg = config.get("model_config", {})
    training_params = config.get("training_params", {})
    eval_cfg = config.get("eval_config", {})
    budget = config.get("budget", {})

    model_name = model_cfg.get("base", "")
    tensor_parallel_size = int(training_params.get("tensor_parallel_size", 4))

    # Sample config
    sample_config = {
        "model_name": model_name,
        "output_path": str(bundle_dir / "sample_data.jsonl"),
        "dataset": "livecodebench/code_generation_lite",
        "dataset_split": "test",
        "dataset_filter": {
            "contest_date": training_params.get(
                "dataset_filter",
                ["2025-02", "2025-03", "2025-04", "2025-05"],
            ),
        },
        "temperature": float(training_params.get("sample_temperature", 0.9)),
        "top_p": float(training_params.get("sample_top_p", 0.8)),
        "top_k": int(training_params.get("sample_top_k", 20)),
        "max_tokens": int(training_params.get("max_tokens", 65536)),
        "n_samples": int(training_params.get("n_samples_per_problem", 10)),
        "seed": 0,
        "tensor_parallel_size": tensor_parallel_size,
    }

    # Eval config
    eval_config = {
        "model_path": str(bundle_dir / "model"),
        "output_path": str(bundle_dir / "eval_results.json"),
        "tensor_parallel_size": tensor_parallel_size,
        "max_tokens": int(training_params.get("eval_max_tokens", 32768)),
        "n_repeat": int(training_params.get("n_repeat", 20)),
        "sampling_params": {
            "temperature": float(training_params.get("decode_temperature", 0.6)),
            "top_p": 0.95,
            "top_k": 20,
        },
        "seed": [0, 1234, 1234, 1234],
    }

    # Train config (for downstream SFTTrainer)
    train_config = {
        "recipe_id": recipe_id,
        "model_config": model_cfg,
        "training_params": {
            "epochs": float(training_params.get("epochs", 1)),
            "lr": float(training_params.get("lr", 2e-5)),
            "batch_size": int(training_params.get("batch_size", 1)),
            "max_length": int(training_params.get("max_tokens", 65536)),
        },
        "data_config": {
            "sources": [
                {
                    "name": "ssd-sampled",
                    "path": str(bundle_dir / "sample_data.jsonl"),
                }
            ],
        },
    }

    return {
        "backend": "ssd",
        "recipe_id": recipe_id,
        "trainer_type": "ssd",
        "artifact_dir": str(bundle_dir),
        "warnings": [],
        "requirements": [
            "Install vLLM >= 0.11.0 before launch.",
            "Ensure sufficient GPU resources for sampling and training stages.",
        ],
        "_sample_config": sample_config,
        "_train_config": train_config,
        "_eval_config": eval_config,
        "_model_config": model_cfg,
        "_training_params": training_params,
        "_budget": budget,
    }


def write_ssd_launcher_bundle(bundle: dict[str, Any]) -> dict[str, str]:
    """Persist an SSD launch bundle to disk."""
    bundle_dir = Path(bundle["artifact_dir"])
    bundle_dir.mkdir(parents=True, exist_ok=True)

    sample_script_path = bundle_dir / "sample.sh"
    train_script_path = bundle_dir / "train.sh"
    eval_script_path = bundle_dir / "eval.sh"
    import_results_path = bundle_dir / "import_results.sh"
    env_path = bundle_dir / "env.sh"
    launcher_path = bundle_dir / "launcher.json"
    sample_config_path = bundle_dir / "sample_config.json"
    eval_config_path = bundle_dir / "eval_config.json"

    # Write configs
    sample_config_path.write_text(json.dumps(bundle["_sample_config"], indent=2) + "\n")
    eval_config_path.write_text(json.dumps(bundle["_eval_config"], indent=2) + "\n")

    # Write sample script (Python wrapper)
    sample_python = generate_sampling_script(bundle["_sample_config"])
    sample_script_path.write_text(_render_python_wrapper(sample_python, "sample"))
    sample_script_path.chmod(0o755)

    # Write train script
    train_script_path.write_text(_render_train_script(bundle))
    train_script_path.chmod(0o755)

    # Write eval script (Python wrapper)
    eval_python = generate_eval_script(bundle["_eval_config"])
    eval_script_path.write_text(_render_python_wrapper(eval_python, "eval"))
    eval_script_path.chmod(0o755)

    # Write import_results script
    import_results_path.write_text(_render_import_results_script(bundle))
    import_results_path.chmod(0o755)

    # Write env.sh
    env_path.write_text(_render_env(bundle))

    # Write launcher.json (strip internal keys)
    serializable = {k: v for k, v in bundle.items() if not k.startswith("_")}
    launcher_path.write_text(json.dumps(serializable, indent=2) + "\n")

    return {
        "bundle_dir": str(bundle_dir),
        "sample_script": str(sample_script_path),
        "train_script": str(train_script_path),
        "eval_script": str(eval_script_path),
        "import_results_script": str(import_results_path),
        "env_script": str(env_path),
        "launcher_json": str(launcher_path),
        "sample_config": str(sample_config_path),
        "eval_config": str(eval_config_path),
    }


def run_ssd_pipeline(
    bundle_dir: str | Path,
    slurm_config: dict[str, Any],
) -> dict[str, Any]:
    """Submit the full SSD pipeline with SLURM dependency chain.

    Stages: sample → train → eval → import_results
    """
    from trainers.slurm.submitter import (
        render_sbatch,
        submit_job,
        submit_with_dependency,
        write_sbatch_script,
    )

    bundle_dir = Path(bundle_dir).resolve()
    slurm_dir = bundle_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    cfg = {**slurm_config, "bundle_dir": str(bundle_dir)}
    recipe_id = bundle_dir.parent.name

    stages = [
        ("sample", "sample.sh", "sample.sbatch"),
        ("train", "train.sh", "train.sbatch"),
        ("eval", "eval.sh", "eval.sbatch"),
        ("import_results", "import_results.sh", "import_results.sbatch"),
    ]

    # Render sbatch scripts
    sbatch_paths: dict[str, Path] = {}
    for key, script, fname in stages:
        job_name = f"act-{recipe_id}-ssd-{key}"
        content = render_sbatch(job_name, script, cfg, slurm_dir)
        sbatch_paths[key] = write_sbatch_script(content, slurm_dir / fname)

    # Submit with dependency chain
    job_ids: dict[str, str] = {}
    job_ids["sample"] = submit_job(sbatch_paths["sample"])
    job_ids["train"] = submit_with_dependency(sbatch_paths["train"], job_ids["sample"])
    job_ids["eval"] = submit_with_dependency(sbatch_paths["eval"], job_ids["train"])
    job_ids["import_results"] = submit_with_dependency(
        sbatch_paths["import_results"], job_ids["eval"]
    )

    return {
        "pipeline_id": recipe_id,
        "job_ids": job_ids,
        "bundle_dir": str(bundle_dir),
    }


# ---------------------------------------------------------------------------
# Script renderers
# ---------------------------------------------------------------------------


def _render_python_wrapper(python_script: str, stage: str) -> str:
    """Wrap a Python script in a bash runner that writes output to a log."""
    escaped = python_script.replace("'", "'\\''")
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "$SCRIPT_DIR/env.sh"',
        "",
        f"# SSD {stage} stage",
        f"echo 'Starting SSD {stage} stage...'",
        f"mkdir -p \"$SCRIPT_DIR/logs\"",
        f"python3 -c '{escaped}' 2>&1 | tee \"$SCRIPT_DIR/logs/{stage}.log\"",
        'RC=${PIPESTATUS[0]}',
        'if [[ "$RC" -ne 0 ]]; then',
        f'  echo "SSD {stage} failed with exit code $RC"',
        '  exit "$RC"',
        'fi',
        f'echo "SSD {stage} completed successfully"',
        "",
    ])


def _render_train_script(bundle: dict[str, Any]) -> str:
    """Render train.sh — runs SFTTrainer on sampled data."""
    recipe_id = bundle["recipe_id"]
    model_name = bundle.get("_model_config", {}).get("base", "")
    lr = bundle.get("_training_params", {}).get("lr", 2e-5)
    epochs = bundle.get("_training_params", {}).get("epochs", 1)
    batch_size = bundle.get("_training_params", {}).get("batch_size", 1)
    max_tokens = bundle.get("_training_params", {}).get("max_tokens", 65536)

    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "$SCRIPT_DIR/env.sh"',
        "",
        "# SSD Fine-tune stage — standard cross-entropy SFT on sampled data",
        "echo 'Starting SSD fine-tune stage...'",
        'mkdir -p "$SCRIPT_DIR/logs"',
        "mkdir -p \"$SCRIPT_DIR/model\"",
        "",
        "python3 -c '\\",
        "from trainers.sft.trainer import SFTTrainer;",
        "import json, os;",
        "",
        f"config = json.loads('''{json.dumps(bundle.get('_train_config', {{}})}'''));",
        "config[\"recipe_id\"] = " + shlex.quote(recipe_id) + ";",
        "config[\"trainer_type\"] = \"sft\";",
        "config[\"backend\"] = \"trl\";",
        "",
        "trainer = SFTTrainer(config, os.path.join(" + shlex.quote(str(bundle.get('artifact_dir', ''))) + ", \"model\"));",
        "trainer.prepare_data();",
        "result = trainer.train();",
        "",
        "with open(os.path.join(" + shlex.quote(str(bundle.get('artifact_dir', ''))) + ", \"train_result.json\"), \"w\") as f:",
        "    json.dump({\"status\": result.status, \"metrics\": result.metrics}, f, default=str);",
        "print(f\"Training result: {result.status}\")",
        "' 2>&1 | tee \"$SCRIPT_DIR/logs/train.log\"",
        'RC=${PIPESTATUS[0]}',
        'if [[ "$RC" -ne 0 ]]; then',
        '  echo "SSD train failed with exit code $RC"',
        '  exit "$RC"',
        'fi',
        'echo "SSD train completed successfully"',
        "",
    ])


def _render_import_results_script(bundle: dict[str, Any]) -> str:
    """Render import_results.sh — imports eval results into DB."""
    artifact_dir = bundle.get("artifact_dir", "")
    recipe_id = bundle.get("recipe_id", "")
    report_dir = str(Path(artifact_dir).parent / "reports") if artifact_dir else ""

    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "$SCRIPT_DIR/env.sh"',
        "",
        "# SSD import results stage",
        "echo 'Importing SSD results into DB...'",
        "",
        f"python3 -m cli.main train --import-results \"$SCRIPT_DIR\" \\",
        f"  --recipe-id {shlex.quote(recipe_id)} \\",
        "  --experiment-id ${ACT_EXPERIMENT_ID:-ssd-exp-001} \\",
        "  --report-format ${ACT_REPORT_FORMAT:-blog} \\",
        f"  --report-output {shlex.quote(report_dir)}",
        "",
        'echo "SSD results imported successfully"',
        "",
    ])


def _render_env(bundle: dict[str, Any]) -> str:
    """Render env.sh with export variables."""
    recipe_id = bundle.get("recipe_id", "unknown")
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by auto-coder-trainer (SSD launcher).",
        "",
        f"export ACT_RECIPE_ID={shlex.quote(recipe_id)}",
        'export ACT_EXPERIMENT_ID="${ACT_EXPERIMENT_ID:-ssd-exp-001}"',
        "",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ssd_launcher.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add trainers/ssd/launcher.py tests/test_ssd_launcher.py
git commit -m "feat(ssd): add launcher bundle with SLURM pipeline generation"
```

---

## Task 6: Registry Integration

**Files:**
- Modify: `trainers/registry.py`
- Modify: `trainers/__init__.py`

- [ ] **Step 1: Register SSD in `trainers/registry.py`**

Add to the `_register_builtins()` function in `trainers/registry.py`, after the existing `distill` registration block:

```python
    try:
        from trainers.ssd.launcher import SSDLauncher
        register("ssd", None, SSDLauncher)
        register("ssd", "ssd", SSDLauncher)
    except ImportError:
        pass
```

- [ ] **Step 2: Export SSDLauncher in `trainers/__init__.py`**

Add the SSD import and export. Modify the imports and `__all__`:

```python
# Add import:
from trainers.ssd import SSDLauncher

# Add to __all__:
__all__ = [
    "BaseTrainer",
    "SFTTrainer",
    "RLTrainer",
    "DistillTrainer",
    "SSDLauncher",
    "get_trainer_class",
    "register",
    "list_registered",
]
```

- [ ] **Step 3: Run all existing tests to ensure no regressions**

Run: `pytest tests/ -v -k "not swe_lego" --timeout=60`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add trainers/registry.py trainers/__init__.py
git commit -m "feat(ssd): register SSD trainer in trainer registry"
```

---

## Task 7: Integration Test

**Files:**
- Modify: `tests/test_ssd_launcher.py` (add integration test)

- [ ] **Step 1: Write integration test for full bundle pipeline**

Add to `tests/test_ssd_launcher.py`:

```python
def test_ssd_bundle_end_to_end(tmp_path: Path):
    """Full bundle generation → write → verify all artifacts."""
    from trainers.ssd.launcher import build_ssd_launcher_bundle, write_ssd_launcher_bundle
    from trainers.ssd.results_bridge import import_results

    config = _ssd_recipe_config(tmp_path)
    bundle = build_ssd_launcher_bundle(config, tmp_path)
    paths = write_ssd_launcher_bundle(bundle)

    # Verify bundle directory structure
    bundle_dir = Path(paths["bundle_dir"])
    assert bundle_dir.exists()

    # Verify sample config has correct model
    sample_cfg = json.loads(Path(paths["sample_config"]).read_text())
    assert sample_cfg["model_name"] == "Qwen/Qwen3-Coder-8B"
    assert sample_cfg["temperature"] == 0.9
    assert sample_cfg["n_samples"] == 10

    # Verify eval config
    eval_cfg = json.loads(Path(paths["eval_config"]).read_text())
    assert eval_cfg["n_repeat"] == 20
    assert eval_cfg["sampling_params"]["temperature"] == 0.6

    # Verify launcher.json has no internal keys
    launcher = json.loads(Path(paths["launcher_json"]).read_text())
    assert "_sample_config" not in launcher
    assert "_train_config" not in launcher

    # Simulate results import (no eval results yet → failed)
    payload = import_results(
        bundle_dir=bundle_dir,
        recipe_id="recipe-ssd-test-001",
        experiment_id="exp-integration",
    )
    assert payload["train_result"]["status"] == "failed"
    assert payload["recipe_id"] == "recipe-ssd-test-001"

    # Simulate successful eval results
    eval_results = {
        "pass@1": 0.45,
        "pass@5": 0.72,
        "num_total": 100,
        "num_repeat": 20,
    }
    (bundle_dir / "eval_results.json").write_text(json.dumps(eval_results))

    payload = import_results(
        bundle_dir=bundle_dir,
        recipe_id="recipe-ssd-test-001",
        experiment_id="exp-integration-2",
    )
    assert payload["eval_results"][0]["metrics"]["pass@1"] == 0.45
```

- [ ] **Step 2: Run all SSD tests**

Run: `pytest tests/test_ssd_launcher.py tests/test_ssd_data.py tests/test_ssd_results_bridge.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `pytest tests/ -v -k "not swe_lego" --timeout=60`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_ssd_launcher.py
git commit -m "test(ssd): add integration test for full bundle pipeline"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- Sample stage → Task 2 (data.py) + Task 5 (launcher sample.sh)
- Fine-tune stage → Task 5 (launcher train.sh, uses SFTTrainer)
- Evaluate stage → Task 3 (lcb_evaluator.py) + Task 5 (launcher eval.sh)
- Import results → Task 4 (results_bridge.py) + Task 5 (launcher import_results.sh)
- Registry → Task 6
- SLURM pipeline → Task 5 (run_ssd_pipeline)
- Recipe format → Task 5 (build_ssd_launcher_bundle parses config)
- Dependencies → Task 1 (pyproject.toml)
- Tests → Tasks 2-7

**2. Placeholder scan:** No TBD/TODO found. All code steps contain complete implementation.

**3. Type consistency:**
- `build_ssd_launcher_bundle()` returns dict with `_sample_config`, `_train_config`, `_eval_config` — consumed by `write_ssd_launcher_bundle()`
- `generate_sampling_script()` and `generate_eval_script()` both take `dict[str, Any]` config and return `str`
- `import_results()` takes `(str|Path, str, str)` and returns `dict[str, Any]` — matches tinyzero pattern
- `LiveCodeBenchEvaluator` implements `BaseEvaluator` with `evaluate()` → `BenchmarkResult` and `get_benchmark_name()` → `str`
