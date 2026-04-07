"""Tests for LiveCodeBench v6 evaluator ported from ml-ssd."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("numpy", reason="numpy required for LCB evaluator tests")


def test_has_code_extracts_python_blocks():
    from trainers.ssd.lcb_evaluator import has_code
    response = "Here is my solution:\n```python\nprint('hello')\n```\nDone."
    matches = has_code(response)
    assert len(matches) == 1
    assert "print('hello')" in matches[0]


def test_has_code_returns_empty_for_no_code():
    from trainers.ssd.lcb_evaluator import has_code
    assert has_code("Just text, no code here.") == []


def test_post_process_code_strips_markdown():
    from trainers.ssd.lcb_evaluator import post_process_code
    code = "```python\nx = 1\nprint(x)\n```"
    result = post_process_code(code)
    assert "```" not in result
    assert "x = 1" in result


def test_compare_strings_exact_match():
    from trainers.ssd.lcb_evaluator import compare_strings_with_decimal_fallback
    assert compare_strings_with_decimal_fallback("hello\nworld", "hello\nworld") is True


def test_compare_strings_numeric_match():
    from trainers.ssd.lcb_evaluator import compare_strings_with_decimal_fallback
    assert compare_strings_with_decimal_fallback("1.0 2.0", "1.0 2.0") is True
    assert compare_strings_with_decimal_fallback("0.5", "0.50") is True


def test_compare_strings_mismatch():
    from trainers.ssd.lcb_evaluator import compare_strings_with_decimal_fallback
    assert compare_strings_with_decimal_fallback("1 2 3", "1 2 4") is False


def test_estimate_pass_at_k():
    import numpy as np
    from trainers.ssd.lcb_evaluator import estimate_pass_at_k
    # estimate_pass_at_k takes (n_array, c_array, k) — vectorized API from ml-ssd
    assert estimate_pass_at_k(np.array([5]), np.array([5]), 1)[0] == 1.0
    assert estimate_pass_at_k(np.array([5]), np.array([0]), 1)[0] == 0.0
    result = estimate_pass_at_k(np.array([10]), np.array([5]), 1)[0]
    assert result == pytest.approx(0.5, abs=0.01)


def test_compute_metrics_from_results():
    from trainers.ssd.lcb_evaluator import compute_metrics_from_results
    results = {
        "task-1": [[1, 1, 1], [1, 1, 0]],
        "task-2": [[1, 0, 0], [0, 0, 0]],
    }
    metrics = compute_metrics_from_results(results, k_list=[1])
    assert "pass@1" in metrics
    assert 0.0 <= metrics["pass@1"] <= 1.0


def test_lcb_evaluator_get_benchmark_name():
    from trainers.ssd.lcb_evaluator import LiveCodeBenchEvaluator
    evaluator = LiveCodeBenchEvaluator(config={})
    assert evaluator.get_benchmark_name() == "livecodebench-v6"


def test_generate_eval_script_creates_valid_python(tmp_path: Path):
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
