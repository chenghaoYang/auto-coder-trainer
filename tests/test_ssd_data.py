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
