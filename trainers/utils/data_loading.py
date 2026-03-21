"""Shared data-loading utilities for SFT and RL trainers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_from_path(path: str) -> list[dict[str, Any]]:
    """Load raw examples from a HuggingFace dataset ID or local JSON/JSONL file."""
    local = Path(path)
    if local.exists() and local.suffix in (".json", ".jsonl"):
        return load_local(local)

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(path, split="train")
        return [dict(row) for row in ds]
    except Exception as exc:
        raise RuntimeError(
            f"Could not load dataset from '{path}'. "
            f"Provide a local JSON/JSONL file or install the datasets package "
            f"for HuggingFace dataset IDs. Original error: {exc}"
        ) from exc


def load_local(path: Path) -> list[dict[str, Any]]:
    """Load examples from a local JSON or JSONL file."""
    with open(path) as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        payload = json.load(f)
    return payload if isinstance(payload, list) else [payload]


def apply_filters(
    examples: list[dict[str, Any]],
    filters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply a chain of standard filters (issue_free, length, quality_score)."""
    filtered = list(examples)
    for fspec in filters:
        filter_type = fspec.get("type")
        params = fspec.get("params", {})
        if filter_type == "issue_free":
            filtered = [
                ex for ex in filtered
                if ex.get("prompt") or ex.get("messages")
            ]
        elif filter_type == "length":
            max_turns = params.get("max_turns", 30)
            max_prompt_chars = params.get("max_prompt_chars")
            next_examples = []
            for ex in filtered:
                turns = ex.get("metadata", {}).get("turns", 0)
                if turns and turns > max_turns:
                    continue
                if max_prompt_chars and len(ex.get("prompt", "")) > max_prompt_chars:
                    continue
                next_examples.append(ex)
            filtered = next_examples
        elif filter_type == "quality_score":
            min_score = params.get("min_score", 0.5)
            filtered = [
                ex for ex in filtered
                if ex.get("metadata", {}).get("quality_score", 1.0) >= min_score
            ]
    return filtered
