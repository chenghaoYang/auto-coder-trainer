"""SFT data loading and formatting utilities."""

from __future__ import annotations

from typing import Any

from trainers.utils.data_loading import apply_filters, load_from_path


def load_trajectory_data(
    sources: list[dict[str, Any]],
    filters: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load trajectory data from specified sources and apply filters."""
    if not sources:
        raise ValueError("At least one dataset source is required")

    examples: list[dict[str, Any]] = []
    for source in sources:
        source_name = source.get("name", "unnamed")
        raw_examples = load_from_path(source.get("path", ""))
        for example in raw_examples:
            normalized = _normalise_example(example, source_name=source_name)
            if normalized is not None:
                examples.append(normalized)

    if filters:
        examples = apply_filters(examples, filters)

    return examples


def format_for_sft(dataset: list[dict[str, Any]], chat_template: str = "chatml") -> list[dict[str, Any]]:
    """Format trajectory data as SFT training examples with a stable ``text`` field."""
    formatted: list[dict[str, Any]] = []
    for example in dataset:
        prompt = example.get("prompt", "").strip()
        response = example.get("response", "").strip()
        messages = example.get("messages", [])
        tests = example.get("tests", [])

        if messages:
            rendered = _render_messages(messages, chat_template=chat_template)
        else:
            rendered = _render_prompt_response(prompt, response)

        if not rendered.strip():
            continue

        formatted.append(
            {
                "text": rendered,
                "prompt": prompt,
                "response": response,
                "messages": messages,
                "tests": tests,
                "metadata": example.get("metadata", {}),
            }
        )
    return formatted


def _normalise_example(example: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    messages = example.get("messages")
    prompt = (
        example.get("prompt")
        or example.get("instruction")
        or example.get("problem_statement")
        or example.get("input")
        or example.get("query")
        or ""
    )
    response = (
        example.get("response")
        or example.get("completion")
        or example.get("output")
        or example.get("answer")
        or example.get("solution")
        or ""
    )
    tests = example.get("tests") or example.get("test_cases") or example.get("test") or []

    if not messages and not (prompt or response):
        return None

    metadata = {
        "source": source_name,
        "instance_id": example.get("instance_id", example.get("id", "")),
        "quality_score": example.get("quality_score", example.get("score", 1.0)),
        "turns": example.get("turns", 0),
        "original_fields": list(example.keys()),
    }
    return {
        "prompt": prompt,
        "response": response,
        "messages": messages if isinstance(messages, list) else [],
        "tests": tests,
        "metadata": metadata,
    }


def _render_prompt_response(prompt: str, response: str) -> str:
    if prompt and response:
        return f"User: {prompt}\nAssistant: {response}"
    return prompt or response


def _render_messages(messages: list[dict[str, Any]], chat_template: str) -> str:
    lines: list[str] = []
    last_role = ""
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
            last_role = role.lower()
    rendered = "\n".join(lines)
    if chat_template == "chatml" and rendered and last_role != "assistant":
        return rendered + "\nAssistant:"
    return rendered
