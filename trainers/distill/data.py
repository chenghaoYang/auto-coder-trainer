"""Data loading and normalization helpers for trajectory/process distillation."""

from __future__ import annotations

from typing import Any

from trainers.utils.data_loading import apply_filters, load_from_path


def load_distillation_data(
    sources: list[dict[str, Any]],
    filters: list[dict[str, Any]] | None = None,
    *,
    distill_config: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load distillation data from local/HF sources.

    Produces two synchronized views:
    - ``positive_examples`` for teacher-trajectory SFT
    - ``pair_examples`` for optional chosen-vs-rejected refinement
    """
    if not sources:
        raise ValueError("At least one dataset source is required")

    distill_config = distill_config or {}
    positive_examples: list[dict[str, Any]] = []
    pair_examples: list[dict[str, Any]] = []

    for source in sources:
        source_name = source.get("name", "unnamed")
        raw_examples = load_from_path(source.get("path", ""))
        for raw_example in raw_examples:
            normalized = _normalise_distill_example(raw_example, source_name=source_name)
            if normalized is None:
                continue
            positive = _to_positive_example(normalized, distill_config)
            if positive is not None:
                positive_examples.append(positive)
            pair = _to_pair_example(normalized, distill_config)
            if pair is not None:
                pair_examples.append(pair)

    if filters:
        positive_examples = apply_filters(positive_examples, filters)
        pair_examples = [
            pair for pair in pair_examples
            if _pair_matches_filters(pair, filters)
        ]

    return {
        "positive_examples": positive_examples,
        "pair_examples": pair_examples,
    }


def _pair_matches_filters(pair: dict[str, Any], filters: list[dict[str, Any]]) -> bool:
    pseudo_example = {
        "prompt": pair.get("prompt", ""),
        "messages": pair.get("prompt_messages", []),
        "metadata": pair.get("metadata", {}),
    }
    return bool(apply_filters([pseudo_example], filters))


def _normalise_distill_example(example: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    prompt = _first_text(
        example.get("prompt"),
        example.get("instruction"),
        example.get("problem_statement"),
        example.get("input"),
        example.get("query"),
        example.get("task"),
    )
    prompt_messages = _as_messages(example.get("prompt_messages"))
    chosen_messages = _as_messages(
        example.get("chosen_messages")
        or example.get("preferred_messages")
        or example.get("teacher_messages")
    )
    rejected_messages = _as_messages(
        example.get("rejected_messages")
        or example.get("dispreferred_messages")
        or example.get("negative_messages")
    )
    messages = _as_messages(example.get("messages"))

    chosen = _first_text(
        example.get("chosen"),
        example.get("preferred"),
        example.get("accepted"),
        example.get("teacher_response"),
        example.get("response"),
        example.get("completion"),
        example.get("output"),
        example.get("answer"),
        example.get("solution"),
    )
    rejected = _first_text(
        example.get("rejected"),
        example.get("dispreferred"),
        example.get("negative_response"),
        example.get("rejected_response"),
        example.get("bad_response"),
        example.get("negative"),
    )
    tests = example.get("tests") or example.get("test_cases") or example.get("test") or []

    if messages and not chosen_messages and not chosen:
        chosen_messages = messages
    if not prompt_messages and prompt:
        prompt_messages = []

    if not prompt and not prompt_messages and chosen_messages:
        prompt_messages, inferred_response = _split_messages(chosen_messages)
        if inferred_response and not chosen:
            chosen = inferred_response
        prompt = _prompt_from_messages(prompt_messages)
    elif chosen_messages and not chosen:
        _, inferred_response = _split_messages(chosen_messages)
        chosen = inferred_response or chosen

    if rejected_messages and not rejected:
        _, inferred_rejected = _split_messages(rejected_messages)
        rejected = inferred_rejected or rejected

    if not prompt and not prompt_messages and not chosen_messages and not chosen:
        return None

    metadata = {
        "source": source_name,
        "instance_id": example.get("instance_id", example.get("id", "")),
        "quality_score": example.get("quality_score", example.get("score", 1.0)),
        "turns": example.get("turns", len(messages or chosen_messages or prompt_messages)),
        "original_fields": list(example.keys()),
    }
    if example.get("teacher_model"):
        metadata["teacher_model"] = example.get("teacher_model")
    if example.get("teacher_score") is not None:
        metadata["teacher_score"] = example.get("teacher_score")
    if example.get("margin") is not None:
        metadata["margin"] = example.get("margin")

    return {
        "prompt": prompt,
        "prompt_messages": prompt_messages,
        "chosen": chosen,
        "chosen_messages": chosen_messages,
        "rejected": rejected,
        "rejected_messages": rejected_messages,
        "tests": tests,
        "metadata": metadata,
    }


def _to_positive_example(example: dict[str, Any], distill_config: dict[str, Any]) -> dict[str, Any] | None:
    chat_template = distill_config.get("trace_template", "chatml")

    if example.get("chosen_messages"):
        prompt_messages, chosen_output = _split_messages(example["chosen_messages"])
        prompt_text = _render_prompt_prefix(
            prompt=example.get("prompt", ""),
            prompt_messages=prompt_messages,
            chat_template=chat_template,
        )
    else:
        chosen_output = str(example.get("chosen", "")).strip()
        prompt_text = _render_prompt_prefix(
            prompt=example.get("prompt", ""),
            prompt_messages=example.get("prompt_messages", []),
            chat_template=chat_template,
        )

    chosen_output = _condense_text(chosen_output, distill_config)
    prompt_text = _condense_text(prompt_text, distill_config)
    if not chosen_output.strip():
        return None

    return {
        "text": prompt_text + chosen_output,
        "prompt": example.get("prompt", ""),
        "prompt_text": prompt_text,
        "response": chosen_output,
        "messages": example.get("chosen_messages", []),
        "tests": example.get("tests", []),
        "metadata": example.get("metadata", {}),
    }


def _to_pair_example(example: dict[str, Any], distill_config: dict[str, Any]) -> dict[str, Any] | None:
    chat_template = distill_config.get("trace_template", "chatml")

    chosen_output = str(example.get("chosen", "")).strip()
    rejected_output = str(example.get("rejected", "")).strip()
    prompt_messages = example.get("prompt_messages", [])

    if example.get("chosen_messages"):
        prompt_messages, inferred = _split_messages(example["chosen_messages"])
        if inferred:
            chosen_output = inferred
    if example.get("rejected_messages"):
        rejected_prompt_messages, inferred = _split_messages(example["rejected_messages"])
        if not prompt_messages:
            prompt_messages = rejected_prompt_messages
        if inferred:
            rejected_output = inferred

    if not chosen_output or not rejected_output:
        return None

    prompt_text = _render_prompt_prefix(
        prompt=example.get("prompt", ""),
        prompt_messages=prompt_messages,
        chat_template=chat_template,
    )
    prompt_text = _condense_text(prompt_text, distill_config)
    chosen_output = _condense_text(chosen_output, distill_config)
    rejected_output = _condense_text(rejected_output, distill_config)

    if chosen_output == rejected_output:
        return None

    return {
        "prompt_raw": example.get("prompt", ""),
        "prompt": prompt_text,
        "prompt_text": prompt_text,
        "prompt_messages": prompt_messages,
        "chosen": chosen_output,
        "rejected": rejected_output,
        "chosen_output": chosen_output,
        "rejected_output": rejected_output,
        "chosen_text": prompt_text + chosen_output,
        "rejected_text": prompt_text + rejected_output,
        "metadata": example.get("metadata", {}),
    }


def _render_prompt_prefix(
    *,
    prompt: str,
    prompt_messages: list[dict[str, Any]],
    chat_template: str,
) -> str:
    if prompt_messages:
        lines: list[str] = []
        for message in prompt_messages:
            role = str(message.get("role", "user")).strip().capitalize()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        rendered = "\n".join(lines)
        if rendered:
            return rendered + "\nAssistant: "
        return "Assistant: "

    prompt = str(prompt).strip()
    if not prompt:
        return "Assistant: "
    if chat_template == "chatml":
        return f"User: {prompt}\nAssistant: "
    return f"{prompt}\nAssistant: "


def _prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "")).lower()
        if role != "assistant":
            content = str(message.get("content", "")).strip()
            if content:
                parts.append(content)
    return "\n".join(parts).strip()


def _split_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if not messages:
        return [], ""
    last = messages[-1]
    if str(last.get("role", "")).lower() == "assistant":
        return messages[:-1], str(last.get("content", "")).strip()
    return messages, ""


def _as_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _condense_text(text: str, distill_config: dict[str, Any]) -> str:
    condense_cfg = distill_config.get("condense", {})
    strategy = condense_cfg.get("strategy", "none")
    if strategy != "edge_preserving":
        return text

    max_chars = int(condense_cfg.get("max_chars", 12000))
    if len(text) <= max_chars:
        return text

    head_chars = int(condense_cfg.get("head_chars", max_chars // 2))
    tail_chars = int(condense_cfg.get("tail_chars", max_chars // 2))
    if head_chars + tail_chars >= len(text):
        return text

    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip()
    return head + "\n[... condensed trajectory middle ...]\n" + tail
