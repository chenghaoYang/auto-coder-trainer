"""Data preparation utilities for SWE-Lego.

Downloads datasets from HuggingFace, generates LLaMA-Factory dataset_info.json,
and converts trajectory data to verifier training format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SWE_LEGO_ROOT = Path(__file__).parent / "SWE-Lego"

_HF_DATASETS = {
    "swe_lego_real_data": "SWE-Lego/SWE-Lego-Real-Data",
    "swe_lego_synthetic_data": "SWE-Lego/SWE-Lego-Synthetic-Data",
}

_SHAREGPT_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system",
}


def download_datasets(output_dir: str | Path) -> dict[str, Path]:
    """Download SWE-Lego datasets from HuggingFace.

    Downloads SWE-Lego/SWE-Lego-Real-Data and SWE-Lego/SWE-Lego-Synthetic-Data.
    Saves as JSON in ShareGPT format to *output_dir*.

    Returns dict mapping dataset name to file path.
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    for name, hf_id in _HF_DATASETS.items():
        logger.info("Downloading %s from %s ...", name, hf_id)
        ds = load_dataset(hf_id)

        for split_name, split_data in ds.items():
            file_stem = f"{name}_{split_name}" if split_name != "train" else name
            out_path = output_dir / f"{file_stem}.json"

            records: list[dict[str, Any]] = []
            for row in split_data:
                messages = row.get("messages")
                if messages:
                    records.append({"messages": messages})
                else:
                    # Fallback: wrap raw fields into a single-turn message
                    prompt = row.get("prompt", row.get("instruction", ""))
                    response = row.get("response", row.get("output", ""))
                    if prompt or response:
                        msgs: list[dict[str, str]] = []
                        if prompt:
                            msgs.append({"role": "user", "content": str(prompt)})
                        if response:
                            msgs.append({"role": "assistant", "content": str(response)})
                        records.append({"messages": msgs})

            out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
            logger.info("  Wrote %d samples to %s", len(records), out_path)
            result[file_stem] = out_path

    return result


def generate_dataset_info(
    data_dir: str | Path,
    llama_factory_data_dir: str | Path,
) -> Path:
    """Generate/update dataset_info.json for LLaMA-Factory.

    Scans *data_dir* for ``*.json`` files and creates ShareGPT-formatted entries.
    Writes the result to ``<llama_factory_data_dir>/dataset_info.json``, merging
    with any existing entries.

    Returns path to generated dataset_info.json.
    """
    data_dir = Path(data_dir)
    lf_data_dir = Path(llama_factory_data_dir)
    info_path = lf_data_dir / "dataset_info.json"

    # Load existing entries if present
    existing: dict[str, Any] = {}
    if info_path.exists():
        existing = json.loads(info_path.read_text())

    for json_file in sorted(data_dir.glob("*.json")):
        entry_name = json_file.stem
        existing[entry_name] = {
            "file_name": json_file.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": dict(_SHAREGPT_TAGS),
        }
        logger.info("Added dataset entry: %s -> %s", entry_name, json_file.name)

    info_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
    logger.info("Wrote dataset_info.json to %s", info_path)
    return info_path


def convert_to_verifier_format(
    trajectories_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert trajectory data to verifier training format.

    Each output sample contains a system prompt explaining the evaluation task,
    followed by the full interaction trajectory and patch, with a YES/NO label.

    Returns path to output file.
    """
    trajectories_path = Path(trajectories_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    SYSTEM_PROMPT = (
        "You are an expert judge evaluating AI assistant interactions. "
        "Your task is to determine if the assistant successfully resolved "
        "the user's request.\n\n"
        "Key evaluation criteria:\n"
        "1. Did the assistant complete the main task requested by the user?\n"
        "2. Did the assistant handle all edge cases and requirements specified?\n"
        "3. Were there any errors or issues in the final solution?\n"
        "4. Did the assistant verify the solution works as intended?\n\n"
        'Respond only with "<judgement>YES</judgement>" or "<judgement>NO</judgement>".'
    )

    raw_text = trajectories_path.read_text(encoding="utf-8").strip()
    if trajectories_path.suffix == ".jsonl":
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw_text)
        records = parsed if isinstance(parsed, list) else [parsed]

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in records:
            if not isinstance(item, dict):
                continue
            for run_rec in _iter_runs(item):
                interaction = _build_interaction_log(run_rec["messages"], run_rec["patch"])
                label = "YES" if run_rec["score"] else "NO"
                out = {
                    "instance_id": run_rec["instance_id"],
                    "run": run_rec["run"],
                    "score": run_rec["score"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": interaction},
                        {"role": "assistant", "content": f"<judgement>{label}</judgement>"},
                    ],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1

    logger.info("Wrote %d verifier samples to %s", count, output_path)
    return output_path


def _iter_runs(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield per-run records from an instance-level trajectory dict."""
    instance_id = item.get("instance_id", "unknown")

    run_keys = sorted(
        (k for k in item if k.startswith("run_") and isinstance(item[k], dict)),
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x,
    )
    if run_keys:
        results = []
        for rk in run_keys:
            rd = item[rk]
            messages = rd.get("funccalloff_messages") or rd.get("messages") or []
            if not messages:
                continue
            patch = rd.get("patch") or rd.get("model_patch") or ""
            score = _score_from_fields(rd)
            results.append({"instance_id": instance_id, "run": rk, "score": score, "messages": messages, "patch": patch})
        return results

    messages = item.get("funccalloff_messages") or item.get("messages") or []
    if messages:
        patch = item.get("patch") or item.get("model_patch") or ""
        return [{
            "instance_id": instance_id,
            "run": item.get("run") or item.get("run_id") or "run_1",
            "score": _score_from_fields(item),
            "messages": messages,
            "patch": patch,
        }]
    return []


def _score_from_fields(obj: dict[str, Any]) -> int:
    if "score" in obj and obj["score"] is not None:
        return int(float(obj["score"]) > 0.5)
    if "resolved" in obj and obj["resolved"] is not None:
        return 1 if bool(obj["resolved"]) else 0
    return 0


def _build_interaction_log(messages: list[dict[str, Any]], patch: str) -> str:
    lines = [
        "Please evaluate the following interaction between an AI assistant and a user:",
        "",
        "=== INTERACTION LOG ===",
        "",
    ]
    for msg in messages:
        role = str(msg.get("role", "")).upper()
        content = str(msg.get("content", ""))
        lines.extend([f"[{role}]", content, f"[/{role}]", ""])

    lines.extend([
        "=== END INTERACTION ===",
        "",
        "=== FINAL PATCH ===",
        "",
        "[PATCH]",
        str(patch) if patch else "No patch generated",
        "[/PATCH]",
        "",
        "=== END FINAL PATCH ===",
        "",
        "Based on the above interaction, did the assistant successfully resolve "
        "the user's initial request? Respond with YES or NO.",
    ])
    return "\n".join(lines)
