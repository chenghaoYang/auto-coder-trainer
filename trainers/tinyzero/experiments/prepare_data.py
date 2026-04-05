"""Prepare all datasets for TinyZero experiments.

Datasets produced:
- GSM8K full + subsets (100,500,1K,2K,5K for scaling experiments)
- GSM8K 1-shot subsets (1,2,5,10,50,100 for exp13)
- Countdown (synthetic, for cross-task transfer exp4)
- MATH (for Pass@k evaluation exp12)
- GSM8K multi-turn (for experiments 17, 20)

Output format: verl-compatible parquet with fields:
  data_source, prompt, ability, reward_model, extra_info
"""

from __future__ import annotations

import argparse
import os
import re
import random
from typing import Any

import datasets


def extract_gsm8k_answer(answer_str: str) -> str | None:
    match = re.search(r"#### (\-?[0-9\.\,]+)", answer_str)
    if match is None:
        return None
    return match.group(1).replace(",", "")


def prepare_gsm8k(output_dir: str) -> None:
    """Prepare full GSM8K + subsets."""
    print("[1/4] Preparing GSM8K dataset...")
    ds = datasets.load_dataset("openai/gsm8k", "main")

    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def process(example: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
        question = example["question"] + " " + instruction
        answer = extract_gsm8k_answer(example["answer"])
        return {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": example["answer"],
                "question": example["question"],
            },
        }

    train = [process(ex, i, "train") for i, ex in enumerate(ds["train"])]
    test = [process(ex, i, "test") for i, ex in enumerate(ds["test"])]

    datasets.Dataset.from_list(train).to_parquet(
        os.path.join(output_dir, "gsm8k_train.parquet")
    )
    datasets.Dataset.from_list(test).to_parquet(
        os.path.join(output_dir, "gsm8k_test.parquet")
    )

    # Subsets for scaling / 1-shot experiments
    random.seed(42)
    for size in [1, 2, 5, 10, 50, 100, 500, 1000, 2000, 5000]:
        if size > len(train):
            continue
        subset = random.sample(train, size)
        datasets.Dataset.from_list(subset).to_parquet(
            os.path.join(output_dir, f"gsm8k_train_{size}.parquet")
        )

    print(f"  GSM8K: {len(train)} train, {len(test)} test + 10 subsets")


def prepare_countdown(output_dir: str, n_train: int = 5000, n_test: int = 500) -> None:
    """Generate synthetic Countdown dataset for cross-task transfer (exp4)."""
    print("[2/4] Preparing Countdown dataset...")
    random.seed(42)
    ops = ["+", "-", "*"]

    def generate_problem() -> dict[str, Any] | None:
        n_nums = random.randint(2, 4)
        numbers = [random.randint(1, 50) for _ in range(n_nums)]
        expr_nums = list(numbers)
        random.shuffle(expr_nums)
        expr = str(expr_nums[0])
        for i in range(1, len(expr_nums)):
            op = random.choice(ops)
            expr = f"({expr} {op} {expr_nums[i]})"
        try:
            target = eval(expr)
            if not isinstance(target, (int, float)) or target < 0 or target > 1000:
                return None
            target = int(target)
        except Exception:
            return None
        return {"numbers": numbers, "target": target, "solution": expr}

    data: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    while len(data) < n_train + n_test:
        prob = generate_problem()
        if prob is None:
            continue
        key = (tuple(sorted(prob["numbers"])), prob["target"])
        if key in seen:
            continue
        seen.add(key)
        data.append(prob)

    random.shuffle(data)
    train_raw, test_raw = data[:n_train], data[n_train : n_train + n_test]

    instruction = (
        "Use the given numbers and basic operations (+, -, *) to reach the target number. "
        'Show your work step by step and output the final answer after "####".'
    )

    def fmt(prob: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
        question = f"Numbers: {prob['numbers']}\nTarget: {prob['target']}\n{instruction}"
        return {
            "data_source": "countdown",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(prob["target"])},
            "extra_info": {
                "split": split,
                "index": idx,
                "numbers": prob["numbers"],
                "target": prob["target"],
                "solution": prob["solution"],
            },
        }

    train = [fmt(p, i, "train") for i, p in enumerate(train_raw)]
    test = [fmt(p, i, "test") for i, p in enumerate(test_raw)]

    datasets.Dataset.from_list(train).to_parquet(
        os.path.join(output_dir, "countdown_train.parquet")
    )
    datasets.Dataset.from_list(test).to_parquet(
        os.path.join(output_dir, "countdown_test.parquet")
    )
    print(f"  Countdown: {len(train)} train, {len(test)} test")


def prepare_math(output_dir: str) -> None:
    """Prepare MATH dataset for Pass@k evaluation (exp12)."""
    print("[3/4] Preparing MATH dataset...")
    try:
        ds = datasets.load_dataset(
            "hendrycks/competition_math", trust_remote_code=True
        )
    except Exception:
        print("  MATH dataset not available, skipping.")
        return

    instruction = 'Solve the problem step by step. Put your final answer after "####".'

    def extract_math_answer(solution: str) -> str:
        match = re.search(r"\\boxed\{([^}]+)\}", solution)
        if match:
            return match.group(1)
        return solution.strip().split("\n")[-1]

    def process(example: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
        question = example.get("problem", "") + " " + instruction
        solution = example.get("solution", "")
        answer = extract_math_answer(solution)
        return {
            "data_source": "math",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "level": example.get("level", ""),
                "type": example.get("type", ""),
            },
        }

    for split_name in ["train", "test"]:
        if split_name in ds:
            records = [process(ex, i, split_name) for i, ex in enumerate(ds[split_name])]
            datasets.Dataset.from_list(records).to_parquet(
                os.path.join(output_dir, f"math_{split_name}.parquet")
            )
            print(f"  MATH {split_name}: {len(records)} samples")


def prepare_gsm8k_multiturn(output_dir: str) -> None:
    """Prepare multi-turn GSM8K for experiments 17/20."""
    print("[4/4] Preparing GSM8K multi-turn dataset...")
    ds = datasets.load_dataset("openai/gsm8k", "main")

    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def create_multiturn(example: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
        answer = extract_gsm8k_answer(example["answer"])
        return {
            "data_source": "openai/gsm8k",
            "prompt": [
                {
                    "role": "user",
                    "content": "I'm going to ask you a math problem. Think carefully and show your reasoning.",
                },
                {
                    "role": "assistant",
                    "content": "I'm ready! Please give me the math problem and I'll solve it step by step.",
                },
                {
                    "role": "user",
                    "content": example["question"] + " " + instruction,
                },
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "format": "multiturn",
            },
        }

    for split_name, split_data in [("train", ds["train"]), ("test", ds["test"])]:
        records = [create_multiturn(ex, i, split_name) for i, ex in enumerate(split_data)]
        datasets.Dataset.from_list(records).to_parquet(
            os.path.join(output_dir, f"gsm8k_multiturn_{split_name}.parquet")
        )
        print(f"  GSM8K multi-turn {split_name}: {len(records)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for TinyZero experiments")
    parser.add_argument(
        "--output_dir",
        default="/scratch/cy2668/auto-coder-trainer/data/tinyzero",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prepare_gsm8k(args.output_dir)
    prepare_countdown(args.output_dir)
    prepare_math(args.output_dir)
    prepare_gsm8k_multiturn(args.output_dir)

    print(f"\nAll datasets saved to {args.output_dir}")
