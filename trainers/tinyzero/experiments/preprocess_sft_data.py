#!/usr/bin/env python3
"""
Preprocess GSM8K parquet for SFT training (exp14).

The original gsm8k_train.parquet has:
  - prompt: numpy array of [{role, content}] dicts (chat format)
  - extra_info: dict with 'answer' key

The SFT trainer expects:
  - prompt: plain string
  - answer: plain string (response)

This script flattens the data and saves as gsm8k_sft_train.parquet.
"""
import pandas as pd
import numpy as np
import sys
import os

def preprocess(input_path, output_path):
    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    prompts = []
    answers = []

    for i, row in df.iterrows():
        # Extract prompt text from chat format
        prompt_val = row['prompt']
        if isinstance(prompt_val, np.ndarray):
            prompt_val = prompt_val.tolist()
        if isinstance(prompt_val, list) and len(prompt_val) > 0:
            # Chat format: [{'role': 'user', 'content': '...'}]
            entry = prompt_val[0] if isinstance(prompt_val[0], dict) else prompt_val[0]
            if isinstance(entry, dict):
                prompt_text = entry.get('content', str(entry))
            else:
                prompt_text = str(entry)
        elif isinstance(prompt_val, str):
            prompt_text = prompt_val
        else:
            prompt_text = str(prompt_val)

        # Extract answer from extra_info
        extra = row.get('extra_info', {})
        if isinstance(extra, dict):
            answer_text = extra.get('answer', '')
        elif isinstance(extra, str):
            answer_text = extra
        else:
            answer_text = str(extra)

        prompts.append(prompt_text)
        answers.append(answer_text)

    out_df = pd.DataFrame({
        'prompt': prompts,
        'answer': answers,
    })

    print(f"Writing: {output_path}")
    print(f"  Rows: {len(out_df)}, Columns: {list(out_df.columns)}")
    print(f"  Sample prompt: {prompts[0][:100]}...")
    print(f"  Sample answer: {answers[0][:100]}...")

    out_df.to_parquet(output_path, index=False)
    print(f"Done! Saved {len(out_df)} rows to {output_path}")

if __name__ == '__main__':
    workdir = os.environ.get('WORKDIR', '/scratch/cy2668/auto-coder-trainer')
    input_path = os.path.join(workdir, 'data/tinyzero/gsm8k_train.parquet')
    output_path = os.path.join(workdir, 'data/tinyzero/gsm8k_sft_train.parquet')
    preprocess(input_path, output_path)
