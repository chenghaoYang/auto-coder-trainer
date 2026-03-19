# Open-R1 Framework Research Notes

> Research for building an automated training system for small models (7B/4B) on SWE coding trajectories using HuggingFace's open-r1 framework.

## 1. Project Overview

**Open-R1** ([github.com/huggingface/open-r1](https://github.com/huggingface/open-r1)) is a community-driven initiative to reproduce DeepSeek-R1. It provides a complete training pipeline for reasoning models.

**Key value for our use case**: Open-R1's pipeline (SFT → GRPO RL) can be adapted to train small coding models on SWE-bench trajectories, replacing math reward functions with code execution rewards.

## 2. Project Structure

```
open-r1/
├── src/open_r1/
│   ├── __init__.py
│   ├── configs.py        # Dataclass configs for SFT, GRPO, datasets
│   ├── sft.py            # Supervised fine-tuning script
│   ├── grpo.py           # Group Relative Policy Optimization (RL) script
│   ├── generate.py       # Synthetic data generation via Distilabel
│   ├── rewards.py        # Reward functions registry
│   └── utils/
│       ├── callbacks.py         # Training callbacks
│       ├── code_providers.py    # Code execution providers (e2b, local, morph)
│       ├── data.py              # Dataset loading/processing
│       ├── evaluation.py        # LightEval benchmark integration
│       ├── hub.py               # HuggingFace Hub utilities
│       ├── import_utils.py      # Module import helpers
│       ├── model_utils.py       # Model loading utilities
│       ├── routed_morph.py      # Morph sandbox routing
│       ├── routed_sandbox.py    # Sandboxed code execution
│       ├── wandb_logging.py     # W&B experiment tracking
│       └── competitive_programming/  # CP evaluation utilities
├── recipes/                     # YAML configs per model
│   ├── accelerate_configs/      # DeepSpeed ZeRO configs
│   │   └── zero3.yaml
│   ├── dataset_filtering/       # Data preprocessing
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/grpo/
│   ├── Qwen2.5-1.5B-Instruct/grpo/
│   ├── Qwen2.5-Coder-7B-Instruct/grpo/
│   ├── OpenR1-Distill-7B/sft/
│   ├── OlympicCoder-7B/sft/
│   └── OlympicCoder-32B/sft/
├── scripts/                     # Utility scripts
├── slurm/                       # Cluster job templates
├── tests/                       # Test suite
└── Makefile                     # Convenience commands
```

## 3. Training Pipeline

### 3.1 Supervised Fine-Tuning (SFT) — `src/open_r1/sft.py`

**Purpose**: Fine-tune a base/instruct model on reasoning datasets (conversation format).

**Pipeline**:
1. Parse args via `TrlParser` → `ScriptArguments`, `SFTConfig`, `ModelConfig`
2. Load dataset, tokenizer, model (with optional PEFT/LoRA)
3. Apply ChatML template if no chat template exists
4. Initialize `SFTTrainer` from TRL
5. Train with checkpoint resumption support
6. Save model + generation config + model card
7. Optionally evaluate and push to HF Hub

**Key features**:
- Distributed training via Accelerate (DDP / DeepSpeed ZeRO-2/3)
- Gradient checkpointing for memory efficiency
- bf16 precision training
- Optional Liger kernel optimization
- W&B logging integration

**Example SFT config** (`recipes/OpenR1-Distill-7B/sft/config_distill.yaml`):
```yaml
# Model
model_name_or_path: open-r1/Qwen2.5-Math-7B-RoPE-300k
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# Dataset
dataset_name: open-r1/Mixture-of-Thoughts

# Training
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 4.0e-05
lr_scheduler_type: cosine
num_train_epochs: 5
max_seq_length: 32768
gradient_checkpointing: true
use_liger_kernel: true
max_grad_norm: 0.2
warmup_ratio: 0.03
```

**Run command**:
```bash
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
  src/open_r1/sft.py recipes/OpenR1-Distill-7B/sft/config_distill.yaml
```

### 3.2 GRPO (Group Relative Policy Optimization) — `src/open_r1/grpo.py`

**Purpose**: RL-based training using reward functions to improve reasoning quality.

**Pipeline**:
1. Parse args → `GRPOScriptArguments`, `GRPOConfig`, `ModelConfig`
2. Load dataset, convert to conversation format with system prompts
3. Load model, tokenizer, and reward functions from registry
4. Initialize `GRPOTrainer` with reward functions, datasets, PEFT config
5. Train with optional checkpoint resumption
6. Save model with aligned generation config

**Key GRPO parameters** (from `recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml`):
```yaml
# Model
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
torch_dtype: bfloat16
attn_implementation: flash_attention_2

# Dataset
dataset_name: open-r1/OpenR1-Math-220k
system_prompt: "... use <think> tags for reasoning, <answer> tags for answer"

# GRPO-specific
num_generations: 16          # Samples per prompt for group comparison
max_prompt_length: 512
max_completion_length: 1024

# Reward functions (registry-based)
reward_funcs:
  - accuracy
  - format
  - tag_count
reward_weights:
  - 1.0
  - 1.0
  - 1.0

# Training
per_device_train_batch_size: 16
gradient_accumulation_steps: 4
learning_rate: 2.0e-05
lr_scheduler_type: cosine
warmup_ratio: 0.1
num_train_epochs: 1
```

### 3.3 Synthetic Data Generation — `src/open_r1/generate.py`

**Purpose**: Generate synthetic reasoning traces using Distilabel + vLLM.

**Usage**:
```bash
python src/open_r1/generate.py \
  --hf-dataset <input_dataset> \
  --prompt-column prompt \
  --model <model_name> \
  --base-url http://localhost:8000/v1 \
  --num-generations 4 \
  --max-new-tokens 8192 \
  --temperature 0.7 \
  --hf-output-dataset <output_name>
```

## 4. Configuration System

### 4.1 Config Dataclasses (`src/open_r1/configs.py`)

**`DatasetConfig`**: Individual dataset specification
- `dataset_name`, `config_name`, `split`, `columns`, `weight`

**`DatasetMixtureConfig`**: Multi-dataset blending
- List of `DatasetConfig`, seed, optional test split

**`ScriptArguments`**: Extends TRL's base args
- Supports single dataset or dataset mixture
- Validates column consistency across mixed datasets

**`GRPOConfig`** / **`SFTConfig`**: Extend TRL training configs
- Custom benchmarks and callbacks
- Chat templates and system prompts
- W&B logging (entity, project, run group)
- Hub model versioning

**`GRPOScriptArguments`**: Reward configuration
- `reward_funcs`: list of reward function names (from registry)
- `reward_weights`: per-function weights
- Code evaluation settings (language, scoring mode)
- Code execution provider: `"e2b"`, `"local"`, `"morph"`

### 4.2 Recipe YAML Files

Training is configured via YAML files in `recipes/`. These map directly to the dataclass fields. Run with:
```bash
accelerate launch --config_file <accelerate_config> \
  src/open_r1/<script>.py <recipe_yaml>
```

### 4.3 Accelerate Configs

DeepSpeed ZeRO-3 config (`recipes/accelerate_configs/zero3.yaml`):
```yaml
distributed_type: DEEPSPEED
zero_stage: 3
zero3_init_flag: true
zero3_save_16bit_model: true
mixed_precision: bf16
num_processes: 8
num_machines: 1
# No optimizer/parameter offloading
```

## 5. Reward Functions (`src/open_r1/rewards.py`)

| Function | Description | Relevance to SWE |
|----------|-------------|-------------------|
| `accuracy` | Compare output to ground truth (LaTeX parsing) | Need SWE equivalent |
| `format` | Check `<think>`/`<answer>` tag structure | Reusable for CoT format |
| `tag_count` | Count opening/closing tags (0.25 each) | Reusable |
| `reasoning_steps` | Detect "Step 1:", numbered lists | Reusable |
| `length` | Penalize verbose completions (Kimi 1.5 approach) | Useful |
| `cosine_scaled` | Cosine schedule: reward shorter correct answers | Useful |
| `repetition_penalty` | N-gram repetition detection → negative reward | Useful |
| `code` / `ioi_code_reward` / `cf_code_reward` | Execute code, verify output | **Key for SWE** |

**For SWE training**, we'd need custom reward functions:
1. **Test pass rate**: Run generated patches against SWE-bench test suites
2. **Lint/type check**: Verify code quality
3. **Diff quality**: Evaluate patch minimality and correctness
4. **Format compliance**: Ensure proper patch/diff format

## 6. Supported Models and Datasets

### Models with Existing Recipes

| Model | Size | Training | Hardware |
|-------|------|----------|----------|
| Qwen2.5-1.5B-Instruct | 1.5B | GRPO | Single node |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | GRPO | Single node |
| Qwen2.5-Coder-7B-Instruct | 7B | GRPO | Single node |
| OpenR1-Distill-7B | 7B | SFT | Single node, ZeRO-3, 8× H100 |
| OlympicCoder-7B | 7B | SFT | Single node, ZeRO-3 |
| OlympicCoder-32B | 32B | SFT | 16 nodes, FSDP + paged AdamW 8-bit |

### Key Datasets

| Dataset | Size | Domain |
|---------|------|--------|
| Mixture-of-Thoughts | 350k traces | Math, coding, science |
| OpenR1-Math-220k | 220k traces | Math reasoning |
| CodeForces-CoTs | 10k problems / 100k solutions | Competitive programming |

## 7. Evaluation

Uses **LightEval** with vLLM backend. Supported benchmarks:
- **Math**: MATH 500, AIME24, AIME25
- **General**: GPQA Diamond
- **Code**: LCB, LCB v4

**Run evaluation**:
```bash
make evaluate MODEL=<model> TASK=<benchmark> NUM_GPUS=8 PARALLEL=data
```

Evaluation is submitted as SLURM jobs with automatic GPU allocation and tensor parallelism for models ≥30B.

## 8. Key Commands (Makefile)

```bash
make install       # Setup Python 3.11 venv with vllm, flash-attn
make style         # Format code (ruff, isort)
make quality       # Lint checks
make test          # Run pytest (fast tests)
make slow_test     # Run slow test suite
make evaluate      # Run LightEval benchmarks
```

## 9. Dependencies

- Python 3.11, CUDA 12.4
- TRL (Transformer Reinforcement Learning library) — core training framework
- Accelerate — distributed training orchestration
- DeepSpeed — ZeRO optimization
- vLLM — fast inference for generation and evaluation
- FlashAttention 2 — memory-efficient attention
- Distilabel — synthetic data generation pipelines
- LightEval — evaluation framework
- Weights & Biases — experiment tracking
- Optional: Liger kernel for training optimization

**Install**:
```bash
uv venv openr1 --python 3.11 && uv pip install --all-extras .
uv pip install vllm==0.7.2 setuptools flash-attn --no-build-isolation
```

## 10. Adaptation Plan for SWE Coding Trajectories

### Phase 1: SFT on SWE Trajectories
1. **Dataset**: Convert SWE-bench trajectories into conversation format
   - System prompt: coding agent instructions
   - User: issue description + repository context
   - Assistant: reasoning chain + patch
2. **Base model**: Start with Qwen2.5-Coder-7B-Instruct or similar coding model
3. **Config**: Adapt `OpenR1-Distill-7B/sft/config_distill.yaml`
   - Increase `max_seq_length` for long coding contexts
   - Use coding-appropriate chat template

### Phase 2: GRPO with Code Rewards
1. **Reward functions** (custom):
   - `swe_test_pass`: Execute patch, run tests → pass rate
   - `patch_format`: Validate diff/patch format
   - `code_quality`: Lint + type check score
   - `patch_minimality`: Penalize unnecessarily large diffs
2. **Code execution**: Use `code_providers.py` infrastructure (e2b/local/morph)
3. **Config**: Adapt `Qwen2.5-Coder-7B-Instruct/grpo/` recipe
   - Longer `max_completion_length` for code generation
   - Fewer `num_generations` (code execution is expensive)

### Phase 3: Evaluation
1. **SWE-bench Lite/Verified**: Primary benchmark
2. **Custom test suites**: Per-repository evaluation
3. Integrate with LightEval or custom evaluation harness

### Key Considerations
- **Sequence length**: SWE trajectories are long (8K-32K tokens) — need ZeRO-3 + gradient checkpointing
- **Reward computation cost**: Running test suites is slow — batch carefully
- **Dataset quality**: Filter trajectories by success rate, patch quality
- **Small model focus (4B-7B)**: Qwen2.5-Coder-7B, DeepSeek-Coder-V2-Lite, CodeLlama-7B
