# SSD Trainer Design Spec

> Simple Self-Distillation (SSD) trainer integration for auto-coder-trainer.
> Based on Apple's [ml-ssd](https://github.com/apple/ml-ssd) repository and the paper
> *"Embarrassingly Simple Self-Distillation Improves Code Generation"* (arXiv: 2604.01193).

## Overview

Integrate the SSD method as a new `trainers/ssd/` module following the launcher bundle pattern
(consistent with `trainers/swe_lego/` and `trainers/tinyzero/`). The SSD method consists of
three steps — **Sample**, **Fine-tune**, **Decode+Evaluate** — running as a SLURM dependency chain.

**No rewards, no verifier, no teacher, no RL.** Standard cross-entropy on raw sampled outputs.

## Architecture

### SLURM Dependency Chain

```
sample.sh (vLLM offline inference, N GPU)
    │ afterok
    ▼
train.sh  (SFTTrainer fine-tuning, M GPU)
    │ afterok
    ▼
eval.sh   (LCB v6 evaluation, K GPU)
    │ afterok
    ▼
import_results.sh (results → SQLite DB)
```

### File Structure

```
trainers/ssd/
├── __init__.py              # Export SSDLauncher
├── launcher.py              # Bundle generation + SLURM submission
├── data.py                  # vLLM sampling logic → JSONL
├── lcb_evaluator.py         # LiveCodeBench v6 evaluator (from ml-ssd)
└── results_bridge.py        # Eval results → DB import

tests/
├── test_ssd_launcher.py       # Bundle generation tests (dry-run)
├── test_ssd_data.py           # Sampling data format validation
├── test_lcb_evaluator.py      # Evaluator unit tests (mock vLLM)
└── test_ssd_results_bridge.py # Results import tests (temp DB)
```

### Module Responsibilities

**launcher.py** — `SSDLauncher` class
- `generate_bundle()` → creates 4 shell scripts + config files in `outputs/<recipe_id>/ssd/`
- `submit()` → submits via `trainers/slurm/submitter.py` with SLURM dependency chain
- `get_status()` → queries SLURM job status
- Generates: `sample_config.json`, `train_config.yaml`, `eval_config.json`

**data.py** — Sampling logic
- `run_sampling()` — loads vLLM LLM, generates code solutions from LiveCodeBench dataset
- Output: JSONL, each line `{ "messages": [...], "metadata": {...} }`
- Supports multi-temperature, n_repeat, seed control

**lcb_evaluator.py** — Evaluation
- Ported from ml-ssd `benchmark.py` + `livecodebench_utils.py`
- Adapts to `BaseEvaluator` interface (`evaluate()`, `get_benchmark_name()`)
- Supports pass@1/5/10/16/20/32 metrics
- Includes code sandbox, stdin/stdout mock, Decimal precision comparison

**results_bridge.py** — Results bridge
- Parses eval.sh JSON output
- Writes to SQLite DB via `results/db.py`

## Recipe Format

```json
{
  "id": "recipe-ssd-001",
  "model": {
    "base": "Qwen/Qwen3-Coder",
    "adapter": "full"
  },
  "trainer": {
    "type": "ssd",
    "backend": "ssd",
    "params": {
      "sample_temperature": 0.9,
      "sample_top_p": 0.8,
      "sample_top_k": 20,
      "n_samples_per_problem": 10,
      "max_tokens": 65536,
      "dataset": "livecodebench/code_generation_lite",
      "dataset_filter": {
        "contest_date": ["2025-02", "2025-03", "2025-04", "2025-05"]
      },
      "epochs": 1,
      "lr": 2e-5,
      "batch_size": 1,
      "decode_temperature": 0.6,
      "n_repeat": 20,
      "eval_max_tokens": 32768
    }
  },
  "budget": {
    "slurm": {
      "sample": { "gpus": 4, "time": "04:00:00" },
      "train":  { "gpus": 8, "time": "08:00:00" },
      "eval":   { "gpus": 4, "time": "04:00:00" }
    }
  }
}
```

## Data Flow

### Step 1: sample.sh
- **Input**: recipe config + HF dataset (LiveCodeBench v6)
- **Process**: vLLM offline inference → generate code solutions
- **Output**: `outputs/<id>/ssd/sample_data.jsonl`
  ```jsonl
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {...}}
  ```

### Step 2: train.sh
- **Input**: `sample_data.jsonl` + recipe config
- **Process**: Load JSONL → Dataset → SFTTrainer fine-tune (standard cross-entropy)
- **Output**: `outputs/<id>/ssd/model/` (HuggingFace checkpoint)

### Step 3: eval.sh
- **Input**: model checkpoint + eval config
- **Process**: vLLM loads model → LCB v6 benchmark → pass@k computation
- **Output**: `outputs/<id>/ssd/eval_results.json`
  ```json
  {
    "pass@1": 0.xx,
    "pass@5": 0.xx,
    "pass@1_easy": 0.xx,
    "detail": {...}
  }
  ```

### Step 4: import_results.sh
- **Input**: `eval_results.json`
- **Process**: Parse → write to SQLite DB
- **Output**: DB records in `experiments` + `eval_runs` tables

## Key Interfaces

### Registry Registration

```python
# trainers/registry.py — add to _register_builtins()
from trainers.ssd.launcher import SSDLauncher
register("ssd", None, SSDLauncher)
register("ssd", "ssd", SSDLauncher)
```

### SSDLauncher Interface

```python
class SSDLauncher(BaseTrainer):
    def __init__(self, config: dict, output_dir: str): ...
    def generate_bundle(self) -> str:        # Returns bundle directory path
    def submit(self) -> dict:                # Returns SLURM job IDs
    def get_status(self) -> dict:            # Query SLURM status
    # Implements BaseTrainer: prepare_data(), train(), evaluate()
```

### LiveCodeBenchEvaluator Interface

```python
class LiveCodeBenchEvaluator(BaseEvaluator):
    def evaluate(self, model_path: str, seed: int = 42) -> BenchmarkResult: ...
    def get_benchmark_name(self) -> str:
        return "livecodebench-v6"
```

## Code Migration Map (ml-ssd → auto-coder-trainer)

| ml-ssd source | Target | Changes |
|---|---|---|
| `eval.py` (CLI + sampling) | `data.py` | Remove argparse, convert to function calls |
| `benchmark.py` (LiveCodeBenchV6) | `lcb_evaluator.py` | Adapt to BaseEvaluator interface |
| `livecodebench_utils.py` (utils) | `lcb_evaluator.py` | Port as-is with Apple license headers |

## Error Handling

- **sample.sh fails**: Empty output or malformed JSONL → train.sh not submitted (SLURM dependency)
- **train.sh fails**: OOM or training exception → eval.sh not submitted
- **eval.sh timeout**: Per-test-case 6s timeout, total `(timeout+1)*n_cases+5` seconds (matches ml-ssd)
- **Results import**: Missing fields → skip with warning log

## Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
ssd = ["vllm>=0.11.0", "scipy>=1.17.0", "sentencepiece>=0.2.0"]
```

Install: `pip install -e ".[ssd]"`

## Boundaries

### In Scope
- Complete SSD pipeline: Sample → Fine-tune → Evaluate → Import results
- LiveCodeBench v6 evaluation with pass@k metrics
- SLURM launcher bundle with dependency chain
- Registry integration for `act train` workflow

### Out of Scope
- Model checkpoint download/management (handled by HuggingFace Hub)
- Custom reward model or RL training (not needed for SSD)
- `uv` package manager integration (use project's existing pip)
- Other benchmarks beyond LCB v6 (can be added later via evaluators/)

## Testing Strategy

- `test_ssd_launcher.py`: Verify bundle generation (shell scripts, configs, dependency chain) without SLURM submission
- `test_ssd_data.py`: Validate sampling data format (JSONL structure, message format)
- `test_lcb_evaluator.py`: Unit test evaluator with mock vLLM model, verify pass@k computation, code sandbox
- `test_ssd_results_bridge.py`: Test results import with temporary DB
- All tests independent of external services (per CLAUDE.md constraint)
