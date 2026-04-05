# Auto-Coder-Trainer — Claude Code Operations Guide

Research Operating System for Coding Agent Training.
Closed loop: Papers → Recipes → Training → Evaluation → Judge → Report.

## Deployment Model

This project runs across **two environments**:

```
┌─ Local (macOS) ──────────────────────────────────┐
│  VS Code + Claude Code extension                  │
│  Role: conversation layer, code editing,          │
│        recipe authoring, result analysis           │
│  This repo lives here as the source of truth      │
└──────────────┬───────────────────────────────────┘
               │  VS Code Remote-SSH
               ▼
┌─ Remote: ssh torch ──────────────────────────────┐
│  GPU cluster with SLURM                           │
│  Role: actual training, inference, evaluation     │
│  Has: CUDA, SLURM (sbatch/squeue/sacct),         │
│       LLaMA-Factory, vLLM, SWE-bench harness     │
│  Project synced via VS Code Remote-SSH            │
└──────────────────────────────────────────────────┘
```

**How it works**: VS Code Remote-SSH connects to `torch`. Claude Code runs inside that remote context — all Bash commands, `act` CLI calls, and file edits execute **on the remote server**. The local machine is only the UI/conversation layer.

**Key implication**: when Claude Code runs `act train`, `sbatch`, `squeue`, or reads SLURM logs, these commands run on `torch` directly. No SSH wrapping needed — VS Code Remote-SSH handles the transport.

## First Session Checklist

> 每个新会话的 agent 应在开始时执行以下步骤：
1. 读取本文件 (CLAUDE.md)
2. 读取 PROGRESS.md 了解当前进度
3. 运行 `act status --open-only` 查看待办任务
4. 运行 `make check` 确认代码库状态健康

## Quick Reference

```bash
# Install (on remote: ssh torch)
pip install -e ".[all,dev]"

# Core commands (all execute on remote via VS Code Remote-SSH)
act train recipes/examples/baseline-sft.recipe.json --dry-run    # Validate recipe
act train recipes/examples/baseline-sft.recipe.json               # Submit training
act status --recipe-id <id> --slurm                               # Live SLURM + DB status
act sync --recipe-id <id>                                         # Import completed results
act status --open-only                                            # Show pending tasks
act rerun --recipe-id <id>                                        # Dispatch pending tasks
act report --recipe-id <id> --format blog                         # Generate report
act pipeline --recipe recipe.json --max-iterations 3              # Full auto loop
```

## Project Architecture

```
Recipe JSON (bridge between research & training)
     │
     ├── recipes/compiler.py        → TrainingConfig
     ├── trainers/registry.py       → dispatch to backend
     │     ├── trainers/sft/        → TRL (native)
     │     ├── trainers/rl/         → veRL/GRPO (native)
     │     ├── trainers/distill/    → TRL distillation (native)
     │     ├── trainers/swe_lego/   → LLaMA-Factory (launcher bundle)
     │     ├── trainers/tinyzero/   → TinyZero (launcher bundle)
     │     └── trainers/upstream/   → Open-R1, Agent-Distill, REDI (launcher)
     ├── evaluators/swe_bench.py    → SWE-bench harness (subprocess)
     ├── judge/judge.py             → 5-check verdict system
     └── results/db.py              → SQLite (data/results.db)
```

**Native backends** (trl, verl): train directly in-process.
**Launcher backends** (swe_lego, tinyzero, openr1, etc.): generate shell scripts + configs → submit via SLURM.

## Hard Constraints

> 以下规则不可违反，任何修改都必须遵守：

1. **所有 CLI 命令通过 `act` 入口**：不直接调用 trainers/ 下的模块
2. **Recipe 变更需通过 schema 验证**：修改 recipe 后必须跑 `make validate-schema`
3. **DB 操作通过 results/db.py**：不直接写 SQL 操作 results.db
4. **本地环境不跑 GPU 训练**：训练命令只在远程 torch 上执行
5. **新增 trainer 必须注册到 registry**：`trainers/registry.py` 是唯一调度入口
6. **测试不能依赖外部服务**：需要 SWE-bench/vLLM 的测试用 `pytest.importorskip` 跳过

## Recipe Format

Recipes live in `recipes/` and follow `recipes/schema/recipe.schema.json`.

Key fields:
- `id`: unique identifier (e.g., `recipe-baseline-sft-001`)
- `model.base`: HuggingFace model ID
- `model.adapter`: `full` | `lora` | `qlora`
- `dataset.sources[].path`: HF dataset or local path
- `trainer.type`: `sft` | `rl` | `grpo` | `dpo` | `distill`
- `trainer.backend`: `trl` | `verl` | `swe_lego` | `tinyzero` | `openr1` | `agent_distill` | `redi`
- `trainer.params`: hyperparameters (lr, epochs, batch_size, etc.)
- `eval.benchmarks`: `swe-bench-lite` | `swe-bench-verified` | `humaneval` | `mbpp`
- `eval.seeds`: default `[42, 123, 456]`
- `budget.slurm`: SLURM resource config (partition, gpus, mem, time)

See `recipes/examples/baseline-sft.recipe.json` for a complete example.

## SWE-Lego Workflow (Most Common)

This is the primary training backend. It generates a bundle of scripts:

```bash
# 1. Generate bundle (dry-run)
act train recipe.json --dry-run --no-submit

# 2. Bundle output at: outputs/<recipe-id>/swe_lego/
#    run.sh              → LLaMA-Factory training
#    serve_and_infer.sh  → vLLM server + OpenHands inference
#    eval.sh             → SWE-bench evaluation
#    verifier_train.sh   → Verifier RLHF training
#    tts.sh              → Test-Time Scaling
#    import_results.sh   → Sync results back to DB

# 3. Submit to SLURM (with dependency chain)
act train recipe.json
# train → infer(afterok:train) → eval(afterok:infer) → import_results(afterok:eval)
#   └── verifier_train(afterok:train) → tts(afterok:eval,verifier_train)
```

```bash
# 4. Check SLURM job status (polls sacct live)
act status --recipe-id <id> --slurm

# 5. When jobs finish, sync results → DB → judge → report
act sync --recipe-id <id>
```

Key generated configs:
- `train_config.yaml`: LLaMA-Factory YAML
- `env.sh`: environment variables
- `launcher.json`: bundle manifest

## SLURM Operations

SLURM commands are in `trainers/slurm/submitter.py`:

```bash
# Check job status
squeue -u $USER
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed

# Cancel job
scancel <job_id>

# View logs (after submission)
cat outputs/<recipe-id>/swe_lego/slurm/slurm-<job_id>-train.out
cat outputs/<recipe-id>/swe_lego/slurm/slurm-<job_id>-train.err
```

## Judge System

The judge (`judge/judge.py`) runs 5 checks after training:

1. **Baseline**: no >5% regression vs prior experiments
2. **Seeds**: all seeds ran, CV < 10%
3. **Ablation**: all configured ablation variants exist
4. **Dedup**: no duplicate config hash in DB
5. **Attribution**: failure root-cause analysis

Verdicts: `ACCEPT` | `NEEDS_RERUN` | `NEEDS_ABLATION` | `REJECT`

## Results Database

SQLite at `data/results.db`. Tables: experiments, eval_runs, ablations, verdicts, artifacts, tasks.

```bash
# Quick queries
sqlite3 data/results.db "SELECT id, recipe_id, status, trainer_type FROM experiments"
sqlite3 data/results.db "SELECT * FROM verdicts ORDER BY timestamp DESC LIMIT 5"
sqlite3 data/results.db "SELECT * FROM tasks WHERE status='pending'"
```

## Common Operations for Claude Code

### Modify hyperparameters
Edit the recipe JSON directly. Key params to tune:
- `trainer.params.lr` (learning rate, e.g., 2e-5)
- `trainer.params.batch_size` / `gradient_accumulation_steps`
- `trainer.params.epochs`
- `dataset.total_samples`
- `budget.slurm.gpus_per_node` / `mem` / `time`

### Fix OOM errors
1. Reduce `batch_size` or increase `gradient_accumulation_steps`
2. Switch `model.adapter` from `full` to `lora`
3. Change DeepSpeed stage (z2_offload → z3_offload)
4. Reduce `dataset.total_samples`

### Change model
Update `model.base` in recipe. For Qwen3.5 models, install with:
```bash
pip install -e ".[swe-lego-qwen35]"
```
Qwen3.5 requires transformers>=4.52.0, vllm>=0.17.0.

### Import external results
```bash
act train --import-results /path/to/completed/bundle \
  --recipe-id <id> --experiment-id <id> --report-format blog
```

### Debug failing training
1. Check SLURM logs: `outputs/<recipe-id>/swe_lego/slurm/`
2. Check execution plan: `outputs/<recipe-id>/execution-plan.md`
3. Check task ledger: `outputs/<recipe-id>/task-ledger.md`
4. Query DB for error: `sqlite3 data/results.db "SELECT error FROM experiments WHERE recipe_id='<id>'"`

## Verification Commands

> agent 在完成任何修改后应运行以下命令确认状态：

```bash
# Lint 检查
ruff check .

# 自动格式化检查
ruff format . --check

# 运行全部测试
pytest tests/ -v

# Schema 验证
make validate-schema

# 完整验证（lint + test）
make check
```

## Testing

```bash
pytest tests/ -v                              # Run all tests
pytest tests/test_cli_pipeline.py -v          # Pipeline tests
pytest tests/test_swe_lego_launcher.py -v     # SWE-Lego tests
pytest tests/ -k "not swe_lego" -v            # Skip SWE-Lego (needs remote)
```

## Environment Setup (on ssh torch)

```bash
# SWE-Lego dependencies
export SWE_LEGO_ROOT=/path/to/SWE-Lego
export LLAMA_FACTORY_DIR=$SWE_LEGO_ROOT/LLaMA-Factory
# Or run: bash trainers/swe_lego/setup_swe_lego.sh

# Results DB override
export ACT_RESULTS_DB=/path/to/results.db

# Anthropic API key (for Claude Code extension to work in remote context)
export ANTHROPIC_API_KEY=sk-ant-...
```

## Workflow: Local Claude Code → Remote Training

Typical session:

1. **Open VS Code**, connect Remote-SSH to `torch`
2. **Open this project** in the remote window
3. **Talk to Claude Code** — all commands execute on `torch`:

```
用户: "帮我改一下这个 recipe 的 learning rate 到 1e-5"
  → Claude Code edits recipe JSON on remote

用户: "dry-run 看看有没有问题"
  → act train recipe.json --dry-run

用户: "提交训练"
  → act train recipe.json
  → SLURM jobs submitted, job IDs tracked in DB

用户: "看看训练状态"
  → act status --recipe-id <id> --slurm
  → Shows: train RUNNING, infer PENDING, eval PENDING...

用户: "训练完了吗？导入结果"
  → act sync --recipe-id <id>
  → Auto-imports results, runs judge, generates report

用户: "结果怎么样"
  → Claude Code reads report + DB + SLURM logs

用户: "OOM 了，帮我调参重跑"
  → Claude Code edits recipe, resubmits via act train
```

**Remember**: Claude Code here is the operator, not the trainer. It reads/writes code and runs commands on the remote server. The actual GPU training is SLURM's job.

## ARIS Research Plane (aris/)

Separate from training plane. Contains 75+ Claude Code skills for paper discovery, review, and writing. Uses Claude Code interactive mode:
```
/idea-discovery "coding agents"
/experiment-bridge
/auto-review-loop "paper topic"
/research-pipeline "direction"
```
Not needed for training operations — training plane is self-contained.
