---
name: train
description: Execute a training experiment from a recipe. Use when user says "开始训练", "run experiment", "train model", or wants to execute a recipe.
argument-hint: [recipe-path]
allowed-tools: Bash(*), Read, Write, Grep, Glob
---

# Train — Experiment Executor

Execute training experiment from: $ARGUMENTS

## Pipeline

1. **Load** — Read and validate the recipe JSON file
2. **Compile** — Use `recipes/compiler.py` to compile recipe into training config
3. **Prepare** — Set up data, model, and environment
4. **Train** — Execute training:
   - SFT → `trainers/sft/trainer.py` (TRL backend)
   - RL/GRPO → `trainers/rl/trainer.py` (veRL backend)
5. **Evaluate** — Run benchmarks from recipe.eval on the trained checkpoint
6. **Judge** — Submit results to experiment judge for verdict
7. **Store** — Save results to result DB
8. **Report** — Print summary with metrics, verdict, and next steps

## Constraints

- ALWAYS set seeds before training (reproducibility)
- ALWAYS run baseline comparison if no baseline exists for this config
- NEVER skip evaluation — every training run must be evaluated
- Log all outputs to `outputs/<recipe-id>/`
