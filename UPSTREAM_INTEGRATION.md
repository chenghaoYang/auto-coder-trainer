# Upstream Integration

This project is intentionally split into two layers:

- **Control plane (owned here)**: Recipe IR, CLI orchestration, result DB, judge, task ledgers, reports, harness policy
- **Algorithm plane (prefer upstream)**: SFT, DPO, RL/GRPO, distillation recipes, teacher-trajectory training stacks

## Integration Policy

Prefer these options in order:

1. Use an upstream library directly through a thin adapter
2. Generate a launcher bundle for an upstream repo
3. Re-implement locally only when the upstream path is missing or too rigid

We do this to avoid long-term fork maintenance while keeping experiment control, recovery, and reporting inside this repo.

## Current Mapping

| Capability | Preferred upstream | How this repo uses it |
| --- | --- | --- |
| SFT | TRL | Native `backend=trl` via `trainers/sft/trainer.py` |
| Pairwise distill refinement | TRL DPO | Native `trainer.type=distill` + `distill.refine_algorithm=dpo` via `trainers/distill/trainer.py` |
| RL / GRPO / PPO | veRL | Native `backend=verl` or TinyZero launcher |
| TinyZero baselines | TinyZero | External launch bundle via `trainers/tinyzero/launcher.py` |
| Open-R1 recipes | Open-R1 | External launch bundle via `trainers/upstream/launcher.py` |
| Agent trajectory distillation | Agent Distillation | External launch bundle via `trainers/upstream/launcher.py` |
| REDI negative-signal refinement | Reinforcement Distillation | External launch bundle via `trainers/upstream/launcher.py` |

## Native vs External Distillation

### Native `backend=trl`

Use this when:

- you already have offline teacher traces
- you want the project to run end-to-end locally
- you are fine with positive trajectory SFT plus TRL-based DPO refinement

### External `backend=redi`

Use this when:

- your dataset contains chosen/rejected traces
- you specifically want the official REDI-style negative-signal refinement recipe
- you prefer to stay closer to the upstream paper implementation than to a local approximation

### External `backend=agent_distill`

Use this when:

- you want tool-using teacher trajectory generation and student-agent training from the official stack
- the task is truly agentic, not just plain response distillation

### External `backend=openr1`

Use this when:

- you want to stay close to the Open-R1 training/config stack
- you are doing reasoning-style SFT/GRPO or distillation-adjacent recipe work

## Ownership Rule

If a future change is mainly:

- **algorithmic**: upstream-first
- **orchestration / recovery / evaluation policy**: keep it in this repo
