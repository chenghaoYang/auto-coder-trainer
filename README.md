# Auto-Coder-Trainer

**Research Operating System for Coding Agent Training**

> From papers to trained models in a closed loop — automatically collect research,
> compose training recipes, run experiments, and generate publication-ready reports.

```
         collect           compose            train             report
      ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
      │  arXiv   │────▶│  Recipe  │────▶│ SFT / RL │────▶│  Blog /  │
      │  GitHub  │     │   IR     │     │ Distill  │     │  LaTeX   │
      └──────────┘     └──────────┘     └──────────┘     └──────────┘
           ▲                                                   │
           └────────────── judge verdict ──────────────────────┘
                          feedback loop
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [One-Command Pipeline](#one-command-pipeline)
- [Architecture](#architecture)
- [CLI Reference](#cli-reference)
- [Recipe IR](#recipe-ir)
- [Training Backends](#training-backends)
- [Experiment Judge](#experiment-judge)
- [Blog-Style Reports](#blog-style-reports)
- [Recovery & State](#recovery--state)
- [ARIS Research Plane](#aris-research-plane)
- [Prompt Caching](#prompt-caching)
- [Project Structure](#project-structure)
- [Development](#development)
- [Status](#status)
- [License](#license)

---

## Quick Start

```bash
# Install
git clone https://github.com/chenghaoYang/auto-coder-trainer.git
cd auto-coder-trainer
pip install -e ".[all,dev]"

# Run the full pipeline in one command
act pipeline --query "coding agent training" --report-format blog

# Or step by step
act collect "coding agent training"
act compose --atoms swe-fuse,entropy-rl
act train recipes/examples/baseline-sft.recipe.json
act report --recipe-id recipe-baseline-sft-001 --format blog
```

---

## One-Command Pipeline

The `act pipeline` command orchestrates the **entire agent team** as a closed loop:

```bash
# Full pipeline: collect → compose → train → judge → report
act pipeline --query "coding agent training" --report-format blog

# Start from an existing recipe (skip collect/compose)
act pipeline --recipe recipes/examples/baseline-sft.recipe.json

# Control iteration depth and output
act pipeline --recipe recipe.json --max-iterations 5 --report-dir reports/

# Validate without training
act pipeline --recipe recipe.json --dry-run
```

**Automatic decision logic** after each judge verdict:

| Verdict | Action |
| --- | --- |
| `accept` | Generate final blog report, done |
| `needs_rerun` | Re-run missing seeds or full experiment, loop |
| `needs_ablation` | Run ablation variants, loop |
| `reject` | Generate report with failure analysis, stop |

The pipeline iterates up to `--max-iterations` (default 3), then generates a final report regardless.

---

## Architecture

```
┌───────────────────────────────────────────────────┐
│           ARIS Research Plane  (aris/)             │
│   75+ skills  ·  4 MCP servers  ·  arXiv tools    │
└────────────────────┬──────────────────────────────┘
                     │ method atoms
              ┌──────▼──────┐
              │  Recipe IR  │  ← JSON Schema (Draft 2020-12)
              │ (recipes/)  │
              └──────┬──────┘
                     │ compiled config
┌────────────────────▼──────────────────────────────┐
│           Training Plane                           │
│   trainers/ · evaluators/ · judge/ · results/      │
│   7 backends · 2 evaluators · 5 judge checks       │
└───────────────────────────────────────────────────┘
```

**Two planes, one IR.** The ARIS Research Plane handles paper collection, idea generation, and research orchestration. The Training Plane handles experiment execution, evaluation, and result persistence. The **Recipe IR** — a structured JSON schema — bridges them.

---

## CLI Reference

| Command | Description | Example |
| --- | --- | --- |
| `act pipeline` | Full closed-loop pipeline | `act pipeline --query "RL for code" --report-format blog` |
| `act collect` | Discover papers & repos → method atoms | `act collect "coding agent training"` |
| `act compose` | Assemble atoms → Recipe IR JSON | `act compose --atoms swe-fuse,entropy-rl` |
| `act train` | Execute experiment from recipe | `act train recipe.json` |
| `act report` | Generate report (blog / markdown / latex) | `act report --recipe-id R --format blog` |
| `act status` | Show experiments & open tasks | `act status --open-only` |
| `act rerun` | Auto-dispatch pending judge tasks | `act rerun --recipe-id R` |

**Claude Code skills** (equivalent):

```
/collect "coding agent trajectory training"
/compose swe-fuse + entropy-rl
/train recipes/examples/trajectory-rl.recipe.json
/report exp_001
```

---

## Recipe IR

Every experiment is defined by a **Recipe IR** — a structured JSON validated against [`recipe.schema.json`](recipes/schema/recipe.schema.json):

```jsonc
{
  "id": "recipe-trajectory-rl-001",
  "name": "GRPO on Coding Trajectories with Entropy-Aware Reward",
  "model": { "base": "Qwen/Qwen2.5-Coder-7B-Instruct", "adapter": "full" },
  "dataset": {
    "sources": [{ "name": "swe-rebench-trajectories", "path": "openhands/..." }],
    "filters": [{ "type": "issue_free" }, { "type": "length", "params": { "max_turns": 30 } }]
  },
  "trainer": {
    "type": "grpo",                       // sft | rl | grpo | distill | dpo
    "backend": "tinyzero",                // trl | verl | tinyzero | openr1 | agent_distill | redi
    "reward": { "type": "entropy_aware" }
  },
  "eval": {
    "benchmarks": ["swe-bench-verified"],
    "seeds": [42, 123, 456]
  },
  "ablation": [{
    "name": "reward_type",
    "variable": "trainer.reward.type",
    "values": ["binary_pass", "entropy_aware"]
  }],
  "budget": { "max_gpu_hours": 48, "gpu_type": "A100-80GB" }
}
```

**Distillation recipes** add an optional `distill` block for teacher-trajectory training:

```jsonc
{
  "trainer": { "type": "distill", "backend": "trl" },
  "distill": {
    "strategy": "trajectory",
    "teacher_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "stages": ["positive_sft", "pairwise_refine"],
    "refine_algorithm": "dpo",
    "condense": { "strategy": "edge_preserving", "max_chars": 12000 }
  }
}
```

See [`recipes/examples/`](recipes/examples/) for complete examples covering SFT, RL/GRPO, and distillation.

---

## Training Backends

| Backend | Type | Used For | Framework |
| --- | --- | --- | --- |
| **TRL** | Native | SFT, distillation, DPO refinement | [huggingface/trl](https://github.com/huggingface/trl) |
| **veRL** | Native | RL, GRPO, PPO | [volcengine/verl](https://github.com/volcengine/verl) |
| **TinyZero** | Launcher | Baseline SFT & RL launch bundles | [Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero) |
| **Open-R1** | Launcher | Reasoning recipe launcher | [huggingface/open-r1](https://github.com/huggingface/open-r1) |
| **Agent Distillation** | Launcher | Teacher-agent trajectory distillation | [Nardien/agent-distillation](https://github.com/Nardien/agent-distillation) |
| **REDI** | Launcher | Negative-signal reinforcement distillation | [Tim-Siu/reinforcement-distillation](https://github.com/Tim-Siu/reinforcement-distillation) |

**Native** trainers run directly. **Launcher** backends generate Hydra configs + shell scripts for external execution. See [UPSTREAM_INTEGRATION.md](UPSTREAM_INTEGRATION.md) for the upstream-first integration policy.

---

## Experiment Judge

Every experiment passes through an automated judge before results are accepted:

| Check | Description |
| --- | --- |
| Baseline alignment | Verifies a baseline run exists; allows 5% regression tolerance |
| Seed consistency | Confirms all seeds evaluated; coefficient of variation < 10% |
| Ablation coverage | Ensures ablation experiments cover all recipe variables |
| Deduplication | Prevents redundant experiments via config hash matching |
| Failure attribution | Analyzes and explains underperforming experiments |

**Verdicts**: `accept` · `reject` · `needs_ablation` · `needs_rerun`

When used with `act pipeline`, verdicts automatically drive the next action — no manual intervention needed.

---

## Blog-Style Reports

Reports can be generated in three formats: `markdown`, `latex`, and `blog`.

The **blog** format follows the structure of Sebastian Raschka's [LoRA Insights](https://lightning.ai/blog/lora-insights) — an experiment diary style optimized for readability:

```bash
act report --recipe-id recipe-001 --format blog
```

**Report structure**:

| Section | Content |
| --- | --- |
| **TL;DR** | Key takeaways, best metrics, seed stability summary |
| **Introduction** | Motivation, method description, source papers |
| **Experimental Setup** | Model, dataset, hyperparameters, hardware, evaluation config |
| **Experiments** | Sequential diary — each run with question, setup, results table, judge verdict, key finding |
| **Ablation Studies** | Per-variable comparison tables with best setting highlighted |
| **Cost & Efficiency** | GPU budget, performance-per-GPU-hour analysis |
| **Practical Recommendations** | Synthesized from judge suggestions, ablation results, seed stability |
| **Reproducibility** | Commands to re-run, config hashes, seed list |
| **Conclusion** | Accept/reject summary, best results, next steps |

---

## Recovery & State

The project maintains resumable state so agents can pick up where they left off:

| Store | Location | Purpose |
| --- | --- | --- |
| Result DB | `data/results.db` | Experiments, eval runs, verdicts, artifacts, tasks (SQLite, WAL mode) |
| Task ledger | `outputs/<recipe-id>/task-ledger.{json,md}` | Human/agent-readable snapshot of completed and pending work |
| Execution plan | `outputs/<recipe-id>/execution-plan.{json,md}` | Next steps for blocked or prepared experiments |

```bash
act status --open-only                                    # What's pending?
act status --recipe-id recipe-001 --output status.md      # Save status report
act rerun --recipe-id recipe-001                          # Auto-dispatch pending tasks
```

Override the DB path for testing: `ACT_RESULTS_DB=/path/to/test.db`

---

## ARIS Research Plane

The [ARIS (Auto-Research-In-Sleep)](aris/README.md) subsystem provides **75+ Claude Code skills** for autonomous ML research:

| Category | Skills |
| --- | --- |
| **Research** | `/research-lit`, `/arxiv`, `/idea-discovery`, `/novelty-check` |
| **Experiment** | `/experiment-plan`, `/experiment-bridge`, `/run-experiment` |
| **Writing** | `/paper-write`, `/paper-figure`, `/paper-slides`, `/paper-poster` |
| **Review** | `/auto-review-loop`, `/research-review`, cross-model review |
| **Pipeline** | `/research-pipeline` — end-to-end from idea to paper |

Plus **4 MCP servers**: claude-review, feishu-bridge, llm-chat, minimax-chat.

See the [ARIS README](aris/README.md) for full documentation.

---

## Prompt Caching

The `prompt_cache/` module implements cache-safe prompt construction for long-running agent sessions, based on Anthropic's prefix-matching cache architecture.

| Layer | Content | Stability |
| --- | --- | --- |
| 0 | System prompt + tool definitions | Frozen at session start |
| 1 | Project context (CLAUDE.md) | Frozen per project |
| 2 | Session context (task, plan) | Append-only within session |
| 3 | Conversation messages | Dynamic, not cached |

**Six rules** (enforced by `prompt_cache/rules.py`): static prefix ordering, messages for updates, tool set stability, model consistency, state via tools, compaction prefix sharing.

```python
from prompt_cache import PromptBuilder, CacheMonitor

builder = PromptBuilder()
builder.set_system_prompt("You are a coding agent...")
builder.set_tools([...])                    # Set once, frozen
builder.add_project_context(claude_md)       # Layer 1
builder.add_session_context("Task: ...")     # Layer 2

monitor = CacheMonitor(alert_threshold=0.8)
monitor.record(response.usage)
if not monitor.is_healthy():
    print(monitor.diagnose_cache_miss())
```

---

## Project Structure

```
auto-coder-trainer/
│
├── cli/                          # CLI entry points
│   ├── main.py                   #   `act` command dispatcher
│   ├── pipeline.py               #   Full agent team orchestrator
│   ├── collect.py                #   Paper/repo discovery → method atoms
│   ├── compose.py                #   Atoms → Recipe IR
│   ├── train.py                  #   Recipe → training execution
│   ├── report.py                 #   Results → blog / markdown / LaTeX
│   ├── status.py                 #   Task & experiment summary
│   └── rerun.py                  #   Auto-dispatch pending tasks
│
├── recipes/                      # Recipe IR layer
│   ├── schema/recipe.schema.json #   JSON Schema (Draft 2020-12)
│   ├── registry/                 #   Method atom registry
│   ├── examples/                 #   3 example recipes (SFT, RL, distill)
│   └── compiler.py               #   Recipe → TrainingConfig compiler
│
├── trainers/                     # Training backends
│   ├── base.py                   #   Abstract TrainResult / EvalResult
│   ├── sft/                      #   SFT trainer (TRL)
│   ├── rl/                       #   RL/GRPO trainer (veRL)
│   ├── distill/                  #   Distillation trainer (SFT + DPO)
│   ├── tinyzero/                 #   TinyZero launcher
│   ├── upstream/                 #   Open-R1, Agent Distill, REDI launchers
│   └── utils/                    #   Budget, checkpoints, LoRA, seeds
│
├── evaluators/                   # Evaluation harness
│   ├── swe_bench.py              #   SWE-bench integration
│   ├── pass_at_k.py              #   pass@k metrics
│   └── runner.py                 #   Multi-benchmark orchestration
│
├── judge/                        # Experiment judge (5 checks, 4 verdicts)
│   ├── judge.py                  #   Core arbiter
│   ├── baseline.py               #   Baseline alignment
│   ├── ablation.py               #   Ablation validation
│   ├── attribution.py            #   Failure analysis
│   └── dedup.py                  #   Config hash deduplication
│
├── results/                      # Persistence & reporting
│   ├── db.py                     #   SQLite result database (8 tables)
│   ├── schema.sql                #   DB schema
│   ├── ledger.py                 #   Task ledger for crash recovery
│   └── report_generator.py       #   Markdown / LaTeX / blog reports
│
├── prompt_cache/                 # Cache-safe prompt construction
│   ├── builder.py                #   4-layer prompt builder
│   ├── monitor.py                #   Cache hit rate monitoring
│   ├── compaction.py             #   Context compression
│   └── rules.py                  #   6 cache safety rules
│
├── aris/                         # ARIS Research Plane
│   ├── skills/                   #   75+ Claude Code skills
│   ├── mcp-servers/              #   4 MCP servers
│   ├── tools/                    #   arXiv fetcher, utilities
│   └── docs/                     #   Setup guides
│
├── skills/                       # Top-level Claude Code skills
│   ├── collect/                  #   /collect
│   ├── compose/                  #   /compose
│   ├── train/                    #   /train
│   └── report/                   #   /report
│
├── tests/                        # Test suite (57 tests)
├── pyproject.toml                # Package config (Python >= 3.10)
├── Makefile                      # Convenience targets
└── UPSTREAM_INTEGRATION.md       # Upstream integration policy
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[all,dev]"

# Run tests
make test                     # or: python -m pytest tests/ -q

# Lint
make lint                     # or: ruff check .

# Run specific test
python -m pytest tests/test_pipeline_and_blog_report.py -v
```

**Optional install extras**:

| Extra | Packages |
| --- | --- |
| `sft` | trl, transformers, datasets, peft, accelerate |
| `rl` | verl |
| `eval` | swebench |
| `dev` | pytest, ruff |
| `all` | All of the above |

---

## Status

- [x] ARIS Research Plane (75+ skills, 4 MCP servers)
- [x] Recipe IR JSON Schema + compiler
- [x] CLI automation (`collect`, `compose`, `train`, `report`, `status`, `rerun`)
- [x] **Full pipeline orchestrator** (`act pipeline`) — closed-loop agent team
- [x] **Blog-style report generator** — LoRA Insights-inspired experiment diary
- [x] Experiment judge (5 checks, 4 verdicts, auto-decision loop)
- [x] SFT trainer (TRL backend, Full/LoRA/QLoRA)
- [x] RL/GRPO trainer (veRL backend, 4 reward types)
- [x] Distillation trainer (trajectory SFT + optional DPO refinement)
- [x] SWE-bench & pass@k evaluators
- [x] Upstream launchers (TinyZero, Open-R1, Agent Distillation, REDI)
- [x] Persistent experiment recovery (SQLite DB, task ledgers, execution plans)
- [x] Prompt cache infrastructure (builder, monitor, compaction, 6 rules)
- [x] 57 tests passing
- [ ] Case studies and reproductions

---

## License

[MIT](LICENSE)

## Citation

```bibtex
@software{auto_coder_trainer,
  title   = {Auto-Coder-Trainer: Research Operating System for Coding Agent Training},
  author  = {Chenghao Yang},
  year    = {2025},
  url     = {https://github.com/chenghaoYang/auto-coder-trainer}
}
```
