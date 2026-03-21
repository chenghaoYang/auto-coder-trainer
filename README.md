# Auto-Coder-Trainer

**Research Operating System for Coding Agent Training**

> From papers to trained models in a closed loop: automatically collect research, compose training recipes, run experiments, and generate reports.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │ Collect  │───▶│ Compose  │───▶│  Train   │───▶│  Report  │  │
│   │          │    │          │    │          │    │          │  │
│   │ papers   │    │ recipe   │    │ SFT / RL │    │ analysis │  │
│   │ methods  │    │   IR     │    │  (veRL)  │    │ verdicts │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│        ▲                                               │        │
│        └───────────────────────────────────────────────┘        │
│                     feedback loop                               │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌───────────────────────────────────────────────┐
│          ARIS Research Plane (aris/)           │
│  skills · mcp-servers · tools · docs          │
│  75+ skills for research, review, paper writing│
└──────────────────┬────────────────────────────┘
                   │ method cards
          ┌────────▼────────┐
          │   Recipe IR     │
          │  (recipes/)     │
          │  JSON schema    │
          └────────┬────────┘
                   │ compiled config
┌──────────────────▼────────────────────────────┐
│          Training Plane                        │
│  tinyzero launcher · trainers/sft · trainers/rl│
│  evaluators/ · judge/ · results/              │
└───────────────────────────────────────────────┘
```

**Two planes, one IR.** The [ARIS Research Plane](aris/README.md) handles paper collection, idea generation, novelty checking, and research orchestration. The Training Plane handles experiment execution, evaluation, and result accumulation. The **Recipe IR** — a structured JSON schema — connects them.

## Quick Start

```bash
# Install
git clone https://github.com/chenghaoYang/auto-coder-trainer.git
cd auto-coder-trainer
python3 -m pip install -e ".[all,dev]"

# Five entry points
act collect "coding agent training"               # Online discovery (arXiv + GitHub) → registry
act compose --atoms swe-fuse,entropy-rl           # Compose → schema-clean recipe
act train recipes/examples/baseline-sft.recipe.json  # Native train or TinyZero launch bundle
act train recipes/examples/trajectory-distill.recipe.json  # Teacher-trajectory distillation
act report --recipe-id recipe-baseline-sft-001    # Report → comparison / verdicts / ablations (also: --experiment-id)
act status --open-only                            # Recovery view: experiments, artifacts, pending tasks
```

Or use `make`:

```bash
make dev
make collect QUERY="coding agent training"
make compose ATOMS="swe-fuse,entropy-rl"
make train RECIPE=recipes/examples/baseline-sft.recipe.json
make report EXP_ID=exp_001
make status RECIPE_ID=recipe-baseline-sft-001
```

Or use Claude Code skills:

```bash
claude
> /collect "coding agent trajectory training"
> /compose swe-fuse + entropy-rl
> /train recipes/examples/trajectory-rl.recipe.json
> /report exp_001
```

`collect` supports both offline import mode (`method_atoms.json` / JSONL / inline JSON) and automated discovery mode via arXiv + GitHub search. `train` now supports three native post-training tracks: SFT, RL/GRPO, and trajectory distillation. Distillation is designed for coding-agent teacher traces: it runs positive teacher-trajectory SFT first, then can optionally refine with TRL DPO on chosen-vs-rejected traces. For stronger upstream alignment, the same control plane can also generate launch bundles for `openr1`, `agent_distill`, and `redi`. Every `train` invocation persists experiment state, evaluation rows, artifacts, and a task ledger so the next agent can resume from the current checkpoint instead of re-auditing the repo.

## Recovery And State

The project now keeps resumable state in two places:

- `data/results.db`: experiments, eval runs, verdicts, artifacts, and tracked tasks
- `outputs/<recipe-id>/task-ledger.{json,md}`: a human/agent-readable snapshot of what finished and what is still open for that recipe

Useful commands:

```bash
act status --open-only
act status --recipe-id recipe-baseline-sft-001 --output outputs/status.md
```

If you want a separate DB while testing, set `ACT_RESULTS_DB=/path/to/results.db`.

## Recipe IR

Every experiment is defined by a **Recipe IR** — a structured JSON file that captures all dimensions of a training experiment:

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
    "type": "grpo",
    "backend": "tinyzero",       // TinyZero/veRL-compatible baseline launcher
    "reward": { "type": "entropy_aware" }
  },
  "eval": { "benchmarks": ["swe-bench-verified"], "seeds": [42, 123, 456] },
  "ablation": [{ "name": "reward_type", "variable": "trainer.reward.type", "values": ["binary_pass", "entropy_aware"] }],
  "budget": { "max_gpu_hours": 48, "gpu_type": "A100-80GB" }
}
```

See [`recipes/schema/recipe.schema.json`](recipes/schema/recipe.schema.json) for the full schema and [`recipes/examples/`](recipes/examples/) for example recipes.

Distillation recipes add an optional `distill` block:

```jsonc
{
  "trainer": { "type": "distill", "backend": "trl" },
  "distill": {
    "strategy": "trajectory",
    "teacher_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "teacher_mode": "offline_dataset",
    "stages": ["positive_sft", "pairwise_refine"],
    "refine_algorithm": "dpo",
    "pairwise_beta": 0.1,
    "condense": { "strategy": "edge_preserving", "max_chars": 12000 }
  }
}
```

## Experiment Judge

Every experiment passes through the **Experiment Judge** before results are accepted:

| Check | What it does |
|-------|-------------|
| Baseline alignment | Verifies a baseline run exists for comparison |
| Seed consistency | Confirms all specified seeds were evaluated |
| Minimal ablation | Ensures ablation experiments cover recipe variables |
| Deduplication | Prevents redundant experiments |
| Failure attribution | Analyzes underperforming experiments |

Verdicts: `accept` · `reject` · `needs_ablation` · `needs_rerun`

## Project Structure

```
auto-coder-trainer/
├── aris/                 # Research Plane — ARIS agent scaffold
│   ├── skills/           #   75+ Claude Code / Codex skills
│   ├── mcp-servers/      #   4 MCP servers (review, LLM chat, Feishu)
│   ├── tools/            #   arXiv fetcher, utilities
│   └── docs/             #   Guides and examples
├── recipes/              # Recipe IR layer
│   ├── schema/           #   JSON Schema definition
│   ├── registry/         #   Method atom registry
│   ├── examples/         #   Example recipes
│   └── compiler.py       #   Recipe → training config compiler
├── trainers/             # Training Plane
│   ├── tinyzero/         #   TinyZero/veRL-compatible launch bundle compiler
│   ├── sft/              #   Native SFT trainer (TRL backend)
│   ├── rl/               #   Native RL trainer (veRL backend)
│   └── utils/            #   Seeds, checkpoints
├── evaluators/           # Evaluation harness
│   ├── swe_bench.py      #   SWE-bench evaluator
│   └── pass_at_k.py      #   pass@k metrics
├── judge/                # Experiment judge
│   ├── judge.py          #   Core arbiter
│   ├── baseline.py       #   Baseline alignment
│   ├── ablation.py       #   Ablation validation
│   └── attribution.py    #   Failure analysis
├── results/              # Result database + ledgers
│   ├── db.py             #   SQLite-backed storage
│   ├── ledger.py         #   Task ledger writer for recovery
│   └── report_generator.py  # Auto report generation
├── prompt_cache/          # Prompt caching infrastructure
│   ├── builder.py        #   Cache-safe prompt construction
│   ├── monitor.py        #   Hit rate monitoring & alerts
│   ├── compaction.py     #   Cache-safe context compression
│   └── rules.py          #   Codified cache safety rules
├── cli/                  # CLI entry points
│   └── main.py           #   `act` command
└── skills/               # Top-level skills for Claude Code
    ├── collect/           #   /collect
    ├── compose/           #   /compose
    ├── train/             #   /train
    └── report/            #   /report
```

## ARIS Research Plane

The [ARIS (Auto-Research-In-Sleep)](aris/README.md) subsystem provides 75+ Claude Code skills for autonomous ML research:

- **Research**: `/research-lit`, `/arxiv`, `/idea-discovery`, `/novelty-check`
- **Experimentation**: `/experiment-plan`, `/experiment-bridge`, `/run-experiment`
- **Writing**: `/paper-write`, `/paper-figure`, `/paper-slides`, `/paper-poster`
- **Review**: `/auto-review-loop`, `/research-review`, cross-model review
- **Pipeline**: `/research-pipeline` — end-to-end from idea to paper

See the [ARIS README](aris/README.md) for full documentation.

## Prompt Caching

"Cache Rules Everything Around Me." The `prompt_cache/` module implements cache-safe prompt construction for long-running agent sessions, based on Anthropic's prefix-matching cache architecture.

**Core principle**: The API caches from the start of each request to each `cache_control` breakpoint. Prompt order and stability determine cache hit rate.

| Layer | Content | Stability | Cache Scope |
|-------|---------|-----------|-------------|
| 0 | System prompt + tool definitions | Frozen at session start | Cross-session |
| 1 | Project context (CLAUDE.md) | Frozen per project | Cross-session |
| 2 | Session context (task, plan) | Append-only | Within session |
| 3 | Conversation messages | Dynamic | Not cached |

**Six rules enforced by `prompt_cache/rules.py`**:

1. **Static prefix ordering** — static content before dynamic content
2. **Messages for updates** — use `<system-reminder>` in messages, never modify system prompt
3. **Tool set stability** — never add/remove tools mid-session; use deferred loading
4. **Model consistency** — never switch models mid-session (cache is model-bound)
5. **State via tools** — model state changes (Plan Mode) via tool calls, not toolset changes
6. **Compaction prefix sharing** — context compression reuses parent's exact prefix

```python
from prompt_cache import PromptBuilder, CacheMonitor

builder = PromptBuilder()
builder.set_system_prompt("You are a coding agent...")
builder.set_tools([...])                    # Set once, frozen
builder.add_project_context(claude_md)       # Layer 1
builder.add_session_context("Task: ...")     # Layer 2
builder.inject_dynamic_update("Time: now")   # Via message, not system prompt

monitor = CacheMonitor(alert_threshold=0.8)
# After API call:
monitor.record(response.usage)
if not monitor.is_healthy():
    print(monitor.diagnose_cache_miss())
```

## Training Backends

| Backend | Used for | Framework |
|---------|----------|-----------|
| **TinyZero** | Baseline SFT / RL launch bundles | [Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero) |
| **veRL** | RL / GRPO / PPO | [volcengine/verl](https://github.com/volcengine/verl) |
| **TRL** | SFT / native trajectory distillation / DPO refinement | [huggingface/trl](https://github.com/huggingface/trl) |
| **Open-R1** | External reasoning / distillation recipe launcher | [huggingface/open-r1](https://github.com/huggingface/open-r1) |
| **Agent Distillation** | External teacher-agent trajectory distillation launcher | [Nardien/agent-distillation](https://github.com/Nardien/agent-distillation) |
| **REDI** | External negative-signal refinement launcher | [Tim-Siu/reinforcement-distillation](https://github.com/Tim-Siu/reinforcement-distillation) |

## Distillation Track

The project supports a coding-agent-friendly distillation path rather than a narrow logits-only KD setup:

- **Positive trajectory distillation**: train the student on teacher-generated agent traces or high-quality coding completions.
- **Optional native pairwise refinement**: when the dataset includes `chosen`/`rejected` traces, run a second-stage TRL DPO refinement to push the student toward better trajectories and away from bad ones.
- **Optional upstream refinement / training stacks**: switch to `backend=redi`, `backend=agent_distill`, or `backend=openr1` when you want to stay closer to the official SOTA implementation than to the native adapter path.
- **Trajectory condensation**: long agent traces can be edge-preserving condensed before training so we keep the beginning/end of the reasoning path without exploding token cost.

This choice is intentional: for coding agents, teacher trajectories and tool-using traces are usually easier to obtain and more reusable than full teacher logits, and recent open work has shown strong results from this regime. See [UPSTREAM_INTEGRATION.md](UPSTREAM_INTEGRATION.md) for the upstream-first policy.

### TinyZero Migration

TinyZero is the current baseline interface for both SFT and RL recipes in this repo. We migrate it at the launcher layer instead of vendoring the whole project:

- keep our Recipe IR, judge, reports, and result DB as the control plane
- compile `backend=tinyzero` recipes into TinyZero-style Hydra overrides and runnable shell scripts
- stay compatible with the underlying veRL entry points that TinyZero itself builds on

This keeps the framework maintainable while preserving baseline reproducibility.

## Status

This project is under active development. Current status:

- [x] ARIS Research Plane (75+ skills, 4 MCP servers)
- [x] Recipe IR JSON Schema
- [x] Project skeleton (trainers, evaluators, judge, results, CLI)
- [x] Prompt cache infrastructure (builder, monitor, compaction, rules)
- [x] CLI automation shell (`collect`, `compose`, `train`, `report`, `status`)
- [x] Experiment judge logic and result DB helpers
- [x] Report generation with verdict / ablation / multi-experiment comparison
- [x] TinyZero baseline launcher for SFT / RL recipes
- [x] SFT trainer implementation (TRL / Transformers fallback, dependency-gated)
- [x] Distillation trainer implementation (trajectory SFT + optional TRL DPO refinement)
- [x] RL trainer implementation (veRL)
- [x] SWE-bench evaluator integration
- [x] Persistent experiment recovery (`eval_runs`, tasks, artifacts, task ledgers)
- [x] Upstream launcher adapters for Open-R1 / Agent Distillation / REDI
- [ ] Case studies and reproductions

## License

[MIT](LICENSE)

## Citation

```bibtex
@software{auto_coder_trainer,
  title={Auto-Coder-Trainer: Research Operating System for Coding Agent Training},
  author={Chenghao Yang},
  year={2026},
  url={https://github.com/chenghaoYang/auto-coder-trainer}
}
```
