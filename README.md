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
python3 -m pip install -e ".[dev,tinyzero]"

# Four entry points
act collect recipes/registry/method_atoms.json    # Import method atoms → registry
act compose --atoms swe-fuse,entropy-rl           # Compose → schema-clean recipe
act train recipes/examples/baseline-sft.recipe.json  # Validate/plan → TinyZero launch bundle or native run
act report --recipe-id recipe-baseline-sft-001    # Report → comparison / verdicts / ablations (also: --experiment-id)
```

Or use `make`:

```bash
make dev
make collect QUERY=path/to/method_atoms.json
make compose ATOMS="swe-fuse,entropy-rl"
make train RECIPE=recipes/examples/baseline-sft.recipe.json
make report EXP_ID=exp_001
```

Or use Claude Code skills:

```bash
claude
> /collect "coding agent trajectory training"
> /compose swe-fuse + entropy-rl
> /train recipes/examples/trajectory-rl.recipe.json
> /report exp_001
```

`collect` is currently most useful in offline import mode (local `method_atoms.json` / JSONL / inline JSON). Automated paper discovery still depends on ARIS research skills. `train` now supports two modes: native backends still validate and plan, while `backend=tinyzero` compiles a ready-to-edit launch bundle (`env.sh`, `hydra-overrides.txt`, `run.sh`) for external baseline runs.

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
├── results/              # Result database
│   ├── db.py             #   SQLite-backed storage
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
| **TRL** | SFT / DPO | [huggingface/trl](https://github.com/huggingface/trl) |

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
- [x] CLI automation shell (`collect` import, `compose`, `train` execution plan / TinyZero bundle, `report`)
- [x] Experiment judge logic and result DB helpers
- [x] Report generation with verdict / ablation / multi-experiment comparison
- [x] TinyZero baseline launcher for SFT / RL recipes
- [ ] SFT trainer implementation (TRL)
- [x] RL trainer implementation (veRL)
- [x] SWE-bench evaluator integration
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
