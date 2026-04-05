# Harness 五子系统改进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 auto-coder-trainer 的 AI agent harness 从当前水平（均分 3.8/5）提升到 4.5+ 水平，按瓶颈优先级依次修复五个子系统。

**Architecture:** 五个子系统各自独立，按影响排序：工具子系统（最大瓶颈）→ 环境子系统 → 反馈子系统 → 状态子系统 → 指令子系统。每个 Task 是一个子系统的改进，产出可独立验证。

**Tech Stack:** Python 3.10+, ruff, pytest, pre-commit, pyright

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Rewrite | `.claude/settings.json` | 工具权限按语义类别授权 |
| Create | `.python-version` | 固定 Python 运行时版本 |
| Create | `.pre-commit-config.yaml` | Git commit 前自动 lint + format |
| Create | `requirements.lock` | 锁定核心依赖版本 |
| Create | `.env.example` | 环境变量模板 |
| Modify | `pyproject.toml` | 加 ruff format + pyright 配置 |
| Modify | `Makefile` | 加 check/format/lock 目标 |
| Modify | `CLAUDE.md` | 加硬约束清单 + 验证命令区 + 进入指引 |
| Create | `PROGRESS.md` | 显式进度文件 |
| Modify | `tests/conftest.py` | 加基本 smoke test fixture |

---

### Task 1: 工具子系统 — 重写 settings.json 权限配置（最大瓶颈）

**Files:**
- Rewrite: `.claude/settings.json`

当前问题：`permissions.allow` 列出了 6 条具体的 grep/pytest 命令，agent 想跑任何新命令都需要重新授权。

- [ ] **Step 1: 重写 `.claude/settings.json`**

将现有的具体命令列表替换为按语义类别授权的配置：

```json
{
  "permissions": {
    "allow": [
      "Bash(python *)",
      "Bash(pip *)",
      "Bash(git *)",
      "Bash(pytest *)",
      "Bash(ruff *)",
      "Bash(make *)",
      "Bash(act *)",
      "Bash(sqlite3 data/results.db *)",
      "Bash(squeue *)",
      "Bash(sacct *)",
      "Bash(scancel *)",
      "Bash(sbatch *)",
      "Bash(cat outputs/*)",
      "Bash(ls *)",
      "Bash(grep *)",
      "Bash(find *)",
      "Bash(mkdir *)",
      "Bash(rm *)",
      "Bash(echo *)",
      "Bash(head *)",
      "Bash(tail *)",
      "Bash(wc *)"
    ]
  }
}
```

- [ ] **Step 2: 验证新配置生效**

Run: `cat .claude/settings.json`
Expected: JSON 格式正确，包含 `python *`, `pytest *`, `ruff *` 等通配符模式

- [ ] **Step 3: 快速 smoke test — 确认 agent 能自主执行**

Run: `python -c "print('settings OK')"`
Expected: `settings OK`（无需额外授权弹窗）

- [ ] **Step 4: Commit**

```bash
git add .claude/settings.json
git commit -m "feat: rewrite settings.json with semantic wildcard permissions for agent autonomy"
```

---

### Task 2: 环境子系统 — 固定运行时 + 锁依赖 + 环境变量模板

**Files:**
- Create: `.python-version`
- Create: `.env.example`
- Create: `requirements.lock`
- Modify: `pyproject.toml`

- [ ] **Step 1: 创建 `.python-version`**

```
3.10
```

- [ ] **Step 2: 创建 `.env.example`**

```bash
# Auto-Coder-Trainer Environment Variables
# Copy to .env and fill in values

# Results DB path (optional, defaults to data/results.db)
ACT_RESULTS_DB=data/results.db

# SWE-Lego (required for swe-lego backend)
SWE_LEGO_ROOT=/path/to/SWE-Lego
LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory

# Anthropic API key (for Claude Code in remote context)
# ANTHROPIC_API_KEY=sk-ant-...
```

- [ ] **Step 3: 生成 `requirements.lock`**

Run: `pip freeze | grep -iE "^(jsonschema|pyyaml|setuptools|wheel)" > requirements.lock`

Expected: `requirements.lock` 包含 jsonschema, pyyaml, setuptools, wheel 的精确版本。

- [ ] **Step 4: 验证文件存在且内容合理**

Run: `cat .python-version && echo "---" && cat .env.example && echo "---" && cat requirements.lock`
Expected: 三个文件均有内容

- [ ] **Step 5: Commit**

```bash
git add .python-version .env.example requirements.lock
git commit -m "feat: add .python-version, .env.example, and requirements.lock for reproducibility"
```

---

### Task 3: 反馈子系统 — pre-commit hooks + ruff format + Makefile check

**Files:**
- Create: `.pre-commit-config.yaml`
- Modify: `pyproject.toml` (加 ruff format 配置)
- Modify: `Makefile` (加 format/check 目标)

- [ ] **Step 1: 创建 `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 2: 修改 `pyproject.toml` — 加 ruff format 配置**

在 `[tool.ruff]` 下追加：

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

注意：`[tool.ruff]` 已存在（line 88-89），在其后追加。

- [ ] **Step 3: 修改 `Makefile` — 加 format 和 check 目标**

在现有 `test` 目标后追加：

```makefile
format: ## Auto-format code with ruff
	$(PYTHON) -m ruff format . && $(PYTHON) -m ruff check --fix .

check: ## Run all checks (lint + test)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest tests/ -v
```

同时在 `.PHONY` 行（line 1）末尾追加 `format check`。

- [ ] **Step 4: 验证 ruff 和 Makefile 正常工作**

Run: `python -m ruff check . --select E,F,I,W`
Expected: 输出 lint 结果（可能有 warning 但不应报错）

Run: `make format`
Expected: ruff format + ruff check --fix 执行成功

- [ ] **Step 5: 安装 pre-commit hooks**

Run: `pip install pre-commit && pre-commit install`
Expected: `pre-commit installed at .git/hooks/pre-commit`

- [ ] **Step 6: Commit**

```bash
git add .pre-commit-config.yaml pyproject.toml Makefile
git commit -m "feat: add pre-commit hooks, ruff format config, and make check/format targets"
```

---

### Task 4: 状态子系统 — PROGRESS.md + CLAUDE.md 进入指引

**Files:**
- Create: `PROGRESS.md`
- Modify: `CLAUDE.md` (加进入指引)

- [ ] **Step 1: 创建 `PROGRESS.md`**

```markdown
# Progress Tracker

> 每个会话结束前更新此文件。新会话开始时读取此文件了解全局状态。

## Current Focus

- [进行中] Harness 五子系统改进

## Completed

- [x] SWE-Lego 后端 + SLURM pipeline
- [x] TinyZero 实验批量提交 (21 experiments, SLURM jobs 5395100-5395123)
- [x] Blog-style 报告生成器
- [x] Judge 系统 (5 checks, 4 verdicts)

## Blocked

- (暂无)

## Recent Changes

- feat: Enhance sync and training workflows with ablation support (14ee32e)
- feat: Add SLURM job tracking and sync functionality (fed937c)
- feat: Enhance SWE-Lego pipeline with external result import (042b13d)
```

- [ ] **Step 2: 修改 `CLAUDE.md` — 在 `## Quick Reference` 之前加"首次进入指引"区**

在 `## Deployment Model` 区块之后、`## Quick Reference` 之前（即 line 30 和 line 32 之间）插入：

```markdown
## First Session Checklist

> 每个新会话的 agent 应在开始时执行以下步骤：
1. 读取本文件 (CLAUDE.md)
2. 读取 PROGRESS.md 了解当前进度
3. 运行 `act status --open-only` 查看待办任务
4. 运行 `make check` 确认代码库状态健康
```

- [ ] **Step 3: 验证 PROGRESS.md 可读**

Run: `cat PROGRESS.md`
Expected: 包含 Current Focus / Completed / Blocked 三个区

- [ ] **Step 4: Commit**

```bash
git add PROGRESS.md CLAUDE.md
git commit -m "feat: add PROGRESS.md and first-session checklist in CLAUDE.md"
```

---

### Task 5: 指令子系统 — CLAUDE.md 硬约束清单 + 验证命令区

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 在 CLAUDE.md 的 `## Testing` 区（line 198）替换为完整的验证命令区**

将现有的 Testing 区（line 198-204）替换为：

```markdown
## Verification Commands

> agent 在完成任何修改后应运行以下命令确认状态：

```bash
# Lint 检查
ruff check .

# 自动格式化
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
```

- [ ] **Step 2: 在 `## Project Architecture` 区之后加硬约束清单**

在 `## Recipe Format` 之前插入：

```markdown
## Hard Constraints

> 以下规则不可违反，任何修改都必须遵守：

1. **所有 CLI 命令通过 `act` 入口**：不直接调用 trainers/ 下的模块
2. **Recipe 变更需通过 schema 验证**：修改 recipe 后必须跑 `make validate-schema`
3. **DB 操作通过 results/db.py**：不直接写 SQL 操作 results.db
4. **本地环境不跑 GPU 训练**：训练命令只在远程 torch 上执行
5. **新增 trainer 必须注册到 registry**：`trainers/registry.py` 是唯一调度入口
6. **测试不能依赖外部服务**：需要 SWE-bench/vLLM 的测试用 `pytest.importorskip` 跳过
```

- [ ] **Step 3: 验证 CLAUDE.md 格式正确**

Run: `grep -c "^## " CLAUDE.md`
Expected: 节标题数量应比原来多 2（Verification Commands + Hard Constraints + First Session Checklist）

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "feat: add verification commands section and hard constraints to CLAUDE.md"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** 诊断报告中的每个子系统改进建议都有对应 Task
- [x] **Placeholder scan:** 无 TBD/TODO/placeholder，每步都有具体内容
- [x] **Type consistency:** 无跨 Task 的函数/变量名冲突（本 plan 不涉及跨模块类型）
- [x] **Dependencies:** Task 1-3 完全独立可并行；Task 4-5 修改 CLAUDE.md，建议串行
