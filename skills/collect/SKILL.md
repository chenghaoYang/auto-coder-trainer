---
name: collect
description: Collect and structure research papers, projects, and methods related to coding agent training. Use when user says "collect papers", "找论文", "search methods", or wants to build a method registry.
argument-hint: [research-query]
allowed-tools: Bash(*), Read, Write, Grep, Glob, WebSearch, WebFetch, Agent
---

# Collect — Research Paper & Method Harvester

Collect and structure research on: $ARGUMENTS

## Pipeline

1. **Search** — Query arXiv, Semantic Scholar, and GitHub for recent papers/projects on the given topic
2. **Filter** — Keep only papers relevant to coding agent training (SFT, RL, trajectory data, evaluation)
3. **Extract** — For each paper, extract a structured method card:
   - Paper metadata (title, authors, venue, date, arXiv ID)
   - Method name and category (data, training, reward, eval, infrastructure)
   - Key innovation (1-2 sentences)
   - Composable atoms (what building blocks does this method introduce?)
   - Dependencies (what does it require? models, data, compute)
   - Reported results (benchmarks, metrics, baselines)
4. **Register** — Append method cards to `recipes/registry/method_atoms.json`
5. **Report** — Output summary of collected methods with novelty assessment

## Output Format

Write method cards to `recipes/registry/method_atoms.json` following the atom schema.
Print a summary table to the user showing: paper | method | category | key innovation.

## Constraints

- Focus on 2025-2026 papers unless user specifies otherwise
- Prioritize: SWE-bench results, agentic coding, trajectory data, RL for code
- Skip survey papers unless specifically requested
- Verify arXiv IDs are real (use arxiv_fetch.py tool if available)
