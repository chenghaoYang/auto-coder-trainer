---
name: report
description: Generate a technical report from experiment results. Use when user says "生成报告", "write report", "summarize experiments", or wants to see experiment analysis.
argument-hint: [experiment-id or recipe-id]
allowed-tools: Bash(*), Read, Write, Grep, Glob
---

# Report — Technical Report Generator

Generate report for: $ARGUMENTS

## Pipeline

1. **Query** — Fetch experiment data from result DB
2. **Gather** — Collect all related ablations, verdicts, and baselines
3. **Analyze** — Compute:
   - Performance vs baseline (deltas, significance)
   - Ablation impact analysis
   - Cost efficiency (performance per GPU-hour)
   - Failure patterns (if any)
4. **Generate** — Write structured report:
   - Method description (from recipe source_papers)
   - Experimental setup (model, data, hyperparams)
   - Results table (main experiments + ablations)
   - Analysis and discussion
   - Conclusions and recommended next steps
5. **Output** — Save as Markdown and optionally LaTeX

## Output Formats

- `reports/<id>_report.md` — Markdown report
- `reports/<id>_report.tex` — LaTeX report (optional, for paper integration)

## Constraints

- Include raw numbers, not just "improved" — show actual metrics
- Always compare with baseline
- Flag any experiments that failed judge review
- Include compute cost analysis
