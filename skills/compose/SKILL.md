---
name: compose
description: Compose method atoms into a training recipe. Use when user says "组合recipe", "compose recipe", "build training config", or wants to create a new experiment configuration.
argument-hint: [atom-names or description]
allowed-tools: Bash(*), Read, Write, Grep, Glob
---

# Compose — Training Recipe Builder

Compose a training recipe from: $ARGUMENTS

## Pipeline

1. **Load** — Read method atoms from `recipes/registry/method_atoms.json`
2. **Select** — Pick atoms matching the user's request (by name or description)
3. **Combine** — Merge selected atoms into a coherent Recipe IR:
   - Resolve conflicts (e.g., two different reward functions → composite)
   - Set model, dataset mix, trainer type, and backend
   - Configure evaluation benchmarks and seeds
   - Set budget constraints
4. **Validate** — Validate the composed recipe against `recipes/schema/recipe.schema.json`
5. **Save** — Write recipe to `recipes/examples/<name>.recipe.json`

## Constraints

- Every recipe MUST specify: id, name, model, dataset, trainer
- Default RL backend is veRL; default SFT backend is TRL
- Include at least one ablation study in every recipe
- Set reasonable budget defaults (24h for SFT, 48h for RL on A100)
- Seeds default to [42, 123, 456]
