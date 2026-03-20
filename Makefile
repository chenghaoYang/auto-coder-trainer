.PHONY: collect compose train report install dev test clean help

# Auto-Coder-Trainer — Research Operating System for Coding Agent Training
# Four entry points: collect → compose → train → report

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install base package
	pip install -e .

dev: ## Install with all dev dependencies
	pip install -e ".[all,dev]"

collect: ## Collect papers and methods (usage: make collect QUERY="coding agent training")
	python -m cli.main collect "$(QUERY)"

compose: ## Compose recipe from method atoms (usage: make compose ATOMS="swe-fuse,entropy-rl")
	python -m cli.main compose --atoms "$(ATOMS)"

train: ## Run training experiment (usage: make train RECIPE=recipes/examples/baseline-sft.recipe.json)
	python -m cli.main train "$(RECIPE)"

report: ## Generate experiment report (usage: make report EXP_ID=exp_001)
	python -m cli.main report --experiment-id "$(EXP_ID)"

test: ## Run tests
	python -m pytest tests/ -v

validate-schema: ## Validate example recipes against schema
	python -c "import json, jsonschema; s=json.load(open('recipes/schema/recipe.schema.json')); r=json.load(open('recipes/examples/baseline-sft.recipe.json')); jsonschema.validate(r, s); print('OK')"

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
