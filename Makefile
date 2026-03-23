.PHONY: collect compose train report status install dev test clean help validate-schema benchmark-quick benchmark-standard benchmark-full result-card swe-lego swe-lego-dry swe-lego-import

PYTHON ?= python3

# Auto-Coder-Trainer — Research Operating System for Coding Agent Training
# Five entry points: collect → compose → train → report → status

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install base package
	$(PYTHON) -m pip install -e .

dev: ## Install with all dev dependencies
	$(PYTHON) -m pip install -e ".[all,dev]"

collect: ## Collect from arXiv/GitHub or import atoms (usage: make collect QUERY="coding agent training")
	$(PYTHON) -m cli.main collect "$(QUERY)"

compose: ## Compose recipe from method atoms (usage: make compose ATOMS="swe-fuse,entropy-rl")
	$(PYTHON) -m cli.main compose --atoms "$(ATOMS)"

train: ## Run training experiment (usage: make train RECIPE=recipes/examples/baseline-sft.recipe.json)
	$(PYTHON) -m cli.main train "$(RECIPE)"

report: ## Generate experiment report (usage: make report EXP_ID=exp_001)
	$(PYTHON) -m cli.main report --experiment-id "$(EXP_ID)"

status: ## Show tracked experiments and open tasks (usage: make status RECIPE_ID=recipe-demo)
	$(PYTHON) -m cli.main status $(if $(RECIPE_ID),--recipe-id "$(RECIPE_ID)",)

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

validate-schema: ## Validate example recipes against schema
	$(PYTHON) -c "import json, jsonschema; s=json.load(open('recipes/schema/recipe.schema.json')); r=json.load(open('recipes/examples/baseline-sft.recipe.json')); jsonschema.validate(r, s); print('OK')"

benchmark-quick: ## Run quick validation benchmark (dry-run, ~1 min)
	$(PYTHON) -m benchmarks.suite --suite quick --dry-run --output-dir outputs/benchmarks

benchmark-standard: ## Run standard benchmark suite (SFT baseline)
	$(PYTHON) -m benchmarks.suite --suite standard --output-dir outputs/benchmarks $(if $(GPU_TYPE),--gpu-type "$(GPU_TYPE)",)

benchmark-full: ## Run full benchmark suite (SFT + RL + distill)
	$(PYTHON) -m benchmarks.suite --suite full --output-dir outputs/benchmarks $(if $(GPU_TYPE),--gpu-type "$(GPU_TYPE)",)

result-card: ## Generate result card from experiment (usage: make result-card EXP_ID=exp_001)
	$(PYTHON) -c "from benchmarks.result_card import generate_result_card, render_result_card_markdown; from results.db import ResultDB; db = ResultDB(); db.connect(); card = generate_result_card('$(EXP_ID)', db); print(render_result_card_markdown(card)); db.close()"

swe-lego: ## Run SWE-Lego 8B full pipeline (generate + submit SLURM)
	$(PYTHON) -m cli.main train recipes/examples/swe-lego-8b.recipe.json --output-dir outputs/

swe-lego-dry: ## Dry-run SWE-Lego 8B (generate scripts only, no SLURM submit)
	$(PYTHON) -m cli.main train recipes/examples/swe-lego-8b.recipe.json --output-dir outputs/ --dry-run

swe-lego-import: ## Import SWE-Lego results into DB and generate report
	$(PYTHON) -m cli.main train recipes/examples/swe-lego-8b.recipe.json --import-results outputs/swe-lego-qwen3-8b/swe_lego/

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
