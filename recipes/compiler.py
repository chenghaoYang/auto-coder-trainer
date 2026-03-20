"""Recipe Compiler — translates Recipe IR JSON into executable training configurations.

Usage:
    python3 -m recipes.compiler recipes/examples/baseline-sft.recipe.json

Reads a Recipe IR JSON file, validates it against the schema, and compiles it
into a training configuration that can be consumed by trainers/.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any


SCHEMA_PATH = Path(__file__).parent / "schema" / "recipe.schema.json"


@dataclass
class TrainingConfig:
    """Compiled training configuration ready for trainer consumption."""
    recipe_id: str
    trainer_type: str
    backend: str
    model_config: dict[str, Any]
    data_config: dict[str, Any]
    training_params: dict[str, Any]
    eval_config: dict[str, Any]
    ablation_configs: list[dict[str, Any]]
    budget: dict[str, Any]


def load_schema() -> dict:
    """Load the Recipe IR JSON Schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def normalize_recipe(recipe: dict) -> dict:
    """Return a copy of *recipe* with ``None`` values removed recursively.

    The schema treats a missing optional field as valid, but an explicit
    ``null`` usually is not.  Normalisation keeps the recipe compact and
    makes composed defaults schema-clean without forcing callers to care
    about every optional field.
    """

    def _strip_none(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cleaned
                for key, cleaned in (
                    (key, _strip_none(val))
                    for key, val in value.items()
                )
                if cleaned is not None
            }
        if isinstance(value, list):
            cleaned_items = []
            for item in value:
                cleaned = _strip_none(item)
                if cleaned is not None:
                    cleaned_items.append(cleaned)
            return cleaned_items
        return value

    return _strip_none(recipe)


def validate_recipe(recipe: dict, schema: dict) -> list[str]:
    """Validate a recipe against the schema. Returns list of errors (empty if valid)."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    errors = []
    for error in sorted(validator.iter_errors(recipe), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{path}: {error.message}")
    return errors


def compile_recipe(recipe: dict) -> TrainingConfig:
    """Compile a validated Recipe IR into an executable TrainingConfig."""
    recipe = normalize_recipe(recipe)
    trainer = recipe["trainer"]
    model = recipe["model"]
    dataset = recipe.get("dataset", {})
    eval_section = recipe.get("eval", {})
    budget = recipe.get("budget", {})
    ablation = recipe.get("ablation", [])

    # Determine backend: use explicit value, or default based on trainer type
    if "backend" in trainer:
        backend = trainer["backend"]
    else:
        backend = "trl" if trainer["type"] == "sft" else "verl"

    return TrainingConfig(
        recipe_id=recipe["id"],
        trainer_type=trainer["type"],
        backend=backend,
        model_config={
            "base": model["base"],
            "size": model.get("size"),
            "adapter": model.get("adapter", "full"),
        },
        data_config={
            "sources": dataset.get("sources", []),
            "filters": dataset.get("filters", []),
            "total_samples": dataset.get("total_samples"),
        },
        training_params={
            **trainer.get("params", {}),
            **({"reward": trainer["reward"]} if "reward" in trainer else {}),
        },
        eval_config={
            "benchmarks": eval_section.get("benchmarks", []),
            "metrics": eval_section.get("metrics", []),
            "seeds": eval_section.get("seeds", [42, 123, 456]),
        },
        ablation_configs=ablation,
        budget=budget,
    )


def main():
    """CLI entry point for recipe compilation."""
    if len(sys.argv) < 2:
        print(f"Usage: python3 -m recipes.compiler <recipe.json>")
        sys.exit(1)

    recipe_path = Path(sys.argv[1])
    with open(recipe_path) as f:
        recipe = json.load(f)

    recipe = normalize_recipe(recipe)
    schema = load_schema()
    errors = validate_recipe(recipe, schema)
    if errors:
        print(f"Validation errors: {errors}")
        sys.exit(1)

    config = compile_recipe(recipe)
    print(f"Compiled recipe '{config.recipe_id}' -> {config.backend}/{config.trainer_type}")


if __name__ == "__main__":
    main()
