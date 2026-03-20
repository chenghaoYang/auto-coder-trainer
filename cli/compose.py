"""Compose command — assemble method atoms into a training recipe.

Takes selected method atoms and composes them into a valid Recipe IR JSON,
validated against the schema.
"""

import argparse
import json
from pathlib import Path


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "recipes" / "registry" / "method_atoms.json"


def _load_registry() -> dict:
    """Load the method atoms registry."""
    if not REGISTRY_PATH.exists():
        print(f"[compose] Warning: registry not found at {REGISTRY_PATH}")
        return {"atoms": []}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _default_recipe(model: str) -> dict:
    """Return a minimal recipe skeleton."""
    return {
        "id": "",
        "name": "",
        "version": "1.0",
        "source_papers": [],
        "model": {
            "base": model,
            "adapter": "lora",
        },
        "dataset": {
            "sources": [],
            "filters": [],
        },
        "trainer": {
            "type": "sft",
            "backend": "trl",
            "params": {},
        },
        "eval": {
            "benchmarks": [],
            "metrics": [],
            "seeds": [42, 123, 456],
        },
        "ablation": [],
        "budget": {},
    }


def _merge_atom(recipe: dict, atom: dict) -> None:
    """Merge a single method atom into a recipe, mutating *recipe* in-place."""
    # Accumulate source papers
    for paper in atom.get("source_papers", []):
        if paper not in recipe["source_papers"]:
            recipe["source_papers"].append(paper)

    # Merge dataset sources
    for src in atom.get("dataset", {}).get("sources", []):
        if src not in recipe["dataset"]["sources"]:
            recipe["dataset"]["sources"].append(src)

    # Merge trainer params (atom params override defaults)
    for key, val in atom.get("trainer", {}).get("params", {}).items():
        recipe["trainer"]["params"][key] = val

    # Override trainer type/backend if atom specifies them
    if "trainer" in atom:
        if "type" in atom["trainer"]:
            recipe["trainer"]["type"] = atom["trainer"]["type"]
        if "backend" in atom["trainer"]:
            recipe["trainer"]["backend"] = atom["trainer"]["backend"]
        if "reward" in atom["trainer"]:
            recipe["trainer"]["reward"] = atom["trainer"]["reward"]

    # Merge eval benchmarks / metrics
    for bench in atom.get("eval", {}).get("benchmarks", []):
        if bench not in recipe["eval"]["benchmarks"]:
            recipe["eval"]["benchmarks"].append(bench)
    for metric in atom.get("eval", {}).get("metrics", []):
        if metric not in recipe["eval"]["metrics"]:
            recipe["eval"]["metrics"].append(metric)

    # Merge ablation specs
    recipe["ablation"].extend(atom.get("ablation", []))


def _build_minimal_recipe_id(atom_names: list[str]) -> str:
    """Build a stable recipe identifier from the selected atoms."""
    if not atom_names:
        return "recipe-default-001"
    slug = "-".join(atom_names)
    return "recipe-" + slug[:60] + "-001"


def run_compose(args: argparse.Namespace) -> None:
    """Execute the compose pipeline.

    Pipeline:
        1. Load method atoms from registry
        2. Select atoms by name
        3. Compose into Recipe IR JSON
        4. Validate against schema
        5. Write output recipe file
    """
    atom_names = [a.strip() for a in args.atoms.split(",") if a.strip()]
    model = getattr(args, "model", "Qwen/Qwen2.5-Coder-7B-Instruct")
    output_path = getattr(args, "output", None)

    print(f"[compose] Requested atoms: {atom_names}")
    print(f"[compose] Base model: {model}")

    # Step 1: Load registry
    registry = _load_registry()
    atoms_by_name = {a["name"]: a for a in registry.get("atoms", []) if "name" in a}
    print(f"[compose] Registry contains {len(atoms_by_name)} atoms")

    # Step 2: Select requested atoms
    selected = []
    for name in atom_names:
        if name in atoms_by_name:
            selected.append(atoms_by_name[name])
            print(f"[compose]   Found atom: {name}")
        else:
            print(f"[compose]   Warning: atom '{name}' not found in registry — skipping")

    # Step 3: Build recipe (start from template or defaults)
    recipe = _default_recipe(model)
    recipe_id = _build_minimal_recipe_id(atom_names)
    recipe["id"] = recipe_id
    recipe["name"] = "Composed: " + ", ".join(atom_names)

    for atom in selected:
        _merge_atom(recipe, atom)
        print(f"[compose]   Merged atom: {atom.get('name', '?')}")

    if not selected:
        print("[compose] No atoms were matched — recipe uses defaults only.")

    try:
        from recipes.compiler import normalize_recipe

        recipe = normalize_recipe(recipe)
    except Exception as exc:
        print(f"[compose] Warning: could not normalize recipe ({exc})")

    # Step 4: Validate using compiler (if available)
    try:
        from recipes.compiler import load_schema, validate_recipe

        schema = load_schema()
        errors = validate_recipe(recipe, schema)
        if errors:
            print(f"[compose] Validation warnings ({len(errors)}):")
            for err in errors:
                print(f"[compose]   - {err}")
        else:
            print("[compose] Recipe passes schema validation.")
    except Exception as exc:
        print(f"[compose] Schema validation skipped: {exc}")

    # Step 5: Save recipe
    if output_path is None:
        output_path = f"recipes/examples/{recipe_id}.recipe.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(recipe, f, indent=2)
    print(f"[compose] Recipe written to {output_path}")
    print("[compose] Done.")
