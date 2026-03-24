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


def _merge_atom(recipe: dict, atom: dict, *, trainer_locked: bool) -> bool:
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

    atom_trainer = atom.get("trainer", {})
    is_reward_atom = atom.get("category") == "reward"
    if atom_trainer:
        if "reward" in atom_trainer:
            recipe["trainer"]["reward"] = atom_trainer["reward"]
        if not (is_reward_atom and trainer_locked):
            if "type" in atom_trainer:
                recipe["trainer"]["type"] = atom_trainer["type"]
            if "backend" in atom_trainer:
                recipe["trainer"]["backend"] = atom_trainer["backend"]
        if not is_reward_atom and any(key in atom_trainer for key in ("type", "backend")):
            trainer_locked = True

    # Merge eval benchmarks / metrics
    for bench in atom.get("eval", {}).get("benchmarks", []):
        if bench not in recipe["eval"]["benchmarks"]:
            recipe["eval"]["benchmarks"].append(bench)
    for metric in atom.get("eval", {}).get("metrics", []):
        if metric not in recipe["eval"]["metrics"]:
            recipe["eval"]["metrics"].append(metric)

    # Merge ablation specs
    for ablation in atom.get("ablation", []):
        if ablation not in recipe["ablation"]:
            recipe["ablation"].append(ablation)
    return trainer_locked


def _atom_merge_priority(atom: dict) -> tuple[int, str]:
    """Merge reward-only atoms after structural recipe atoms."""
    is_reward_atom = atom.get("category") == "reward"
    return (1 if is_reward_atom else 0, atom.get("name", ""))


def _looks_like_qwen3_coder(model_name: str) -> bool:
    lowered = model_name.lower()
    return "qwen/qwen3" in lowered or "qwen/qwen3.5" in lowered


def _apply_workflow_defaults(recipe: dict) -> None:
    """Apply pragmatic defaults for the dialogue-driven SWE-Lego workflow."""
    model_base = recipe.get("model", {}).get("base", "")
    trainer = recipe.get("trainer", {})
    trainer_type = trainer.get("type", "sft")
    current_backend = trainer.get("backend", "trl")
    benchmarks = recipe.get("eval", {}).get("benchmarks", [])
    has_swe_bench = any(str(bench).startswith("swe-bench") for bench in benchmarks)

    if (
        trainer_type == "sft"
        and has_swe_bench
        and current_backend in {"trl", "swe_lego"}
        and _looks_like_qwen3_coder(model_base)
    ):
        trainer["backend"] = "swe_lego"
        recipe["model"]["adapter"] = "full"

        normalized_benchmarks = [
            "swe-bench-verified" if str(bench).startswith("swe-bench") else bench
            for bench in benchmarks
        ] or ["swe-bench-verified"]
        recipe["eval"]["benchmarks"] = list(dict.fromkeys(normalized_benchmarks))

        params = trainer.setdefault("params", {})
        defaults = {
            "lr": 1e-4,
            "epochs": 4,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_length": 131072,
            "warmup_ratio": 0.1,
            "lr_scheduler": "cosine",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "turn_mask": True,
            "rope_scaling": "yarn",
            "flash_attn": "fa2",
            "liger_kernel": True,
            "gradient_checkpointing": True,
            "preprocessing_num_workers": 16,
            "dataloader_num_workers": 4,
        }
        defaults["template"] = "qwen3" if "qwen3.5" in model_base.lower() else "qwen3_nothink"
        defaults["deepspeed"] = "z2_offload"
        for key, value in defaults.items():
            params.setdefault(key, value)

        budget = recipe.setdefault("budget", {})
        budget.setdefault("max_gpu_hours", 96)
        budget.setdefault("gpu_type", "1xH200-141GB")
        slurm = budget.setdefault("slurm", {})
        slurm_defaults = {
            "partition": "gpu",
            "nodes": 1,
            "gpus_per_node": 1,
            "cpus_per_task": 16,
            "mem": "256G",
            "time": "72:00:00",
        }
        for key, value in slurm_defaults.items():
            slurm.setdefault(key, value)

        if not recipe.get("ablation") and "lr" in params:
            center_lr = float(params["lr"])
            lr_values = [center_lr / 10.0, center_lr, center_lr * 5.0]
            deduped_values = []
            for candidate in lr_values:
                rounded = float(f"{candidate:.10g}")
                if rounded not in deduped_values:
                    deduped_values.append(rounded)
            recipe["ablation"] = [
                {
                    "name": "lr_sweep",
                    "variable": "trainer.params.lr",
                    "values": deduped_values,
                }
            ]


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
    trainer_type_override = getattr(args, "trainer_type", None)
    backend_override = getattr(args, "backend", None)

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

    trainer_locked = False
    for atom in sorted(selected, key=_atom_merge_priority):
        trainer_locked = _merge_atom(recipe, atom, trainer_locked=trainer_locked)
        print(f"[compose]   Merged atom: {atom.get('name', '?')}")

    if not selected:
        print("[compose] No atoms were matched — recipe uses defaults only.")

    if trainer_type_override:
        recipe["trainer"]["type"] = trainer_type_override
    if backend_override:
        recipe["trainer"]["backend"] = backend_override

    _apply_workflow_defaults(recipe)

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
        output_path = "recipes/composed.recipe.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(recipe, f, indent=2)
    print(f"[compose] Recipe written to {output_path}")
    print("[compose] Done.")
