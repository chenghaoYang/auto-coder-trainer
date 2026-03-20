"""Collect command — gather papers, projects, and methods into method cards.

Bridges ARIS Research Plane skills (research-lit, arxiv) to produce
structured method atoms in recipes/registry/.
"""

import argparse
import json
from pathlib import Path


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "recipes" / "registry" / "method_atoms.json"


def _resolve_registry_path(output: Path) -> Path:
    """Resolve the on-disk registry file from a user-supplied output path."""
    if output.suffix in {".json", ".jsonl"}:
        return output
    return output / "method_atoms.json"


def _load_registry(path: Path) -> dict:
    """Load the method atoms registry, creating a skeleton if absent."""
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("atoms", [])
                return data
        except json.JSONDecodeError:
            print(f"[collect] Warning: registry at {path} is not valid JSON; recreating it.")
    return {"atoms": []}


def _load_atoms_from_source(source: str) -> list[dict]:
    """Load atoms from a local path or inline JSON blob."""
    candidate = Path(source)

    if candidate.exists():
        if candidate.is_dir():
            for name in ("method_atoms.json", "atoms.json"):
                nested = candidate / name
                if nested.exists():
                    candidate = nested
                    break
            if candidate.is_dir():
                return []
        if candidate.suffix == ".jsonl":
            atoms: list[dict] = []
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        atoms.append(json.loads(line))
            return atoms
        with open(candidate) as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "atoms" in payload:
            atoms = payload["atoms"]
            return atoms if isinstance(atoms, list) else []
        if isinstance(payload, list):
            return payload
        return []

    stripped = source.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        payload = json.loads(stripped)
        if isinstance(payload, dict) and "atoms" in payload:
            atoms = payload["atoms"]
            return atoms if isinstance(atoms, list) else []
        if isinstance(payload, list):
            return payload
    return []


def _merge_atoms(registry: dict, imported_atoms: list[dict]) -> int:
    """Merge imported atoms into the registry and return the new atom count."""
    atoms = registry.setdefault("atoms", [])
    seen = {
        atom.get("name")
        for atom in atoms
        if isinstance(atom, dict) and atom.get("name")
    }

    added = 0
    for atom in imported_atoms:
        if not isinstance(atom, dict):
            continue
        name = atom.get("name")
        if not name or name in seen:
            continue
        atoms.append(atom)
        seen.add(name)
        added += 1
    return added


def run_collect(args: argparse.Namespace) -> None:
    """Execute the collect pipeline.

    Pipeline:
        1. Search arXiv/Scholar for papers matching query
        2. Extract method descriptions from each paper
        3. Structure as method atom cards
        4. Save to recipes/registry/method_atoms.json

    Full search requires ARIS research-lit and arxiv skill integration.
    This skeleton prints progress messages and loads/saves the registry.
    """
    query = args.query
    max_papers = getattr(args, "max_papers", 20)
    output_dir = Path(getattr(args, "output", str(REGISTRY_PATH.parent)))
    registry_path = _resolve_registry_path(output_dir)

    print(f"[collect] Starting collection for query: '{query}'")
    print(f"[collect] Max papers: {max_papers}")

    # Step 1: Load existing registry
    registry = _load_registry(registry_path)
    existing_count = len(registry.get("atoms", []))
    print(f"[collect] Loaded registry with {existing_count} existing atoms")

    # Step 2: Resolve offline/import mode first.
    imported_atoms = []
    import_source = None
    try:
        imported_atoms = _load_atoms_from_source(query)
        if imported_atoms:
            import_source = query
    except Exception as exc:
        print(f"[collect] Warning: could not import atoms from '{query}': {exc}")
        imported_atoms = []

    if imported_atoms:
        print(f"[collect] Imported {len(imported_atoms)} atom(s) from: {import_source}")
        added = _merge_atoms(registry, imported_atoms)
        print(f"[collect] Merged {added} new atom(s) into registry")
    else:
        # Step 3: Offline research stub.
        print(f"[collect] Searching arXiv / Scholar for: '{query}' ...")
        print("[collect]   NOTE: Full paper search requires ARIS integration (research-lit, arxiv skills).")
        print("[collect]   Use a local registry JSON / JSONL path here to import atoms offline.")
        print("[collect] Structuring method atoms ...")
        print("[collect]   No new atoms to add in offline mode.")

    # Step 4: Save registry (write back even if unchanged, to ensure file exists)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    total_count = len(registry.get("atoms", []))
    print(f"[collect] Registry saved to {registry_path}  ({total_count} atoms)")

    if not imported_atoms:
        print("[collect] Done. To populate atoms automatically, pass a local registry file or integrate ARIS research skills.")
    else:
        print("[collect] Done. Imported atoms are ready for compose/train.")
