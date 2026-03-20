"""Collect command — gather papers, projects, and methods into method cards.

Bridges ARIS Research Plane skills (research-lit, arxiv) to produce
structured method atoms in recipes/registry/.
"""

import argparse
import json
from pathlib import Path


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "recipes" / "registry" / "method_atoms.json"


def _load_registry(path: Path) -> dict:
    """Load the method atoms registry, creating a skeleton if absent."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"atoms": []}


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

    print(f"[collect] Starting collection for query: '{query}'")
    print(f"[collect] Max papers: {max_papers}")

    # Step 1: Load existing registry
    registry_path = output_dir / "method_atoms.json" if output_dir.is_dir() else output_dir
    registry = _load_registry(registry_path)
    existing_count = len(registry.get("atoms", []))
    print(f"[collect] Loaded registry with {existing_count} existing atoms")

    # Step 2: Search for papers
    print(f"[collect] Searching arXiv / Scholar for: '{query}' ...")
    print("[collect]   NOTE: Full paper search requires ARIS integration (research-lit, arxiv skills).")
    print("[collect]   Connect ARIS skills to enable automated paper discovery and method extraction.")

    # Step 3: Extract methods (placeholder — requires ARIS)
    print("[collect] Extracting method descriptions from retrieved papers ...")
    print("[collect]   Skipped — no papers retrieved without ARIS connection.")

    # Step 4: Structure as atom cards
    print("[collect] Structuring method atoms ...")
    print("[collect]   No new atoms to add in offline mode.")

    # Step 5: Save registry (write back even if unchanged, to ensure file exists)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"[collect] Registry saved to {registry_path}  ({existing_count} atoms)")

    print("[collect] Done. To populate atoms, integrate ARIS research skills or add atoms manually.")
