"""Collect command — gather papers, projects, and methods into method cards.

Bridges ARIS Research Plane skills (research-lit, arxiv) to produce
structured method atoms in recipes/registry/.
"""

import argparse
import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "recipes" / "registry" / "method_atoms.json"
GITHUB_SEARCH_API = "https://api.github.com/search/repositories"


def _slugify(value: str, *, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:limit].strip("-") or "unnamed-atom"


def _infer_category(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("reward", "entropy", "advantage", "grpo", "ppo", "rlhf")):
        return "reward"
    if any(token in lowered for token in ("distill", "distillation", "teacher", "student")):
        return "training"
    if any(token in lowered for token in ("benchmark", "evaluation", "judge", "pass@k", "swe-bench", "humaneval", "mbpp")):
        return "eval"
    if any(token in lowered for token in ("dataset", "trajectory", "corpus", "data mixture", "distillation data")):
        return "data"
    if any(token in lowered for token in ("cache", "orchestration", "pipeline", "sandbox", "launcher", "infra")):
        return "infrastructure"
    return "training"


def _infer_trainer(text: str) -> dict[str, Any]:
    lowered = text.lower()
    trainer: dict[str, Any] = {"params": {}}
    if any(token in lowered for token in ("distill", "distillation", "teacher model", "student model", "trajectory distillation")):
        trainer["type"] = "distill"
        if "open-r1" in lowered or "openr1" in lowered:
            trainer["backend"] = "openr1"
        elif "agent distillation" in lowered:
            trainer["backend"] = "agent_distill"
        elif "redi" in lowered or "reinforcement distillation" in lowered:
            trainer["backend"] = "redi"
        else:
            trainer["backend"] = "trl"
        return trainer
    if any(token in lowered for token in ("grpo", "ppo", "reinforcement", "rlhf")):
        trainer["type"] = "grpo" if "grpo" in lowered else "rl"
        trainer["backend"] = "verl"
    else:
        trainer["type"] = "sft"
        trainer["backend"] = "trl"
    if "entropy" in lowered:
        trainer["reward"] = {"type": "entropy_aware"}
    return trainer


def _infer_eval(text: str) -> dict[str, Any]:
    lowered = text.lower()
    benchmarks: list[str] = []
    if "swe-rebench" in lowered:
        benchmarks.append("swe-rebench")
    if "swe-bench verified" in lowered or "swe-bench-verified" in lowered:
        benchmarks.append("swe-bench-verified")
    if "swe-bench" in lowered and "swe-bench-verified" not in benchmarks:
        benchmarks.append("swe-bench-lite")
    if "humaneval" in lowered:
        benchmarks.append("humaneval")
    if "mbpp" in lowered:
        benchmarks.append("mbpp")
    metrics = []
    if any(token in lowered for token in ("resolve", "swe-bench", "rebench")):
        metrics.append("resolve_rate")
    if any(token in lowered for token in ("pass@1", "humaneval", "mbpp")):
        metrics.append("pass@1")
    return {
        "benchmarks": benchmarks,
        "metrics": metrics,
        "seeds": [42, 123, 456],
    }


def _infer_dataset_sources(text: str) -> list[dict[str, Any]]:
    lowered = text.lower()
    sources: list[dict[str, Any]] = []
    if "swe-rebench" in lowered:
        sources.append(
            {
                "name": "swe-rebench-trajectories",
                "path": "openhands/swe-rebench-openhands-trajectories",
                "mix_weight": 1.0,
            }
        )
    if "swe-bench" in lowered and not any(source["name"] == "swe-bench-trajectories" for source in sources):
        sources.append(
            {
                "name": "swe-bench-trajectories",
                "path": "bigcode/swe-bench-trajectories",
                "mix_weight": 1.0,
            }
        )
    return sources


def _extract_innovation_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = []
    for tag in (
        "trajectory",
        "grpo",
        "ppo",
        "entropy",
        "reward",
        "swe-bench",
        "humaneval",
        "benchmark",
        "cache",
        "sandbox",
        "distillation",
        "sft",
        "rlhf",
    ):
        if tag in lowered:
            tags.append(tag)
    return tags


def _suggest_ablations(text: str) -> list[dict[str, Any]]:
    lowered = text.lower()
    ablations: list[dict[str, Any]] = []
    if "entropy" in lowered:
        ablations.append(
            {
                "name": "reward_type",
                "variable": "trainer.reward.type",
                "values": ["binary_pass", "entropy_aware"],
            }
        )
    if "lora" in lowered:
        ablations.append(
            {
                "name": "adapter",
                "variable": "model.adapter",
                "values": ["full", "lora"],
            }
        )
    return ablations


def _paper_to_atom(paper: dict[str, Any]) -> dict[str, Any]:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    summary_text = f"{title}. {abstract}".strip()
    return {
        "name": _slugify(title or paper.get("id", "paper")),
        "kind": "paper",
        "category": _infer_category(summary_text),
        "title": title,
        "source_papers": [paper.get("id")] if paper.get("id") else [],
        "source_projects": [],
        "key_innovation": abstract[:280] if abstract else title,
        "innovation_tags": _extract_innovation_tags(summary_text),
        "dependencies": {
            "benchmarks": _infer_eval(summary_text).get("benchmarks", []),
            "compute": "unknown",
        },
        "reported_results": [],
        "dataset": {"sources": _infer_dataset_sources(summary_text)},
        "trainer": _infer_trainer(summary_text),
        "eval": _infer_eval(summary_text),
        "ablation": _suggest_ablations(summary_text),
        "evidence": {"paper": paper},
    }


def _repo_to_atom(repo: dict[str, Any]) -> dict[str, Any]:
    description = repo.get("description") or ""
    title = repo.get("full_name") or repo.get("name") or "repo"
    summary_text = f"{title}. {description}".strip()
    return {
        "name": _slugify(title),
        "kind": "repo",
        "category": _infer_category(summary_text),
        "title": title,
        "source_papers": [],
        "source_projects": [repo.get("html_url")] if repo.get("html_url") else [],
        "key_innovation": description[:280] if description else title,
        "innovation_tags": _extract_innovation_tags(summary_text),
        "dependencies": {
            "license": repo.get("license"),
            "stars": repo.get("stargazers_count", 0),
        },
        "reported_results": [],
        "dataset": {"sources": _infer_dataset_sources(summary_text)},
        "trainer": _infer_trainer(summary_text),
        "eval": _infer_eval(summary_text),
        "ablation": _suggest_ablations(summary_text),
        "evidence": {"repo": repo},
    }


def _search_arxiv_papers(query: str, max_papers: int) -> list[dict[str, Any]]:
    from aris.tools.arxiv_fetch import search as arxiv_search

    return arxiv_search(query, max_results=max_papers)


def _search_github_repos(query: str, max_repos: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "q": f"{query} (coding agent OR swe-bench OR grpo OR rlhf OR sft)",
            "sort": "stars",
            "order": "desc",
            "per_page": max_repos,
        }
    )
    req = urllib.request.Request(
        f"{GITHUB_SEARCH_API}?{params}",
        headers={"User-Agent": "auto-coder-trainer/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    items = payload.get("items", [])
    repos: list[dict[str, Any]] = []
    for item in items:
        repos.append(
            {
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "html_url": item.get("html_url"),
                "description": item.get("description"),
                "stargazers_count": item.get("stargazers_count", 0),
                "updated_at": item.get("updated_at"),
                "license": item.get("license", {}).get("spdx_id") if isinstance(item.get("license"), dict) else None,
            }
        )
    return repos


def _collect_online_atoms(query: str, max_papers: int, max_repos: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    repos: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        papers = _search_arxiv_papers(query, max_papers=max_papers)
    except Exception as exc:
        errors.append(f"arXiv search failed: {exc}")

    try:
        repos = _search_github_repos(query, max_repos=max_repos)
    except Exception as exc:
        errors.append(f"GitHub search failed: {exc}")

    atoms = [_paper_to_atom(paper) for paper in papers] + [_repo_to_atom(repo) for repo in repos]
    metadata = {
        "query": query,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "papers_found": len(papers),
        "repos_found": len(repos),
        "errors": errors,
    }
    return atoms, metadata


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
                data.setdefault("collections", [])
                return data
        except json.JSONDecodeError:
            print(f"[collect] Warning: registry at {path} is not valid JSON; recreating it.")
    return {"atoms": [], "collections": []}


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
    seen_names = {
        atom.get("name")
        for atom in atoms
        if isinstance(atom, dict) and atom.get("name")
    }
    seen_papers = {
        paper
        for atom in atoms
        if isinstance(atom, dict)
        for paper in atom.get("source_papers", [])
    }
    seen_projects = {
        project
        for atom in atoms
        if isinstance(atom, dict)
        for project in atom.get("source_projects", [])
    }

    added = 0
    for atom in imported_atoms:
        if not isinstance(atom, dict):
            continue
        name = atom.get("name")
        paper_ids = [paper for paper in atom.get("source_papers", []) if paper]
        project_urls = [project for project in atom.get("source_projects", []) if project]
        duplicate = (
            not name
            or name in seen_names
            or any(paper in seen_papers for paper in paper_ids)
            or any(project in seen_projects for project in project_urls)
        )
        if duplicate:
            continue
        atoms.append(atom)
        seen_names.add(name)
        seen_papers.update(paper_ids)
        seen_projects.update(project_urls)
        added += 1
    return added


def _register_collection(registry: dict, metadata: dict[str, Any]) -> None:
    collections = registry.setdefault("collections", [])
    collections.append(metadata)
    registry["last_collection"] = metadata


def _print_summary(imported_atoms: list[dict[str, Any]]) -> None:
    if not imported_atoms:
        print("[collect] No atoms to summarize.")
        return
    print("[collect] Summary:")
    print("[collect]   type       | name                           | category       | innovation")
    print("[collect]   -----------+--------------------------------+----------------+------------------------------")
    for atom in imported_atoms[:10]:
        kind = str(atom.get("kind", "atom"))[:11].ljust(11)
        name = str(atom.get("name", "?"))[:30].ljust(30)
        category = str(atom.get("category", "?"))[:14].ljust(14)
        innovation = str(atom.get("key_innovation", ""))[:30]
        print(f"[collect]   {kind} | {name} | {category} | {innovation}")


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
    max_repos = getattr(args, "max_repos", min(10, max_papers))
    output_dir = Path(getattr(args, "output", str(REGISTRY_PATH.parent)))
    registry_path = _resolve_registry_path(output_dir)

    print(f"[collect] Starting collection for query: '{query}'")
    print(f"[collect] Max papers: {max_papers}")
    print(f"[collect] Max repos: {max_repos}")

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

    collection_metadata: dict[str, Any] | None = None
    if imported_atoms:
        print(f"[collect] Imported {len(imported_atoms)} atom(s) from: {import_source}")
        added = _merge_atoms(registry, imported_atoms)
        print(f"[collect] Merged {added} new atom(s) into registry")
    else:
        print(f"[collect] Searching arXiv for: '{query}' ...")
        print(f"[collect] Searching GitHub repos for: '{query}' ...")
        imported_atoms, collection_metadata = _collect_online_atoms(
            query,
            max_papers=max_papers,
            max_repos=max_repos,
        )
        if collection_metadata:
            _register_collection(registry, collection_metadata)
            if collection_metadata.get("errors"):
                for error in collection_metadata["errors"]:
                    print(f"[collect] Warning: {error}")
        if imported_atoms:
            print(f"[collect] Structured {len(imported_atoms)} atom(s) from online discovery")
            added = _merge_atoms(registry, imported_atoms)
            print(f"[collect] Merged {added} new atom(s) into registry")
            _print_summary(imported_atoms)
        else:
            print("[collect] No new atoms discovered from online sources.")

    # Step 4: Save registry (write back even if unchanged, to ensure file exists)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    total_count = len(registry.get("atoms", []))
    print(f"[collect] Registry saved to {registry_path}  ({total_count} atoms)")

    if not imported_atoms:
        print("[collect] Done. Registry unchanged.")
    else:
        print("[collect] Done. Imported atoms are ready for compose/train.")
