"""Collect command — gather papers, projects, and methods into method cards.

Bridges ARIS Research Plane skills (research-lit, arxiv) to produce
structured method atoms in recipes/registry/.
"""

import argparse
import datetime as dt
import json
import math
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


# ---------------------------------------------------------------------------
# Evidence scoring
# ---------------------------------------------------------------------------

# Keywords that signal relevance to the target tasks (SWE-bench, coding agents)
_RELEVANCE_KEYWORDS: list[str] = [
    "swe-bench", "coding agent", "code generation", "code repair",
    "automated software engineering", "program synthesis", "code editing",
    "humaneval", "mbpp", "software agent", "repository-level",
]

# Weights for the composite score
_WEIGHT_NOVELTY = 0.30
_WEIGHT_REPRODUCIBILITY = 0.35
_WEIGHT_RELEVANCE = 0.35


def _score_evidence(atom: dict, existing_tags: set[str] | None = None) -> dict:
    """Compute evidence quality scores for a single atom.

    Adds ``atom["evidence"]["scores"]`` with novelty, reproducibility,
    relevance, and composite scores (all 0-1).  Returns the scores dict.
    """
    if existing_tags is None:
        existing_tags = set()

    evidence = atom.setdefault("evidence", {})

    # --- novelty_score ---
    # Time-decay based on publication / update date.
    pub_date_str = (
        atom.get("evidence", {}).get("paper", {}).get("published")
        or atom.get("evidence", {}).get("repo", {}).get("updated_at")
        or ""
    )
    novelty = 0.5  # default when date is unknown
    if pub_date_str:
        try:
            # Handle both ISO 8601 variants (with or without 'Z', with 'T')
            cleaned = pub_date_str.replace("Z", "+00:00")
            pub_dt = dt.datetime.fromisoformat(cleaned)
            # Ensure timezone-aware for safe subtraction
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=dt.timezone.utc)
            now = dt.datetime.now(dt.timezone.utc)
            age_days = max((now - pub_dt).days, 0)
            # Exponential decay: half-life of ~180 days
            novelty = math.exp(-0.00385 * age_days)  # ln(2)/180 ≈ 0.00385
        except (ValueError, TypeError):
            pass

    # Bonus for tags that are NOT already in the registry
    atom_tags = set(atom.get("innovation_tags", []))
    if atom_tags:
        unique_ratio = len(atom_tags - existing_tags) / len(atom_tags)
        novelty = min(1.0, novelty + 0.15 * unique_ratio)

    # --- reproducibility_score ---
    repro = 0.0
    # Code repository present?
    if atom.get("source_projects"):
        repro += 0.40
    if atom.get("evidence", {}).get("repo", {}).get("html_url"):
        repro += 0.10
    # Dataset availability
    dataset_sources = atom.get("dataset", {}).get("sources", [])
    if dataset_sources:
        repro += 0.25
    # Clear methodology proxies: has trainer config, eval benchmarks
    if atom.get("trainer", {}).get("type"):
        repro += 0.15
    if atom.get("eval", {}).get("benchmarks"):
        repro += 0.10
    repro = min(1.0, repro)

    # --- relevance_score ---
    summary_text = f"{atom.get('title', '')} {atom.get('key_innovation', '')}".lower()
    matched = sum(1 for kw in _RELEVANCE_KEYWORDS if kw in summary_text)
    relevance = min(1.0, matched / max(len(_RELEVANCE_KEYWORDS) * 0.3, 1))

    # --- composite ---
    composite = (
        _WEIGHT_NOVELTY * novelty
        + _WEIGHT_REPRODUCIBILITY * repro
        + _WEIGHT_RELEVANCE * relevance
    )

    scores = {
        "novelty_score": round(novelty, 4),
        "reproducibility_score": round(repro, 4),
        "relevance_score": round(relevance, 4),
        "composite_score": round(composite, 4),
    }
    evidence["scores"] = scores
    return scores


# ---------------------------------------------------------------------------
# Research planner
# ---------------------------------------------------------------------------

# Canonical categories and their representative search terms
_CATEGORY_QUERY_MAP: dict[str, list[str]] = {
    "reward": ["reward model coding agent", "RLHF code generation", "GRPO software engineering"],
    "training": ["distillation coding agent", "SFT code LLM", "trajectory training software agent"],
    "eval": ["SWE-bench evaluation", "code benchmark LLM", "HumanEval MBPP evaluation"],
    "data": ["code trajectory dataset", "SWE-bench trajectories", "code generation training data"],
    "infrastructure": ["coding agent sandbox", "LLM agent orchestration", "code execution pipeline"],
}


def _plan_research_queries(
    query: str,
    existing_atoms: list[dict],
) -> list[dict]:
    """Analyze gaps in the existing registry and propose expanded queries.

    Returns a list of ``{query, rationale, target_category, priority}`` dicts
    sorted by descending priority.
    """
    # Count atoms per category
    category_counts: dict[str, int] = {}
    all_tags: set[str] = set()
    for atom in existing_atoms:
        cat = atom.get("category", "training")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        all_tags.update(atom.get("innovation_tags", []))

    total = max(sum(category_counts.values()), 1)

    plans: list[dict] = []
    for category, search_terms in _CATEGORY_QUERY_MAP.items():
        count = category_counts.get(category, 0)
        share = count / total
        # Under-represented if less than 10% share (or zero)
        if share < 0.10 or count == 0:
            priority = 1.0 - share  # higher priority for rarer categories
            # Pick the search term most related to the user's original query
            best_term = search_terms[0]
            for term in search_terms:
                if any(tok in term.lower() for tok in query.lower().split()):
                    best_term = term
                    break
            plans.append({
                "query": f"{query} {best_term}",
                "rationale": (
                    f"Category '{category}' is underrepresented "
                    f"({count}/{total} atoms, {share:.0%} share)"
                ),
                "target_category": category,
                "priority": round(priority, 3),
            })

    # Sort by priority descending
    plans.sort(key=lambda p: p["priority"], reverse=True)
    return plans


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def _filter_low_quality_atoms(
    atoms: list[dict],
    min_composite_score: float = 0.3,
) -> list[dict]:
    """Return only atoms whose composite evidence score meets the threshold.

    Atoms that have not been scored yet are kept (to avoid discarding
    legacy data that pre-dates the scoring system).
    """
    kept: list[dict] = []
    for atom in atoms:
        scores = atom.get("evidence", {}).get("scores")
        if scores is None:
            # Not scored — keep by default
            kept.append(atom)
            continue
        if scores.get("composite_score", 0.0) >= min_composite_score:
            kept.append(atom)
    return kept


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
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
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


def _existing_innovation_tags(registry: dict) -> set[str]:
    """Gather all innovation tags already present in the registry."""
    tags: set[str] = set()
    for atom in registry.get("atoms", []):
        if isinstance(atom, dict):
            tags.update(atom.get("innovation_tags", []))
    return tags


def _score_and_filter_atoms(
    atoms: list[dict],
    existing_tags: set[str],
    min_composite_score: float,
) -> list[dict]:
    """Score every atom and discard those below the quality threshold."""
    for atom in atoms:
        _score_evidence(atom, existing_tags=existing_tags)

    before = len(atoms)
    kept = _filter_low_quality_atoms(atoms, min_composite_score=min_composite_score)
    removed = before - len(kept)
    if removed:
        print(f"[collect] Filtered out {removed} low-quality atom(s) "
              f"(below composite score {min_composite_score})")
    return kept


def run_collect(args: argparse.Namespace) -> None:
    """Execute the collect pipeline.

    Pipeline:
        1. Search arXiv/Scholar for papers matching query
        2. Extract method descriptions from each paper
        3. Structure as method atom cards
        4. Score evidence and filter low-quality atoms
        5. Plan additional research queries for gap-filling (up to 2 rounds)
        6. Save to recipes/registry/method_atoms.json

    Full search requires ARIS research-lit and arxiv skill integration.
    This skeleton prints progress messages and loads/saves the registry.
    """
    query = args.query
    max_papers = getattr(args, "max_papers", 20)
    max_repos = getattr(args, "max_repos", min(10, max_papers))
    evidence_threshold: float = getattr(args, "evidence_threshold", 0.3)
    output_dir = Path(getattr(args, "output", str(REGISTRY_PATH.parent)))
    registry_path = _resolve_registry_path(output_dir)

    print(f"[collect] Starting collection for query: '{query}'")
    print(f"[collect] Max papers: {max_papers}")
    print(f"[collect] Max repos: {max_repos}")
    print(f"[collect] Evidence threshold: {evidence_threshold}")

    # Step 1: Load existing registry
    registry = _load_registry(registry_path)
    existing_count = len(registry.get("atoms", []))
    print(f"[collect] Loaded registry with {existing_count} existing atoms")

    # Step 2: Resolve offline/import mode first.
    imported_atoms: list[dict] = []
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
        # --- Primary online collection round ---
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

        # --- Score and filter primary atoms ---
        if imported_atoms:
            existing_tags = _existing_innovation_tags(registry)
            imported_atoms = _score_and_filter_atoms(
                imported_atoms, existing_tags, evidence_threshold,
            )

        if imported_atoms:
            print(f"[collect] Structured {len(imported_atoms)} atom(s) from online discovery")
            added = _merge_atoms(registry, imported_atoms)
            print(f"[collect] Merged {added} new atom(s) into registry")
            _print_summary(imported_atoms)
        else:
            print("[collect] No new atoms discovered from online sources.")

        # --- Research planner: up to 2 extra collection rounds ---
        all_atoms_so_far = registry.get("atoms", [])
        planned_queries = _plan_research_queries(query, all_atoms_so_far)

        max_extra_rounds = 2
        extra_round = 0
        for plan in planned_queries:
            if extra_round >= max_extra_rounds:
                break
            extra_round += 1
            eq = plan["query"]
            print(
                f"[collect] Research planner round {extra_round}/{max_extra_rounds}: "
                f"'{eq}' (target: {plan['target_category']}, "
                f"priority: {plan['priority']})"
            )
            print(f"[collect]   Rationale: {plan['rationale']}")
            try:
                extra_atoms, extra_meta = _collect_online_atoms(
                    eq,
                    max_papers=max(max_papers // 2, 5),
                    max_repos=max(max_repos // 2, 3),
                )
            except Exception as exc:
                print(f"[collect]   Extra round failed: {exc}")
                continue

            if extra_meta:
                # Only append to collections list; don't overwrite last_collection
                # (which should reflect the primary user query, not planner rounds)
                registry.setdefault("collections", []).append(extra_meta)
                if extra_meta.get("errors"):
                    for error in extra_meta["errors"]:
                        print(f"[collect]   Warning: {error}")

            if extra_atoms:
                existing_tags = _existing_innovation_tags(registry)
                extra_atoms = _score_and_filter_atoms(
                    extra_atoms, existing_tags, evidence_threshold,
                )
            if extra_atoms:
                added = _merge_atoms(registry, extra_atoms)
                print(f"[collect]   Added {added} atom(s) from planner round {extra_round}")
                imported_atoms.extend(extra_atoms)
            else:
                print(f"[collect]   No new atoms from planner round {extra_round}")

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
