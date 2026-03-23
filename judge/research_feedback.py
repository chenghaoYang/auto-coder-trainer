"""Research feedback generation from judge verdicts.

When the judge returns REJECT or NEEDS_RERUN, this module analyzes the
failure attribution and generates actionable research queries and recipe
modifications to feed back into the collect → compose loop.
"""

from __future__ import annotations

from typing import Any

from judge.judge import JudgementResult, Verdict


# ---------------------------------------------------------------------------
# Query / modification templates keyed by failure cause
# ---------------------------------------------------------------------------

_CAUSE_TO_QUERIES: dict[str, list[dict[str, Any]]] = {
    "out_of_memory": [
        {
            "query": "memory efficient training techniques LoRA QLoRA gradient checkpointing",
            "rationale": "Training failed with OOM — search for parameter-efficient methods",
            "priority": 1,
            "target_category": "training_technique",
        },
        {
            "query": "mixed precision training bfloat16 memory reduction LLM",
            "rationale": "Explore precision-reduction strategies to lower memory footprint",
            "priority": 2,
            "target_category": "training_technique",
        },
    ],
    "timeout": [
        {
            "query": "fast convergence training schedules warmup cosine annealing",
            "rationale": "Training timed out — look for faster convergence strategies",
            "priority": 1,
            "target_category": "training_technique",
        },
    ],
    "underfitting": [
        {
            "query": "learning rate scheduling warmup strategies for code LLMs",
            "rationale": "Model underfitting — explore learning rate and scheduling improvements",
            "priority": 1,
            "target_category": "hyperparameter",
        },
        {
            "query": "data augmentation techniques for code generation training",
            "rationale": "Underfitting may be caused by insufficient data diversity",
            "priority": 2,
            "target_category": "data",
        },
    ],
    "overfitting": [
        {
            "query": "regularisation techniques dropout weight decay for LLM fine-tuning",
            "rationale": "Model overfitting — search for regularisation approaches",
            "priority": 1,
            "target_category": "hyperparameter",
        },
        {
            "query": "data augmentation diverse training data code generation",
            "rationale": "Overfitting may indicate insufficient training data diversity",
            "priority": 2,
            "target_category": "data",
        },
    ],
    "reward_design": [
        {
            "query": "reward function design RLHF code generation reward shaping",
            "rationale": "Reward-related metrics degraded — search for better reward designs",
            "priority": 1,
            "target_category": "reward",
        },
        {
            "query": "reward clipping scaling normalisation reinforcement learning LLM",
            "rationale": "Explore reward engineering techniques to stabilise training",
            "priority": 2,
            "target_category": "reward",
        },
    ],
    "hyperparameter_mismatch": [
        {
            "query": "hyperparameter tuning best practices LLM fine-tuning",
            "rationale": "Metrics regressed without clear single cause — broad HP search",
            "priority": 1,
            "target_category": "hyperparameter",
        },
        {
            "query": "ablation study methodology training recipe optimisation",
            "rationale": "Need systematic ablation to isolate the problematic parameter",
            "priority": 2,
            "target_category": "training_technique",
        },
    ],
    "missing_eval": [
        {
            "query": "evaluation harness setup code generation benchmarks",
            "rationale": "No eval results produced — ensure eval pipeline is correct",
            "priority": 1,
            "target_category": "evaluation",
        },
    ],
    "training_error": [
        {
            "query": "common training failures LLM fine-tuning debugging",
            "rationale": "Training failed with an unclassified error",
            "priority": 1,
            "target_category": "training_technique",
        },
    ],
}

_CAUSE_TO_MODIFICATIONS: dict[str, list[dict[str, Any]]] = {
    "out_of_memory": [
        {
            "parameter_path": "training.per_device_train_batch_size",
            "current_value": None,
            "suggested_value": "halve current value (min 1)",
            "rationale": "Reduce batch size to lower peak memory usage",
        },
        {
            "parameter_path": "training.gradient_checkpointing",
            "current_value": None,
            "suggested_value": True,
            "rationale": "Enable gradient checkpointing to trade compute for memory",
        },
    ],
    "timeout": [
        {
            "parameter_path": "training.num_train_epochs",
            "current_value": None,
            "suggested_value": "halve current value (min 1)",
            "rationale": "Reduce epochs to fit within time budget",
        },
    ],
    "underfitting": [
        {
            "parameter_path": "training.learning_rate",
            "current_value": None,
            "suggested_value": "increase by 2-3x",
            "rationale": "Higher learning rate may improve convergence",
        },
        {
            "parameter_path": "training.num_train_epochs",
            "current_value": None,
            "suggested_value": "increase by 50%",
            "rationale": "More training epochs to allow the model to learn",
        },
    ],
    "overfitting": [
        {
            "parameter_path": "training.weight_decay",
            "current_value": None,
            "suggested_value": 0.1,
            "rationale": "Add or increase weight decay for regularisation",
        },
        {
            "parameter_path": "training.num_train_epochs",
            "current_value": None,
            "suggested_value": "reduce by 30-50%",
            "rationale": "Fewer epochs to prevent overfitting",
        },
    ],
    "reward_design": [
        {
            "parameter_path": "reward.scale",
            "current_value": None,
            "suggested_value": "review and recalibrate",
            "rationale": "Reward scaling may be misaligned with target metric",
        },
        {
            "parameter_path": "reward.clip_range",
            "current_value": None,
            "suggested_value": 0.2,
            "rationale": "Tighter clipping to stabilise reward-driven updates",
        },
    ],
    "hyperparameter_mismatch": [
        {
            "parameter_path": "training.learning_rate",
            "current_value": None,
            "suggested_value": "align with baseline value",
            "rationale": "Learning rate may have diverged from baseline configuration",
        },
    ],
}

# Causes that justify triggering a full new research collection cycle
_CAUSES_TRIGGERING_COLLECTION = {
    "reward_design",
    "hyperparameter_mismatch",
    "underfitting",
    "overfitting",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ResearchFeedback:
    """Generate research suggestions from a judge verdict.

    Given a :class:`JudgementResult` and the recipe that was judged, this
    class produces:
    - new literature search queries targeting the identified weakness
    - specific recipe parameter modifications
    - a recommendation on whether to trigger a new collection cycle
    """

    def suggest_research_queries(
        self,
        verdict: JudgementResult,
        recipe: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Analyze failure attribution and return prioritised search queries.

        Each query dict contains:
            query           – search string for the collect phase
            rationale       – why this query is relevant
            priority        – 1 (highest) to N
            target_category – method-atom category to look for
        """
        cause = self._extract_cause(verdict)
        queries = list(_CAUSE_TO_QUERIES.get(cause, []))

        # If no predefined queries match, generate a generic one from the
        # verdict reasoning so we always return *something* useful.
        if not queries:
            queries.append({
                "query": f"improve training recipe {cause}",
                "rationale": f"Judge reasoning: {verdict.reasoning}",
                "priority": 1,
                "target_category": "training_technique",
            })

        # Enrich queries with recipe context (model name, task type) so
        # that collect can produce more targeted results.
        model_name = recipe.get("model", recipe.get("base_model", ""))
        if model_name:
            for q in queries:
                if model_name.lower() not in q["query"].lower():
                    q["query"] += f" {model_name}"

        return queries

    def suggest_recipe_modifications(
        self,
        verdict: JudgementResult,
        recipe: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return concrete recipe parameter changes based on the failure.

        Each modification dict contains:
            parameter_path  – dot-separated path inside the recipe
            current_value   – value in the current recipe (or None)
            suggested_value – recommended new value
            rationale       – human-readable explanation
        """
        cause = self._extract_cause(verdict)
        templates = _CAUSE_TO_MODIFICATIONS.get(cause, [])

        modifications: list[dict[str, Any]] = []
        for tpl in templates:
            mod = dict(tpl)
            # Resolve current_value from the recipe if possible
            mod["current_value"] = self._resolve_param(
                recipe, mod["parameter_path"],
            )
            modifications.append(mod)

        return modifications

    def should_trigger_new_collection(
        self,
        verdict: JudgementResult,
    ) -> bool:
        """Decide whether the failure warrants a new research collection cycle.

        Collection is expensive, so we only trigger it when the failure cause
        is likely to benefit from new literature (e.g., reward design issues
        or persistent hyperparameter mismatches).
        """
        if verdict.verdict not in (Verdict.REJECT, Verdict.NEEDS_RERUN):
            return False

        cause = self._extract_cause(verdict)
        return cause in _CAUSES_TRIGGERING_COLLECTION

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cause(verdict: JudgementResult) -> str:
        """Extract the likely_cause tag from the verdict reasoning."""
        reasoning = verdict.reasoning.lower()

        # The judge embeds "Likely cause: <tag>." in reasoning when
        # attribution runs.  Parse it out.
        if "likely cause:" in reasoning:
            after = reasoning.split("likely cause:")[-1].strip()
            cause = after.split(".")[0].strip()
            return cause

        # Fallback heuristics based on keywords in reasoning
        if "out of memory" in reasoning or "oom" in reasoning:
            return "out_of_memory"
        if "timeout" in reasoning:
            return "timeout"
        if "seed" in reasoning:
            return "seed_variance"
        if "duplicate" in reasoning:
            return "duplicate"
        if "ablation" in reasoning:
            return "missing_ablation"

        return "unknown"

    @staticmethod
    def _resolve_param(recipe: dict[str, Any], dotted_path: str) -> Any:
        """Walk a dotted path like 'training.learning_rate' into the recipe."""
        parts = dotted_path.split(".")
        current: Any = recipe
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current
