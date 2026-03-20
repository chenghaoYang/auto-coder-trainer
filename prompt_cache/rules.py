"""Prompt Cache Rules — codified cache safety constraints.

These rules are enforced by the PromptBuilder and can be used as a
pre-flight check before any API call. They encode the six principles
from the Anthropic prompt caching architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheRule:
    """A single cache safety rule."""
    id: str
    name: str
    description: str
    severity: str  # "critical" | "warning" | "info"

    def check(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Check if this rule is satisfied. Returns (passed, message)."""
        raise NotImplementedError


class StaticPrefixRule(CacheRule):
    """Rule 1: Static content must come before dynamic content.

    The prompt must be ordered: static system prompt + tools → project context
    → session context → conversation messages. Any dynamic content (timestamps,
    random values) in the static prefix breaks caching.
    """

    def __init__(self):
        super().__init__(
            id="CACHE-001",
            name="Static Prefix Ordering",
            description="Static content must precede dynamic content in the prompt",
            severity="critical",
        )

    def check(self, context: dict[str, Any]) -> tuple[bool, str]:
        system = context.get("system", [])
        for i, block in enumerate(system):
            text = block.get("text", "") if isinstance(block, dict) else str(block)
            # Check for common dynamic content in system prompt
            dynamic_patterns = [
                "current time",
                "current date",
                "datetime.now",
                "time.time",
                "random.",
                "uuid.",
            ]
            for pattern in dynamic_patterns:
                if pattern.lower() in text.lower():
                    return False, (
                        f"Block {i} in system prompt contains dynamic content '{pattern}'. "
                        f"Move dynamic info to a <system-reminder> in messages instead."
                    )
        return True, "OK"


class ToolStabilityRule(CacheRule):
    """Rule 3: Tool definitions must not change mid-session.

    Adding, removing, or reordering tools destroys the cache prefix.
    The tool set must be fixed at session start. Use deferred loading
    (ToolSearch) for rarely-used tools.
    """

    def __init__(self):
        super().__init__(
            id="CACHE-003",
            name="Tool Set Stability",
            description="Tool definitions must remain constant throughout the session",
            severity="critical",
        )

    def check(self, context: dict[str, Any]) -> tuple[bool, str]:
        prev_tools = context.get("previous_tools")
        curr_tools = context.get("current_tools")
        if prev_tools is None or curr_tools is None:
            return True, "OK — no comparison available"

        prev_names = sorted(t.get("name", "") for t in prev_tools)
        curr_names = sorted(t.get("name", "") for t in curr_tools)

        if prev_names != curr_names:
            added = set(curr_names) - set(prev_names)
            removed = set(prev_names) - set(curr_names)
            parts = []
            if added:
                parts.append(f"added: {added}")
            if removed:
                parts.append(f"removed: {removed}")
            return False, f"Tool set changed ({', '.join(parts)}). This destroys cache."

        return True, "OK"


class ModelConsistencyRule(CacheRule):
    """Rule 3b: Model must not switch mid-session.

    Prompt cache is bound to a specific model. Switching from e.g.,
    claude-opus-4-6 to claude-haiku-4-5-20251001 forces a complete cache rebuild.
    """

    def __init__(self):
        super().__init__(
            id="CACHE-003b",
            name="Model Consistency",
            description="Model must remain constant throughout the session",
            severity="critical",
        )

    def check(self, context: dict[str, Any]) -> tuple[bool, str]:
        prev_model = context.get("previous_model")
        curr_model = context.get("current_model")
        if prev_model and curr_model and prev_model != curr_model:
            return False, (
                f"Model switched from '{prev_model}' to '{curr_model}'. "
                f"This forces full cache rebuild and is MORE expensive."
            )
        return True, "OK"


class DynamicUpdateRule(CacheRule):
    """Rule 2: Use messages, not system prompt, for dynamic updates.

    When you need to inject time-varying information (file changes,
    timestamps, status updates), use <system-reminder> tags in the
    next user message or tool result. Never modify the system prompt.
    """

    def __init__(self):
        super().__init__(
            id="CACHE-002",
            name="Dynamic Update Via Messages",
            description="Dynamic info must be injected via messages, not system prompt",
            severity="critical",
        )

    def check(self, context: dict[str, Any]) -> tuple[bool, str]:
        prev_system = context.get("previous_system")
        curr_system = context.get("current_system")
        if prev_system is not None and curr_system is not None:
            if prev_system != curr_system:
                return False, (
                    "System prompt was modified between requests. "
                    "Use <system-reminder> in messages for dynamic updates."
                )
        return True, "OK"


class CompactionPrefixRule(CacheRule):
    """Rule 5: Compaction must share parent's exact prefix.

    When compacting (summarizing) a conversation, the new request must
    use the exact same system prompt, tools, and context structure.
    Only append the compaction summary at the end.
    """

    def __init__(self):
        super().__init__(
            id="CACHE-005",
            name="Compaction Prefix Sharing",
            description="Compacted sessions must share parent's prefix for cache reuse",
            severity="critical",
        )

    def check(self, context: dict[str, Any]) -> tuple[bool, str]:
        parent_hash = context.get("parent_prefix_hash")
        child_hash = context.get("child_prefix_hash")
        if parent_hash and child_hash and parent_hash != child_hash:
            return False, (
                "Compacted session has different prefix hash than parent. "
                "Cache prefix will not be shared — this defeats the purpose."
            )
        return True, "OK"


# All rules in evaluation order
ALL_RULES: list[CacheRule] = [
    StaticPrefixRule(),
    DynamicUpdateRule(),
    ToolStabilityRule(),
    ModelConsistencyRule(),
    CompactionPrefixRule(),
]


def run_preflight_check(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Run all cache safety rules as a pre-flight check.

    Args:
        context: Dict with keys like 'system', 'tools', 'previous_tools',
                 'current_tools', 'previous_model', 'current_model', etc.

    Returns:
        List of rule results: [{rule_id, name, passed, message, severity}]
    """
    results = []
    for rule in ALL_RULES:
        passed, message = rule.check(context)
        results.append({
            "rule_id": rule.id,
            "name": rule.name,
            "passed": passed,
            "message": message,
            "severity": rule.severity,
        })
    return results


def has_critical_violations(context: dict[str, Any]) -> bool:
    """Quick check: are there any critical rule violations?"""
    results = run_preflight_check(context)
    return any(not r["passed"] and r["severity"] == "critical" for r in results)
