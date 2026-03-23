"""Cache-safe context compaction.

When the context window fills up, we need to compress/summarize the
conversation. The key insight: compaction MUST use the exact same
system prompt, tools, and context structure as the parent session.
Only append the compaction instruction at the end.

This ensures the compacted session reuses the parent's cached prefix,
avoiding an expensive full cache rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt_cache.builder import PromptBuilder, CacheBreakpoint


COMPACTION_INSTRUCTION = """\
You are continuing a conversation that has been compacted to fit within \
the context window. Below is a summary of the conversation so far. \
Continue from where the summary ends.

<conversation-summary>
{summary}
</conversation-summary>

Key context preserved:
- All file modifications and their current state
- Outstanding tasks and their status
- Decisions made and their rationale
- Any errors encountered and their resolution
"""


@dataclass
class CacheSafeCompactor:
    """Compacts conversation context while preserving cache prefix.

    The critical rule: the compacted prompt must have EXACTLY the same
    prefix (system prompt + tools + project context) as the parent.
    Only the conversation messages are compressed, and the compaction
    instruction is appended at the end.

    Usage:
        compactor = CacheSafeCompactor(parent_builder)

        # When context window is filling up
        if compactor.should_compact(current_tokens, max_tokens):
            new_builder = compactor.compact(
                summary="User asked to implement feature X. We created files A, B, C..."
            )
            # new_builder shares parent's cache prefix
            assert new_builder.is_cache_compatible(parent_builder)
    """

    parent_builder: PromptBuilder
    compact_threshold: float = 0.85  # Compact when context is 85% full

    def should_compact(self, current_tokens: int, max_tokens: int) -> bool:
        """Check if compaction is needed based on token usage."""
        return current_tokens / max_tokens >= self.compact_threshold

    def compact(self, summary: str) -> PromptBuilder:
        """Create a compacted prompt builder that shares the parent's cache prefix.

        The returned builder has:
        - Identical Layer 0 (system prompt + tools) — shares parent cache
        - Identical Layer 1 (project context) — shares parent cache
        - Identical Layer 2 (session context) — shares parent cache
        - New Layer 3 (compaction summary + continuation) — dynamic

        This means the API will get a cache HIT on the entire prefix,
        and only the new compacted messages will be processed fresh.

        Args:
            summary: A text summary of the conversation so far.

        Returns:
            A new PromptBuilder with shared prefix and compacted messages.
        """
        # Create new builder with EXACT same prefix as parent
        child = PromptBuilder()

        # Copy Layer 0: System prompt + tools (exact same for cache sharing)
        child._layers[CacheBreakpoint.SYSTEM_TOOLS] = list(
            self.parent_builder._layers[CacheBreakpoint.SYSTEM_TOOLS]
        )
        child._tools = list(self.parent_builder._tools)
        child._tools_frozen = True
        child._model = self.parent_builder._model
        child._model_frozen = True

        # Copy Layer 1: Project context (exact same)
        child._layers[CacheBreakpoint.PROJECT_CONTEXT] = list(
            self.parent_builder._layers[CacheBreakpoint.PROJECT_CONTEXT]
        )

        # Copy Layer 2: Session context (exact same)
        child._layers[CacheBreakpoint.SESSION_CONTEXT] = list(
            self.parent_builder._layers[CacheBreakpoint.SESSION_CONTEXT]
        )

        # Layer 3: Compacted conversation
        compaction_msg = COMPACTION_INSTRUCTION.format(summary=summary)
        child.add_message(role="user", content=compaction_msg)

        # Verify cache compatibility
        if not child.is_cache_compatible(self.parent_builder):
            raise RuntimeError(
                "BUG: Compacted builder does not share parent's cache prefix. "
                "This means something in layers 0-2 was modified during compaction."
            )

        return child

    def estimate_compaction_savings(
        self,
        current_tokens: int,
        summary_tokens: int,
    ) -> dict[str, Any]:
        """Estimate the token and cost savings from compaction.

        Args:
            current_tokens: Current total input tokens.
            summary_tokens: Estimated tokens in the compacted summary.

        Returns:
            Dict with savings estimates.
        """
        # Estimate prefix tokens (cached layers 0-2)
        # These will be cache READ (cheap) in the compacted request
        prefix_estimate = current_tokens * 0.3  # Rough estimate: 30% is prefix

        return {
            "current_tokens": current_tokens,
            "estimated_after_compaction": int(prefix_estimate + summary_tokens),
            "tokens_saved": int(current_tokens - prefix_estimate - summary_tokens),
            "prefix_tokens_reused": int(prefix_estimate),
            "cache_hit_on_prefix": True,  # Guaranteed by design
        }


@dataclass
class SubagentPrefixSharer:
    """Shares prompt prefix between parent agent and sub-agents.

    When spawning sub-agents (e.g., Explore, Plan agents), the sub-agent
    should reuse the parent's cached prefix to avoid paying for a full
    cache rebuild.

    Usage:
        sharer = SubagentPrefixSharer(parent_builder)
        child_builder = sharer.create_child(
            task="Search for all files matching *.py",
            additional_context="Focus on the trainers/ directory"
        )
        # child_builder shares parent's cache prefix
    """

    parent_builder: PromptBuilder

    def create_child(
        self,
        task: str,
        additional_context: str = "",
        override_tools: list[dict[str, Any]] | None = None,
    ) -> PromptBuilder:
        """Create a child builder that shares the parent's prefix cache.

        IMPORTANT: If override_tools is provided, the child will NOT share
        the parent's cache (tools are part of the prefix). Only use this
        when the sub-agent genuinely needs different tools.

        Args:
            task: The task description for the sub-agent.
            additional_context: Extra context for the sub-agent.
            override_tools: Optional different tool set (breaks cache sharing).

        Returns:
            A new PromptBuilder for the sub-agent.
        """
        child = PromptBuilder()

        # Share Layer 0: System prompt
        child._layers[CacheBreakpoint.SYSTEM_TOOLS] = list(
            self.parent_builder._layers[CacheBreakpoint.SYSTEM_TOOLS]
        )
        child._model = self.parent_builder._model
        child._model_frozen = True

        if override_tools is not None:
            # WARNING: This breaks cache sharing
            child.set_tools(override_tools)
        else:
            # Share parent's tools — preserves cache
            child._tools = list(self.parent_builder._tools)
            child._tools_frozen = True

        # Share Layer 1: Project context
        child._layers[CacheBreakpoint.PROJECT_CONTEXT] = list(
            self.parent_builder._layers[CacheBreakpoint.PROJECT_CONTEXT]
        )

        # Layer 2: Task-specific session context
        child.add_session_context(f"Sub-agent task: {task}")
        if additional_context:
            child.add_session_context(additional_context)

        return child
