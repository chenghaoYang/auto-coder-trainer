"""Cache-safe prompt builder.

Constructs prompts in the optimal order for maximum cache hit rate:

    Layer 0: Global static system prompt + tool definitions  (cache_control)
    Layer 1: Project-level context (CLAUDE.md, repo structure) (cache_control)
    Layer 2: Session context (task description, plan)          (cache_control)
    Layer 3: Conversation messages                             (dynamic, no cache)

Any content added to an earlier layer invalidates all later layers' cache.
The builder enforces append-only semantics within each layer and prevents
modifications after finalization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class CacheBreakpoint(IntEnum):
    """Cache breakpoint layers, ordered by stability (most stable first).

    The integer value determines the position in the prompt prefix.
    Lower values = closer to the start = more stable = higher cache reuse.
    """
    SYSTEM_TOOLS = 0      # Global system prompt + tool definitions
    PROJECT_CONTEXT = 1   # CLAUDE.md, repo structure, project rules
    SESSION_CONTEXT = 2   # Task description, plan, session-specific state
    CONVERSATION = 3      # Messages — dynamic, not cached


@dataclass
class PromptBlock:
    """A block of content within a prompt layer."""
    content: str | list[dict[str, Any]]
    role: str = "system"
    cache_control: bool = False
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            raw = json.dumps(self.content, sort_keys=True) if isinstance(self.content, (list, dict)) else self.content
            self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class PromptBuilder:
    """Cache-safe prompt builder that enforces optimal ordering.

    Usage:
        builder = PromptBuilder()

        # Layer 0: Static system prompt + tools (set once, never change)
        builder.set_system_prompt("You are a coding agent...")
        builder.set_tools([...])  # Must be set once and frozen

        # Layer 1: Project context
        builder.add_project_context("CLAUDE.md content here...")

        # Layer 2: Session context
        builder.add_session_context("Current task: implement feature X")

        # Layer 3: Conversation messages (dynamic)
        builder.add_message(role="user", content="Hello")
        builder.add_message(role="assistant", content="Hi!")

        # Build the final prompt
        prompt = builder.build()

        # Check cache compatibility with a previous prompt
        builder.is_cache_compatible(previous_prompt)
    """

    _layers: dict[CacheBreakpoint, list[PromptBlock]] = field(default_factory=dict)
    _tools: list[dict[str, Any]] = field(default_factory=list)
    _tools_frozen: bool = False
    _model: str = ""
    _model_frozen: bool = False
    _finalized_layers: set[CacheBreakpoint] = field(default_factory=set)

    def __post_init__(self):
        for layer in CacheBreakpoint:
            self._layers.setdefault(layer, [])

    # ── Layer 0: System Prompt + Tools ──────────────────────────────

    def set_system_prompt(self, prompt: str) -> PromptBuilder:
        """Set the global static system prompt. Should be called once.

        WARNING: Changing this after the first API call invalidates ALL cache.
        """
        self._assert_not_finalized(CacheBreakpoint.SYSTEM_TOOLS)
        self._layers[CacheBreakpoint.SYSTEM_TOOLS] = [
            PromptBlock(content=prompt, role="system", cache_control=True)
        ]
        return self

    def set_tools(self, tools: list[dict[str, Any]]) -> PromptBuilder:
        """Set tool definitions. MUST be called exactly once per session.

        Rules:
        - Tool order must be deterministic (sorted by name)
        - Never add/remove tools mid-session
        - Use deferred loading (ToolSearch) for rarely-used tools

        Raises:
            RuntimeError: If tools have already been frozen.
        """
        if self._tools_frozen:
            raise RuntimeError(
                "Tools already frozen. Adding/removing tools mid-session "
                "destroys prompt cache. Use ToolSearch for dynamic tool loading."
            )
        # Sort tools by name for deterministic ordering
        self._tools = sorted(tools, key=lambda t: t.get("name", ""))
        self._tools_frozen = True
        return self

    def set_model(self, model: str) -> PromptBuilder:
        """Set the model for this session. MUST NOT change mid-session.

        Prompt cache is model-bound. Switching models (e.g., Opus → Haiku)
        forces a full cache rebuild.

        Raises:
            RuntimeError: If model has already been set to a different value.
        """
        if self._model_frozen and self._model != model:
            raise RuntimeError(
                f"Cannot switch model from '{self._model}' to '{model}' mid-session. "
                f"Prompt cache is model-bound — switching forces full cache rebuild."
            )
        self._model = model
        self._model_frozen = True
        return self

    # ── Layer 1: Project Context ────────────────────────────────────

    def add_project_context(self, content: str, label: str = "CLAUDE.md") -> PromptBuilder:
        """Add project-level context (CLAUDE.md, repo structure, etc.).

        This layer is cached across sessions within the same project.
        Append-only after finalization.
        """
        self._assert_not_finalized(CacheBreakpoint.PROJECT_CONTEXT)
        self._layers[CacheBreakpoint.PROJECT_CONTEXT].append(
            PromptBlock(content=content, role="system", cache_control=True)
        )
        return self

    # ── Layer 2: Session Context ────────────────────────────────────

    def add_session_context(self, content: str) -> PromptBuilder:
        """Add session-specific context (task description, plan, etc.).

        Cached within a single session. Append-only.
        """
        self._layers[CacheBreakpoint.SESSION_CONTEXT].append(
            PromptBlock(content=content, role="system", cache_control=True)
        )
        return self

    # ── Layer 3: Conversation Messages ──────────────────────────────

    def add_message(self, role: str, content: str | list[dict]) -> PromptBuilder:
        """Add a conversation message (user/assistant/tool_result).

        Messages are dynamic and never cached. Dynamic information
        (timestamps, file changes, status updates) should be injected here
        using <system-reminder> tags, NOT by modifying the system prompt.
        """
        self._layers[CacheBreakpoint.CONVERSATION].append(
            PromptBlock(content=content, role=role, cache_control=False)
        )
        return self

    def inject_dynamic_update(self, update: str) -> PromptBuilder:
        """Inject a dynamic update via <system-reminder> in the next user message.

        This is the CORRECT way to pass time-varying information (timestamps,
        file changes, status) without breaking the cache. Never modify the
        system prompt for dynamic information.
        """
        reminder = f"<system-reminder>\n{update}\n</system-reminder>"
        self._layers[CacheBreakpoint.CONVERSATION].append(
            PromptBlock(content=reminder, role="user", cache_control=False)
        )
        return self

    # ── Build & Validate ────────────────────────────────────────────

    def build(self) -> dict[str, Any]:
        """Build the final prompt payload for the Anthropic API.

        Returns a dict with 'system', 'tools', and 'messages' keys,
        with cache_control breakpoints set at optimal positions.
        """
        system_blocks = []
        messages = []

        # Layer 0: System prompt
        for block in self._layers[CacheBreakpoint.SYSTEM_TOOLS]:
            entry = {"type": "text", "text": block.content}
            if block.cache_control:
                entry["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(entry)

        # Layer 1: Project context
        for block in self._layers[CacheBreakpoint.PROJECT_CONTEXT]:
            entry = {"type": "text", "text": block.content}
            if block.cache_control:
                entry["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(entry)

        # Layer 2: Session context
        for block in self._layers[CacheBreakpoint.SESSION_CONTEXT]:
            entry = {"type": "text", "text": block.content}
            if block.cache_control:
                entry["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(entry)

        # Layer 3: Conversation messages
        for block in self._layers[CacheBreakpoint.CONVERSATION]:
            messages.append({"role": block.role, "content": block.content})

        return {
            "model": self._model,
            "system": system_blocks,
            "tools": self._tools,
            "messages": messages,
        }

    def compute_prefix_hash(self, up_to_layer: CacheBreakpoint = CacheBreakpoint.SESSION_CONTEXT) -> str:
        """Compute a hash of the prompt prefix up to the given layer.

        Used to check cache compatibility between requests.
        Two requests with the same prefix hash will share cache.
        """
        parts = []
        parts.append(self._model)
        parts.append(json.dumps(self._tools, sort_keys=True))
        for layer in CacheBreakpoint:
            if layer > up_to_layer:
                break
            for block in self._layers[layer]:
                parts.append(block.fingerprint)
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def is_cache_compatible(self, other: PromptBuilder, up_to_layer: CacheBreakpoint = CacheBreakpoint.SESSION_CONTEXT) -> bool:
        """Check if two builders share the same prefix (and thus cache).

        Returns True if the prefix up to `up_to_layer` is identical.
        """
        return self.compute_prefix_hash(up_to_layer) == other.compute_prefix_hash(up_to_layer)

    def finalize_layer(self, layer: CacheBreakpoint) -> None:
        """Mark a layer as finalized. No further modifications allowed."""
        self._finalized_layers.add(layer)

    # ── Internal ────────────────────────────────────────────────────

    def _assert_not_finalized(self, layer: CacheBreakpoint) -> None:
        if layer in self._finalized_layers:
            raise RuntimeError(
                f"Layer {layer.name} is finalized. Modifying it would invalidate "
                f"all downstream cache. Use inject_dynamic_update() for dynamic info."
            )


def validate_cache_safety(before: PromptBuilder, after: PromptBuilder) -> list[str]:
    """Validate that a prompt modification is cache-safe.

    Returns a list of violations (empty if safe).
    """
    violations = []

    # Check model consistency
    if before._model and after._model and before._model != after._model:
        violations.append(
            f"Model switched from '{before._model}' to '{after._model}'. "
            f"Cache is model-bound — this forces full rebuild."
        )

    # Check tool consistency
    if before._tools_frozen and after._tools_frozen:
        before_names = {t.get("name") for t in before._tools}
        after_names = {t.get("name") for t in after._tools}
        added = after_names - before_names
        removed = before_names - after_names
        if added:
            violations.append(f"Tools added mid-session: {added}. Use ToolSearch instead.")
        if removed:
            violations.append(f"Tools removed mid-session: {removed}. Keep toolset constant.")

    # Check prefix stability per layer
    for layer in [CacheBreakpoint.SYSTEM_TOOLS, CacheBreakpoint.PROJECT_CONTEXT]:
        before_fps = [b.fingerprint for b in before._layers.get(layer, [])]
        after_fps = [b.fingerprint for b in after._layers.get(layer, [])]
        if before_fps != after_fps[:len(before_fps)]:
            violations.append(
                f"Layer {layer.name} was modified (not append-only). "
                f"This invalidates all downstream cache."
            )

    return violations
