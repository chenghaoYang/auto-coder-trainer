"""Prompt Cache module — cache-safe prompt construction for long-running agent systems.

"Cache Rules Everything Around Me"

Core principle: Anthropic's prompt caching uses prefix matching. The API caches
content from the start of the request to each cache_control breakpoint. Therefore,
prompt ORDER and STABILITY are critical for cache hits.

This module enforces the optimal prompt layout:

    [Static System Prompt + Tools]     ← cached across ALL sessions
    [Project-level CLAUDE.md]          ← cached across project sessions
    [Session context]                  ← cached within session
    [Conversation messages]            ← dynamic, never cached

Any mutation to earlier layers invalidates ALL downstream cache.

Rules:
1. Static content first, dynamic content last
2. Never inject timestamps/random values into system prompt
3. Never reorder or modify tool definitions mid-session
4. Use <system-reminder> in messages for dynamic updates, not system prompt edits
5. Model/tool switching destroys cache — keep toolset constant
6. Compaction must share parent's prefix exactly
"""

from prompt_cache.builder import PromptBuilder, CacheBreakpoint
from prompt_cache.monitor import CacheMonitor
from prompt_cache.compaction import CacheSafeCompactor

__all__ = [
    "PromptBuilder",
    "CacheBreakpoint",
    "CacheMonitor",
    "CacheSafeCompactor",
]
