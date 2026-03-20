"""Prompt cache hit rate monitor.

"Monitor cache hit rate like you monitor uptime."

Tracks cache_read_input_tokens vs cache_creation_input_tokens from
Anthropic API responses to compute hit rates. Triggers alerts when
hit rate drops below threshold.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CacheEvent:
    """A single API request's cache statistics."""
    timestamp: float
    cache_creation_tokens: int
    cache_read_tokens: int
    input_tokens: int
    model: str
    prefix_hash: str = ""

    @property
    def hit_rate(self) -> float:
        """Cache hit rate for this individual request."""
        total = self.cache_creation_tokens + self.cache_read_tokens
        if total == 0:
            return 0.0
        return self.cache_read_tokens / total

    @property
    def cost_savings_ratio(self) -> float:
        """Estimated cost savings from caching.

        cache_read tokens cost 0.1x of regular input tokens.
        cache_creation tokens cost 1.25x of regular input tokens.
        """
        if self.input_tokens == 0:
            return 0.0
        uncached_cost = self.input_tokens  # baseline: 1x per token
        cached_cost = (
            (self.input_tokens - self.cache_creation_tokens - self.cache_read_tokens) * 1.0
            + self.cache_creation_tokens * 1.25
            + self.cache_read_tokens * 0.1
        )
        if uncached_cost == 0:
            return 0.0
        return 1.0 - (cached_cost / uncached_cost)


@dataclass
class CacheMonitor:
    """Monitor and track prompt cache hit rates across API requests.

    Usage:
        monitor = CacheMonitor(alert_threshold=0.8)

        # After each API call, record the cache stats from the response
        monitor.record(response.usage)

        # Check health
        if not monitor.is_healthy():
            print(f"ALERT: Cache hit rate {monitor.rolling_hit_rate():.1%} "
                  f"below threshold {monitor.alert_threshold:.1%}")

        # Get stats
        stats = monitor.get_stats()
    """

    alert_threshold: float = 0.8  # Alert if hit rate drops below this
    rolling_window: int = 20      # Number of recent events for rolling stats
    persist_path: Path | None = None  # Optional path to persist metrics

    _events: list[CacheEvent] = field(default_factory=list)
    _alerts: list[dict[str, Any]] = field(default_factory=list)

    def record(self, usage: dict[str, Any], model: str = "", prefix_hash: str = "") -> CacheEvent:
        """Record cache statistics from an API response's usage dict.

        Expected usage dict format (from Anthropic API):
            {
                "input_tokens": 2577,
                "output_tokens": 325,
                "cache_creation_input_tokens": 2048,
                "cache_read_input_tokens": 529
            }
        """
        event = CacheEvent(
            timestamp=time.time(),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            input_tokens=usage.get("input_tokens", 0),
            model=model,
            prefix_hash=prefix_hash,
        )
        self._events.append(event)

        # Check alert condition on rolling window
        rate = self.rolling_hit_rate()
        if len(self._events) >= 3 and rate < self.alert_threshold:
            alert = {
                "timestamp": event.timestamp,
                "hit_rate": rate,
                "threshold": self.alert_threshold,
                "severity": "SEV2" if rate < 0.5 else "SEV3",
                "message": (
                    f"Cache hit rate {rate:.1%} below threshold {self.alert_threshold:.1%}. "
                    f"Possible cause: system prompt or tool definition changed."
                ),
            }
            self._alerts.append(alert)

        if self.persist_path:
            self._persist()

        return event

    def rolling_hit_rate(self) -> float:
        """Compute rolling average cache hit rate over recent events."""
        window = self._events[-self.rolling_window:]
        if not window:
            return 0.0
        total_creation = sum(e.cache_creation_tokens for e in window)
        total_read = sum(e.cache_read_tokens for e in window)
        total = total_creation + total_read
        if total == 0:
            return 0.0
        return total_read / total

    def lifetime_hit_rate(self) -> float:
        """Compute overall cache hit rate across all events."""
        if not self._events:
            return 0.0
        total_creation = sum(e.cache_creation_tokens for e in self._events)
        total_read = sum(e.cache_read_tokens for e in self._events)
        total = total_creation + total_read
        if total == 0:
            return 0.0
        return total_read / total

    def total_cost_savings_ratio(self) -> float:
        """Compute overall cost savings from caching."""
        if not self._events:
            return 0.0
        total_input = sum(e.input_tokens for e in self._events)
        total_creation = sum(e.cache_creation_tokens for e in self._events)
        total_read = sum(e.cache_read_tokens for e in self._events)
        if total_input == 0:
            return 0.0
        uncached_cost = total_input
        cached_cost = (
            (total_input - total_creation - total_read) * 1.0
            + total_creation * 1.25
            + total_read * 0.1
        )
        return 1.0 - (cached_cost / uncached_cost)

    def is_healthy(self) -> bool:
        """Check if cache hit rate is above threshold."""
        if len(self._events) < 3:
            return True  # Not enough data to judge
        return self.rolling_hit_rate() >= self.alert_threshold

    def get_recent_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent cache health alerts."""
        return self._alerts[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "total_requests": len(self._events),
            "rolling_hit_rate": self.rolling_hit_rate(),
            "lifetime_hit_rate": self.lifetime_hit_rate(),
            "cost_savings_ratio": self.total_cost_savings_ratio(),
            "is_healthy": self.is_healthy(),
            "alert_threshold": self.alert_threshold,
            "active_alerts": len([a for a in self._alerts if a["timestamp"] > time.time() - 3600]),
            "total_cache_read_tokens": sum(e.cache_read_tokens for e in self._events),
            "total_cache_creation_tokens": sum(e.cache_creation_tokens for e in self._events),
        }

    def diagnose_cache_miss(self) -> list[str]:
        """Analyze recent events to diagnose why cache misses are occurring.

        Returns list of likely causes.
        """
        if not self._events:
            return ["No events recorded yet."]

        diagnoses = []
        recent = self._events[-5:]

        # Check if all recent events are cache misses (creation >> read)
        all_miss = all(e.cache_creation_tokens > e.cache_read_tokens for e in recent)
        if all_miss:
            diagnoses.append(
                "All recent requests are cache misses. Likely causes:\n"
                "  - System prompt was modified\n"
                "  - Tool definitions were added/removed/reordered\n"
                "  - Model was switched\n"
                "  - Session was restarted without prefix sharing"
            )

        # Check for model switches
        models = set(e.model for e in recent if e.model)
        if len(models) > 1:
            diagnoses.append(
                f"Multiple models detected in recent requests: {models}. "
                f"Cache is model-bound — switching forces full rebuild."
            )

        # Check for prefix hash changes
        hashes = [e.prefix_hash for e in recent if e.prefix_hash]
        if len(set(hashes)) > 1:
            diagnoses.append(
                "Prefix hash changed between requests. "
                "Something in the static layers (system prompt, tools, project context) was modified."
            )

        if not diagnoses:
            diagnoses.append("No obvious issues detected. Cache may be warming up.")

        return diagnoses

    def _persist(self) -> None:
        """Persist metrics to disk."""
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        stats = self.get_stats()
        stats["events"] = [
            {
                "timestamp": e.timestamp,
                "hit_rate": e.hit_rate,
                "cache_creation": e.cache_creation_tokens,
                "cache_read": e.cache_read_tokens,
                "model": e.model,
            }
            for e in self._events[-100:]  # Keep last 100 events
        ]
        with open(self.persist_path, "w") as f:
            json.dump(stats, f, indent=2)
