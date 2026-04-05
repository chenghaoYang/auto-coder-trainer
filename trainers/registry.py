"""Trainer Registry — maps (trainer_type, backend) pairs to trainer classes.

Provides a registration-based dispatch mechanism so that adding a new backend
or trainer type requires only a ``register()`` call rather than modifying
core dispatch logic in ``cli/train.py``.

Usage::

    from trainers.registry import get_trainer_class, register

    # Register a custom trainer
    register("my_algo", "my_backend", MyTrainerClass)

    # Look up a trainer class
    cls = get_trainer_class("sft", "trl")  # -> SFTTrainer
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainers.base import BaseTrainer

# Maps (trainer_type, backend) -> trainer class.
# Entries with backend=None act as fallback for any backend.
_REGISTRY: dict[tuple[str, str | None], type[BaseTrainer]] = {}


def register(
    trainer_type: str,
    backend: str | None,
    cls: type[BaseTrainer],
) -> None:
    """Register a trainer class for a (trainer_type, backend) pair.

    Args:
        trainer_type: The recipe trainer type (e.g. "sft", "rl", "grpo", "distill").
        backend: The backend identifier (e.g. "trl", "verl"), or ``None`` to
                 register as the default for *any* backend with that trainer type.
        cls: The trainer class (must be a subclass of ``BaseTrainer``).
    """
    _REGISTRY[(trainer_type, backend)] = cls


def get_trainer_class(
    trainer_type: str,
    backend: str,
) -> type[BaseTrainer] | None:
    """Look up the trainer class for a (trainer_type, backend) pair.

    Resolution order:
        1. Exact match on (trainer_type, backend)
        2. Fallback match on (trainer_type, None)
        3. None if no match found

    Returns:
        The trainer class, or ``None`` if no matching registration exists.
    """
    # Exact match
    cls = _REGISTRY.get((trainer_type, backend))
    if cls is not None:
        return cls
    # Fallback: trainer_type with any backend
    return _REGISTRY.get((trainer_type, None))


def list_registered() -> list[dict[str, str | None]]:
    """Return a list of all registered (trainer_type, backend) pairs."""
    return [
        {"trainer_type": tt, "backend": be}
        for tt, be in sorted(_REGISTRY.keys(), key=lambda k: (k[0], k[1] or ""))
    ]


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------
# These are deferred imports to avoid circular dependencies and to keep
# startup fast when only a subset of trainers is needed.


def _register_builtins() -> None:
    """Register all built-in trainer classes."""
    try:
        from trainers.sft.trainer import SFTTrainer
        register("sft", None, SFTTrainer)      # SFT works with any backend
        register("sft", "trl", SFTTrainer)
    except ImportError:
        pass

    try:
        from trainers.rl.trainer import RLTrainer
        register("rl", None, RLTrainer)
        register("rl", "verl", RLTrainer)
        register("rl", "trl", RLTrainer)
        register("grpo", None, RLTrainer)
        register("grpo", "verl", RLTrainer)
        register("grpo", "trl", RLTrainer)
    except ImportError:
        pass

    try:
        from trainers.distill.trainer import DistillTrainer
        register("distill", None, DistillTrainer)
        register("distill", "trl", DistillTrainer)
        register("dpo", None, DistillTrainer)
        register("dpo", "trl", DistillTrainer)
    except ImportError:
        pass


_register_builtins()
