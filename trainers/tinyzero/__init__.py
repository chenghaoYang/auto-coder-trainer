"""TinyZero-compatible launcher helpers.

This package does not vendor TinyZero itself. Instead, it compiles our
Recipe IR into a TinyZero/veRL-compatible launch bundle so users can
prepare baseline SFT/RL jobs from the same training entry point.
"""

from trainers.tinyzero.launcher import (
    build_tinyzero_launcher_bundle,
    write_tinyzero_launcher_bundle,
)

__all__ = [
    "build_tinyzero_launcher_bundle",
    "write_tinyzero_launcher_bundle",
]
