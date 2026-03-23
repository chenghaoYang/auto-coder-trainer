"""SWE-Lego training backend — LLaMA-Factory based SFT for coding agents."""

from trainers.swe_lego.launcher import (
    build_swe_lego_launcher_bundle,
    write_swe_lego_launcher_bundle,
)

__all__ = [
    "build_swe_lego_launcher_bundle",
    "write_swe_lego_launcher_bundle",
]
