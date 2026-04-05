"""Shared test fixtures and import-skip helpers for optional dependencies."""

import pytest

# ---------------------------------------------------------------------------
# Optional dependency skip markers
# ---------------------------------------------------------------------------
# These allow test files to gracefully skip when optional packages are not
# installed.  Usage at the top of a test file:
#
#   pytest.importorskip("yaml")          # skips the whole module
#   pytest.importorskip("swebench")
#
# Or per-test / per-class:
#   @pytest.mark.skipif(not _has_yaml, reason="requires pyyaml")
# ---------------------------------------------------------------------------

# Check optional deps once at collection time so test modules can decide
# whether to skip without importing heavy packages.
try:
    import yaml  # noqa: F401

    _has_yaml = True
except ImportError:
    _has_yaml = False

try:
    import swebench  # noqa: F401

    _has_swebench = True
except ImportError:
    _has_swebench = False

try:
    import trl  # noqa: F401

    _has_trl = True
except ImportError:
    _has_trl = False

try:
    import verl  # noqa: F401

    _has_verl = True
except ImportError:
    _has_verl = False


# Expose as boolean pytest markers so individual tests can use skipif.
def pytest_configure(config):
    config.addinivalue_line("markers", "needs_yaml: test requires pyyaml")
    config.addinivalue_line("markers", "needs_swebench: test requires swebench")
    config.addinivalue_line("markers", "needs_trl: test requires trl")
    config.addinivalue_line("markers", "needs_verl: test requires verl")
