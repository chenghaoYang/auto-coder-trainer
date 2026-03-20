"""Seed management utilities for reproducible experiments."""

import os
import random


def set_all_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, PyTorch, and CUDA.

    Sets seeds across all common sources of randomness to improve
    reproducibility of training and evaluation runs.

    Note: ``PYTHONHASHSEED`` is set as an environment variable here for
    documentation purposes, but Python reads this variable only at interpreter
    startup. Setting it at runtime has **no effect** on hash randomization for
    the current process. To enforce hash seed reproducibility, export
    ``PYTHONHASHSEED=<seed>`` in your shell or launcher script before starting
    the Python interpreter.
    """
    # Python stdlib
    random.seed(seed)
    # NOTE: has no effect at runtime — see docstring above.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch (CPU + CUDA)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_seed_list(config: dict) -> list[int]:
    """Extract seed list from recipe eval config. Defaults to [42, 123, 456]."""
    return config.get("eval", {}).get("seeds", [42, 123, 456])
