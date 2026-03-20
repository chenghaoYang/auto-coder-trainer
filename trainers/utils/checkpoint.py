"""Checkpoint management utilities."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_checkpoint(
    model_path: str | Path,
    recipe_id: str,
    metrics: dict[str, Any],
    checkpoint_dir: str | Path,
) -> str:
    """Save a training checkpoint with metadata.

    Creates a timestamped subdirectory inside *checkpoint_dir* containing:
      - metadata.json  (recipe_id, metrics, timestamp)
      - model/         symlink or copy of the model artefacts

    Returns the path to the created checkpoint directory.
    """
    model_path = Path(model_path)
    checkpoint_dir = Path(checkpoint_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ckpt_name = f"ckpt-{recipe_id}-{timestamp}"
    ckpt_path = checkpoint_dir / ckpt_name
    ckpt_path.mkdir(parents=True)

    # Write metadata
    metadata = {
        "recipe_id": recipe_id,
        "metrics": metrics,
        "timestamp": timestamp,
        "model_source": str(model_path),
    }
    (ckpt_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Symlink (preferred) or copy the model artefacts
    dest = ckpt_path / "model"
    try:
        dest.symlink_to(model_path.resolve())
    except OSError:
        # Fallback to copy if symlinks are unsupported
        if model_path.is_dir():
            shutil.copytree(model_path, dest)
        else:
            shutil.copy2(model_path, dest)

    return str(ckpt_path)


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a training checkpoint.

    Returns a dict with the metadata fields plus a ``path`` key pointing
    to the checkpoint directory.
    """
    checkpoint_path = Path(checkpoint_path)
    meta_file = checkpoint_path / "metadata.json"

    if not meta_file.exists():
        raise FileNotFoundError(f"No metadata.json found in {checkpoint_path}")

    metadata = json.loads(meta_file.read_text())
    metadata["path"] = str(checkpoint_path)
    return metadata


def list_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """List all checkpoint subdirectories, sorted by timestamp (oldest first).

    Only directories that contain a ``metadata.json`` file are included.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints: list[tuple[str, Path]] = []
    for child in checkpoint_dir.iterdir():
        if child.is_dir() and (child / "metadata.json").exists():
            try:
                meta = json.loads((child / "metadata.json").read_text())
                ts = meta.get("timestamp", "")
            except (json.JSONDecodeError, OSError):
                ts = ""
            checkpoints.append((ts, child))

    checkpoints.sort(key=lambda t: t[0])
    return [path for _, path in checkpoints]
