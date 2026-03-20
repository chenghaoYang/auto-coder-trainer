"""SFT data loading and formatting utilities."""

from typing import Any


def load_trajectory_data(sources: list[dict[str, Any]], filters: list[dict[str, Any]] | None = None) -> Any:
    """Load trajectory data from specified sources and apply filters.

    Args:
        sources: List of dataset source specs [{name, path, mix_weight}]
        filters: Optional list of filter specs [{type, params}]

    Returns:
        Processed dataset ready for SFT training.

    TODO: Implement with datasets library.
    """
    raise NotImplementedError("Trajectory data loading not yet implemented")


def format_for_sft(dataset: Any, chat_template: str = "chatml") -> Any:
    """Format trajectory data as SFT training examples.

    Args:
        dataset: Raw trajectory dataset
        chat_template: Chat template format

    Returns:
        Formatted dataset with input/output pairs.

    TODO: Implement chat formatting.
    """
    raise NotImplementedError("SFT formatting not yet implemented")
