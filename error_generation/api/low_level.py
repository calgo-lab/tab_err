from __future__ import annotations

from typing import TYPE_CHECKING

from error_generation.utils import ErrorConfig, set_column

if TYPE_CHECKING:
    import pandas as pd


def create_errors(
    table: pd.DataFrame,
    column: str | int,
    error_config: ErrorConfig | dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given column of a pandas DataFrame.

    Args:
        table: The pandas DataFrame to create errors in.
        column: The column to create errors in.
        error_config: The error configuration to use.

    Returns:
        A tuple of the original DataFrame and the error mask.
    """
    if isinstance(error_config, dict):
        error_config = ErrorConfig(**error_config)

    error_mask = error_config.mechanism.sample(table, error_config.error_rate, error_config.condition_to_column, seed=None)
    series = error_config.error_type.apply(table, error_mask, column)
    set_column(table, column, series)
    return table, error_mask
