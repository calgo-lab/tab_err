from __future__ import annotations

from typing import TYPE_CHECKING

from error_generation.utils import set_column

if TYPE_CHECKING:
    import pandas as pd

    from error_generation.error_mechanism import ErrorMechanism
    from error_generation.error_type import ErrorType


def create_errors(table: pd.DataFrame, column: str | int, error_mechanism: ErrorMechanism, error_type: ErrorType) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given column of a pandas DataFrame.

    Args:
        table: The pandas DataFrame to create errors in.
        column: The column to create errors in.
        error_mechanism: The mechanism, controls the error distribution.
        error_type: The type of the error that will be distributed.

    Returns:
        A tuple of the original DataFrame and the error mask.
    """
    error_mask = error_mechanism.sample(table, seed=None)
    series = error_type.apply(table, error_mask, column)
    set_column(table, column, series)
    return table, error_mask
