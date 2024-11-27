from __future__ import annotations

from typing import TYPE_CHECKING

from tab_err.utils import set_column

if TYPE_CHECKING:
    import pandas as pd

    from tab_err.error_mechanism import ErrorMechanism
    from tab_err.error_type import ErrorType


def create_errors(
    table: pd.DataFrame, column: str | int, error_rate: float, error_mechanism: ErrorMechanism, error_type: ErrorType
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given column of a pandas DataFrame.

    Args:
        table: The pandas DataFrame to create errors in.
        column: The column to create errors in.
        error_rate: The rate at which errors will be created.
        error_mechanism: The mechanism, controls the error distribution.
        error_type: The type of the error that will be distributed.

    Returns:
        A tuple of a copy of the table with errors, and the error mask.
    """
    table_copy = table.copy()

    error_mask = error_mechanism.sample(table_copy, column, error_rate, error_mask=None)
    series = error_type.apply(table_copy, error_mask, column)
    set_column(table_copy, column, series)
    return table_copy, error_mask
