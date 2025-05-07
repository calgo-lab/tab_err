from __future__ import annotations

from typing import TYPE_CHECKING

from tab_err._utils import check_data_emptiness, check_error_rate, set_column

if TYPE_CHECKING:
    import pandas as pd

    from tab_err import ErrorMechanism, ErrorType


def create_errors(
    data: pd.DataFrame, column: str | int, error_rate: float, error_mechanism: ErrorMechanism, error_type: ErrorType
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given column of a pandas DataFrame.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        column (str | int): The column to create errors in.
        error_rate (float): The rate at which errors will be created.
        error_mechanism (ErrorMechanism): The mechanism, controls the error distribution.
        error_type (ErrorType): The type of the error that will be distributed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - The first element is a copy of 'data' with errors.
            - The second element is the associated error mask.
    """
    check_error_rate(error_rate)
    check_data_emptiness(data)
    data_copy = data.copy()

    error_mask = error_mechanism.sample(data_copy, column, error_rate, error_mask=None)
    series = error_type.apply(data_copy, error_mask, column)
    set_column(data_copy, column, series)

    return data_copy, error_mask
