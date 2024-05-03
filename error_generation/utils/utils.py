from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from error_generation.error_mechanism import ErrorMechanism
    from error_generation.error_type import ErrorType


@dataclass
class ErrorConfig:
    """Parameters that describe the error and its distribution.

    Args:
        error_rate: The rate at which the error occurs.
        mechanism: The mechanism that generates the error.
        error_type: The type of error that is generated.
        condition_to_column: The column that determines whether the error is generated.
    """

    error_rate: float
    mechanism: ErrorMechanism
    error_type: ErrorType
    condition_to_column: int | str | None = None


def get_column(table: pd.DataFrame, column: int | str) -> pd.Series:
    """Selects a column from a dataframe and returns it as a series."""
    if isinstance(column, int):
        series = table.iloc[:, column]
    elif isinstance(column, str):
        series = table.loc[:, column]
    else:
        msg = f"Column must be an int or str, not {type(column)}"
        raise TypeError(msg)
    return series


def set_column(table: pd.DataFrame, column: int | str, series: pd.Series) -> pd.Series:
    """Replaces a column in a dataframe with a series. Mutates table."""
    if isinstance(column, int):
        table.iloc[:, column] = series
    elif isinstance(column, str):
        table.loc[:, column] = series
    return table
